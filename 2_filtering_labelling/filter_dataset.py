#!/usr/bin/env python3
"""filter_dataset.py

Triage images into clean / no_vehicle / no_plate WITHOUT assigning front/back
(so it works on a mixed front+back batch like dataset3).

For each image:
  * yolo26 detects vehicles -> if none pass conf/min-area  -> no_vehicle/
  * else ANPR checks for a visible plate on the full image:
        no plate >= --plate-conf                            -> no_plate/
        plate present                                       -> clean/

Images are MOVED into <out>/{clean,no_vehicle,no_plate}. A CSV log records the
decision + reason per image. Orientation labeling (front/back) happens later,
only on the clean/ set.

Run:
    python filter_dataset.py --images dataset3_images \
        --weights /mnt/datadisk/avani/front_back/yolo26x.pt
"""

from __future__ import annotations

import argparse
import csv
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from ultralytics import YOLO

# COCO vehicle class ids (yolo26x is COCO-trained).
VEHICLE_CLASS_IDS = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ANPR service
ANPR_API_URL = "http://216.48.185.18:5000/upload"
ANPR_FILE_FIELD = "file"
ANPR_TIMEOUT = 30
ANPR_MAX_RETRIES = 3
ANPR_RESULTS_KEY = "plates"
ANPR_CONF_KEY = "confidence"

_gpu_lock = threading.Lock()
_csv_lock = threading.Lock()


def iter_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_session(pool: int) -> requests.Session:
    s = requests.Session()
    a = HTTPAdapter(pool_connections=pool, pool_maxsize=pool * 2, max_retries=0)
    s.mount("http://", a)
    s.mount("https://", a)
    return s


def anpr_has_plate(session: requests.Session, img_path: Path,
                   conf_thr: float) -> Optional[bool]:
    """True/False if a plate >= conf_thr is present; None if the API failed."""
    try:
        blob = img_path.read_bytes()
    except OSError:
        return None
    for attempt in range(1, ANPR_MAX_RETRIES + 1):
        try:
            r = session.post(ANPR_API_URL,
                             files={ANPR_FILE_FIELD: (img_path.name, blob, "image/jpeg")},
                             timeout=ANPR_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError):
            if attempt < ANPR_MAX_RETRIES:
                time.sleep(attempt)
            continue
        plates = data.get(ANPR_RESULTS_KEY) or []
        best = 0.0
        if isinstance(plates, list):
            for p in plates:
                if isinstance(p, dict):
                    try:
                        best = max(best, float(p.get(ANPR_CONF_KEY, 0.0) or 0.0))
                    except (TypeError, ValueError):
                        pass
        return best >= conf_thr
    return None


def has_vehicle(model: YOLO, img_path: Path, conf: float,
                min_area_frac: float, device: str) -> Optional[bool]:
    """True if >=1 vehicle box passes conf + min-area; None if unreadable."""
    im = cv2.imread(str(img_path))
    if im is None:
        return None
    h, w = im.shape[:2]
    with _gpu_lock:
        res = model.predict(source=im, conf=conf, classes=list(VEHICLE_CLASS_IDS),
                            device=device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return False
    for b in res.boxes.xyxy.cpu().numpy():
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if area / (w * h) >= min_area_frac:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Triage: clean / no_vehicle / no_plate")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--weights", type=Path,
                    default=Path("/mnt/datadisk/avani/front_back/yolo26x.pt"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.4, help="vehicle det conf")
    ap.add_argument("--min-area-frac", type=float, default=0.02)
    ap.add_argument("--plate-conf", type=float, default=0.45)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = args.out or (args.images.parent / f"{args.images.name}_filtered")
    dirs = {k: out / k for k in ("clean", "no_vehicle", "no_plate")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    log_path = out / "filter_log.csv"

    images = iter_images(args.images)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images in {args.images}")

    print(f"Loading yolo26: {args.weights}")
    model = YOLO(str(args.weights))
    session = build_session(args.workers)
    device = args.device

    counts = {"clean": 0, "no_vehicle": 0, "no_plate": 0, "api_fail": 0, "unreadable": 0}
    log_f = log_path.open("w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow(["image", "decision"])

    def work(p: Path) -> Tuple[Path, str]:
        veh = has_vehicle(model, p, args.conf, args.min_area_frac, device)
        if veh is None:
            return p, "unreadable"
        if not veh:
            return p, "no_vehicle"
        plate = anpr_has_plate(session, p, args.plate_conf)
        if plate is None:
            return p, "api_fail"
        return p, ("clean" if plate else "no_plate")

    progress = tqdm(total=len(images), desc="Filtering", unit="img")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, p): p for p in images}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    _, decision = fut.result()
                except Exception:
                    decision = "api_fail"
                counts[decision] = counts.get(decision, 0) + 1
                if decision in dirs:
                    try:
                        shutil.move(str(p), str(dirs[decision] / p.name))
                    except (OSError, shutil.Error):
                        pass
                with _csv_lock:
                    log_w.writerow([p.name, decision])
                progress.update(1)
    finally:
        progress.close()
        session.close()
        log_f.close()

    print("\n==================== FILTER SUMMARY ====================")
    print(f"clean (vehicle + plate) : {counts['clean']}")
    print(f"no_vehicle              : {counts['no_vehicle']}")
    print(f"no_plate                : {counts['no_plate']}")
    print(f"unreadable              : {counts['unreadable']}")
    print(f"ANPR failures (left in place): {counts['api_fail']}")
    print(f"Output -> {out}  (clean/, no_vehicle/, no_plate/)")
    print(f"Log    -> {log_path}")
    print("=======================================================")


if __name__ == "__main__":
    main()
