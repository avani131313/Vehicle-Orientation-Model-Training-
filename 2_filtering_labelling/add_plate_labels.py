#!/usr/bin/env python3
"""add_plate_labels.py

Add a new class-4 'reg_plate' box to every merged label file by running each
image through the ANPR service and appending the plate bounding box(es).

For each image:
  * POST it to the ANPR service, which returns plates with a pixel bbox and a
    detection confidence.
  * Keep plates with confidence >= PLATE_CONF_THRESHOLD.
  * Convert each plate bbox to normalized YOLO xywh and append '4 cx cy w h'
    to that image's label file (existing vehicle boxes 0-3 are preserved).

Idempotent: existing class-4 lines are stripped before new ones are written, so
re-running never double-adds. Resumable via a checkpoint of processed images.
Threaded for throughput; each worker owns its own label file (no write races).

Run:
    python add_plate_labels.py \
        --images /mnt/datadisk/avani/front_back/main/Dataset_merged/images \
        --labels /mnt/datadisk/avani/front_back/main/Dataset_merged/labels
"""

from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ANPR service (returns plate bbox + confidence on a full image)
# ---------------------------------------------------------------------------
ANPR_API_URL = "http://216.48.185.18:5000/upload"   # set to your working URL
ANPR_FILE_FIELD = "file"
ANPR_TIMEOUT = 30
ANPR_MAX_RETRIES = 3
ANPR_RETRY_BACKOFF = 1.0
ANPR_RESULTS_KEY = "plates"
ANPR_CONF_KEY = "confidence"
ANPR_BBOX_KEY = "bbox"            # [x1, y1, x2, y2] in image pixels
PLATE_CONF_THRESHOLD = 0.65
PLATE_CLASS_ID = 4               # reg_plate

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

_print_lock = threading.Lock()


def build_session(pool: int) -> requests.Session:
    s = requests.Session()
    a = HTTPAdapter(pool_connections=pool, pool_maxsize=pool * 2, max_retries=0)
    s.mount("http://", a)
    s.mount("https://", a)
    return s


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def anpr_plates(session: requests.Session, img_path: Path,
                conf_thr: float = PLATE_CONF_THRESHOLD
                ) -> Tuple[Optional[List[Tuple[float, float, float, float]]], str]:
    """Return ([(x1,y1,x2,y2), ...], status). None on failure (never raises)."""
    try:
        blob = img_path.read_bytes()
    except OSError:
        return None, "read_failed"

    last = "unknown"
    for attempt in range(1, ANPR_MAX_RETRIES + 1):
        try:
            resp = session.post(
                ANPR_API_URL,
                files={ANPR_FILE_FIELD: (img_path.name, blob, "image/jpeg")},
                timeout=ANPR_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            last = "timeout"
        except requests.RequestException as exc:
            last = f"request_error:{exc.__class__.__name__}"
        except ValueError:
            last = "invalid_json"
        else:
            plates = data.get(ANPR_RESULTS_KEY) or []
            boxes: List[Tuple[float, float, float, float]] = []
            if isinstance(plates, list):
                for p in plates:
                    if not isinstance(p, dict):
                        continue
                    try:
                        conf = float(p.get(ANPR_CONF_KEY, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        conf = 0.0
                    bbox = p.get(ANPR_BBOX_KEY)
                    if conf >= conf_thr and isinstance(bbox, (list, tuple)) \
                            and len(bbox) >= 4:
                        try:
                            boxes.append((float(bbox[0]), float(bbox[1]),
                                          float(bbox[2]), float(bbox[3])))
                        except (TypeError, ValueError):
                            continue
            return boxes, "ok"
        if attempt < ANPR_MAX_RETRIES:
            time.sleep(ANPR_RETRY_BACKOFF * attempt)
    return None, last


def image_size(img_path: Path) -> Optional[Tuple[int, int]]:
    """Return (w, h) via Pillow without a full decode, or None."""
    try:
        with Image.open(img_path) as im:
            return im.size  # (w, h)
    except (OSError, ValueError):
        return None


def write_plate_lines(label_path: Path,
                      plate_lines: List[str]) -> None:
    """Rewrite a label file keeping non-plate lines and setting plate lines.

    Existing class-4 lines are removed first (idempotent), vehicle lines kept.
    """
    kept: List[str] = []
    if label_path.exists():
        try:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if cid == PLATE_CLASS_ID:
                    continue  # drop old plate lines
                kept.append(line.strip())
        except OSError:
            pass
    out = kept + plate_lines
    label_path.write_text(("\n".join(out) + "\n") if out else "", encoding="utf-8")


def process_one(session: requests.Session, img_path: Path, labels_dir: Path,
                conf_thr: float = PLATE_CONF_THRESHOLD) -> str:
    """Add plate boxes for one image. Returns a status string."""
    boxes, status = anpr_plates(session, img_path, conf_thr)
    if boxes is None:
        return f"api_fail:{status}"

    label_path = labels_dir / (img_path.stem + ".txt")
    if not boxes:
        # No confident plate: still strip any stale plate lines, keep vehicles.
        write_plate_lines(label_path, [])
        return "no_plate"

    size = image_size(img_path)
    if size is None:
        return "size_fail"
    w, h = size

    plate_lines: List[str] = []
    for (x1, y1, x2, y2) in boxes:
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        cx = min(1.0, max(0.0, cx)); cy = min(1.0, max(0.0, cy))
        bw = min(1.0, max(0.0, bw)); bh = min(1.0, max(0.0, bh))
        if bw <= 0 or bh <= 0:
            continue
        plate_lines.append(
            f"{PLATE_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    write_plate_lines(label_path, plate_lines)
    return "plated" if plate_lines else "no_plate"


class NameCheckpoint:
    """Thread-safe append-only set of processed image names, for resume."""

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._h = path.open("a", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> set:
        names: set = set()
        if path.exists():
            try:
                for line in path.open("r", encoding="utf-8"):
                    s = line.strip()
                    if s:
                        names.add(s)
            except OSError:
                pass
        return names

    def add(self, name: str) -> None:
        with self._lock:
            self._h.write(name + "\n")
            self._h.flush()

    def close(self) -> None:
        with self._lock:
            if not self._h.closed:
                self._h.flush()
                self._h.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Append reg_plate (class 4) boxes")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--plate-conf", type=float, default=PLATE_CONF_THRESHOLD,
                    help=f"Plate detection confidence cutoff "
                         f"(default {PLATE_CONF_THRESHOLD})")
    args = ap.parse_args()

    images = iter_images(args.images)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    ckpt_path = args.labels.parent / "plate_labels_checkpoint.txt"
    done = NameCheckpoint.load(ckpt_path)
    pending = [p for p in images if p.name not in done]
    if done:
        print(f"Resuming: {len(done)} already processed, {len(pending)} remaining")

    session = build_session(args.workers)
    ckpt = NameCheckpoint(ckpt_path)
    counts = {"plated": 0, "no_plate": 0, "size_fail": 0}
    api_fail = 0

    progress = tqdm(total=len(pending), desc="Plate-labeling", unit="img")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_one, session, p, args.labels,
                              args.plate_conf): p
                    for p in pending}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    status = fut.result()
                except Exception:
                    status = "api_fail:worker_exception"
                if status.startswith("api_fail"):
                    api_fail += 1
                    # do NOT checkpoint failures, so a re-run retries them
                else:
                    counts[status] = counts.get(status, 0) + 1
                    ckpt.add(p.name)
                progress.update(1)
    finally:
        progress.close()
        session.close()
        ckpt.close()

    print("\n==================== PLATE-LABELING SUMMARY ====================")
    print(f"Images with plate box added : {counts.get('plated', 0)}")
    print(f"No confident plate           : {counts.get('no_plate', 0)}")
    print(f"Size read failures           : {counts.get('size_fail', 0)}")
    print(f"ANPR failures (will retry)   : {api_fail}")
    print(f"Labels dir                   : {args.labels}")
    print("===============================================================")


if __name__ == "__main__":
    main()
