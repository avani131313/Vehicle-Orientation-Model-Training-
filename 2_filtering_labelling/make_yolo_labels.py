#!/usr/bin/env python3
"""
Generate YOLO .txt labels for the NIGHT (rear-only) image set.

All images in this folder are BACK views, so there is no front/back
classification step (Gemini is removed). YOLO only:
  1. detects vehicles and decides bike vs non_bike,
  2. every kept box is labelled as the corresponding *_back class:
        bike     -> bike_back      (id 1)
        non_bike -> non_bike_back  (id 3)

Two rejection checks move an image into <dataset-dir>/deleted_images/
(never deleted, just copied aside):
  * NO BOX     - YOLO makes no proper bounding box (no detection, all boxes
                 too small, empty crop, too many vehicles, or unreadable image).
  * NO PLATE   - the ANPR service does not see a visible number plate on a
                 kept vehicle crop.

Uses:
  - detect_vehicles_yolo.py -> vehicle boxes (yolo26x.pt)
  - the ANPR HTTP service    -> number-plate visibility check

Place this file in the SAME folder as detect_vehicles_yolo.py, with
yolo26x.pt in the parent folder, and make sure the ANPR service is running.
Then:

    python make_yolo_labels.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from ultralytics import YOLO

from detect_vehicles_yolo import IMAGE_EXTS, VEHICLE_CLASS_IDS, list_images


def _json_default(obj):
    """JSON fallback: convert numpy scalars/arrays to native Python types.

    Under numpy 2 (NEP 50) many computed values stay numpy float32, which the
    stdlib json encoder rejects. This makes every log row serializable.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

# ---------------------------------------------------------------------------
# Class mapping (all rear -> only the *_back classes are ever emitted)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["bike_front", "bike_back", "non_bike_front", "non_bike_back"]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

BIKE_COCO = {"bicycle", "motorcycle"}

# BGR colors per class for debug overlays
CLASS_COLORS = {
    "bike_front": (0, 200, 0),
    "bike_back": (0, 140, 255),
    "non_bike_front": (255, 180, 0),
    "non_bike_back": (0, 0, 220),
}

# ---------------------------------------------------------------------------
# ANPR service (number-plate visibility check)
# ---------------------------------------------------------------------------
ANPR_API_URL = "http://216.48.185.18:5000/upload"
ANPR_FILE_FIELD = "file"
ANPR_TIMEOUT = 30
ANPR_MAX_RETRIES = 3
ANPR_RETRY_BACKOFF = 1.0
# Response schema: {"plates": [{"confidence": .., "ocr_confidence": .., "plate": ..}]}
ANPR_RESULTS_KEY = "plates"
ANPR_CONF_KEY = "confidence"
# A plate is "visible" if at least one detection meets this confidence.
PLATE_CONF_THRESHOLD = 0.65


def is_bike(coco_name: str) -> bool:
    return coco_name in BIKE_COCO


def build_anpr_session(pool_size: int) -> requests.Session:
    """Create a pooled requests session for the ANPR service."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=pool_size, pool_maxsize=pool_size * 2, max_retries=0
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def anpr_plate_visible(session: requests.Session, crop_bgr) -> tuple[bool, dict]:
    """Return (visible, info) for a vehicle crop via the ANPR service.

    ``visible`` is True when the service returns at least one plate whose
    detection confidence meets ``PLATE_CONF_THRESHOLD``. Never raises: on any
    failure it returns (False, {...error}).
    """
    ok, buf = cv2.imencode(".jpg", crop_bgr)
    if not ok:
        return False, {"error": "encode_failed"}
    blob = buf.tobytes()

    last_error = "unknown"
    for attempt in range(1, ANPR_MAX_RETRIES + 1):
        try:
            resp = session.post(
                ANPR_API_URL,
                files={ANPR_FILE_FIELD: ("crop.jpg", blob, "image/jpeg")},
                timeout=ANPR_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.Timeout:
            last_error = "timeout"
        except requests.RequestException as exc:
            last_error = f"request_error:{exc.__class__.__name__}"
        except ValueError:
            last_error = "invalid_json"
        else:
            plates = data.get(ANPR_RESULTS_KEY) or []
            best = 0.0
            if isinstance(plates, list):
                for plate in plates:
                    if not isinstance(plate, dict):
                        continue
                    try:
                        conf = float(plate.get(ANPR_CONF_KEY, 0.0) or 0.0)
                    except (TypeError, ValueError):
                        conf = 0.0
                    best = max(best, conf)
            return (best >= PLATE_CONF_THRESHOLD), {
                "plates": len(plates) if isinstance(plates, list) else 0,
                "best_conf": round(best, 4),
            }

        if attempt < ANPR_MAX_RETRIES:
            time.sleep(ANPR_RETRY_BACKOFF * attempt)

    return False, {"error": last_error}


def collect_source_images(source: Path) -> list[Path]:
    """Collect images from a flat folder (night_images) or an images/ tree."""
    source = Path(source)
    if source.is_file():
        return list_images(source)
    if not source.exists():
        return []

    collected: list[Path] = []
    for p in source.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        parts = set(p.parts)
        if "images_no_ocr" in parts:
            continue
        if "images" not in parts:
            continue
        collected.append(p)

    if collected:
        return sorted(collected)
    return list_images(source)


def draw_label_inside_box(img, xyxy, class_name: str, conf: float | None = None):
    """Draw box and write class label inside the top-left of the box."""
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    color = CLASS_COLORS.get(class_name, (0, 200, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text = class_name if conf is None else f"{class_name} {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    pad = 4
    tx = x1 + pad
    ty = y1 + th + pad + 2
    if ty + baseline > y2:
        ty = max(y1 + th + pad, y2 - pad)

    bg_x2 = min(x2, tx + tw + pad)
    bg_y1 = max(y1, ty - th - pad)
    bg_y2 = min(y2, ty + baseline + 2)
    cv2.rectangle(img, (tx - 2, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(img, text, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return img


def box_area_frac(xyxy, img_w: int, img_h: int) -> float:
    x1, y1, x2, y2 = xyxy
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(1.0, float(img_w * img_h))


def xyxy_to_yolo(xyxy, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx / img_w, cy / img_h, bw / img_w, bh / img_h)


def crop_box(bgr, xyxy):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2]


def write_text_flush(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def write_image_flush(path: Path, bgr) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        return False
    with path.open("rb") as f:
        os.fsync(f.fileno())
    return True


def copy_image_flush(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    with dst.open("rb") as f:
        os.fsync(f.fileno())


def write_data_yaml(path: Path) -> None:
    lines = [f"nc: {len(CLASS_NAMES)}", "names:"]
    for n in CLASS_NAMES:
        lines.append(f"  - {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def process_image(
    session: requests.Session,
    model: YOLO,
    yolo_lock: asyncio.Lock,
    img_path: Path,
    out_labels: Path,
    out_images: Path,
    deleted_dir: Path,
    conf: float,
    iou: float,
    min_area_frac: float,
    max_vehicles: int,
    device: str | None,
    debug_dir: Path | None = None,
) -> dict:
    """Detect, plate-check, and label one rear-view image.

    Rejected images (no proper box / no visible plate) are copied into
    ``deleted_dir``; accepted images get a label + image copy (+ debug overlay).
    """
    row: dict = {
        "file": img_path.name,
        "path": str(img_path),
        "status": "ok",
        "detections_raw": 0,
        "skipped_small": 0,
        "kept": [],
        "skipped_boxes": [],
    }

    def reject(reason: str) -> dict:
        """Copy the source image into deleted_images and tag the row."""
        row["status"] = reason
        try:
            copy_image_flush(img_path, deleted_dir / img_path.name)
            row["deleted_file"] = str(deleted_dir / img_path.name)
        except (OSError, shutil.Error) as exc:
            row["deleted_error"] = str(exc)
        return row

    bgr = await asyncio.to_thread(cv2.imread, str(img_path))
    if bgr is None:
        return reject("read_failed")

    img_h, img_w = bgr.shape[:2]

    async with yolo_lock:
        results = await asyncio.to_thread(
            lambda: model.predict(
                source=str(img_path),
                conf=conf,
                iou=iou,
                classes=list(VEHICLE_CLASS_IDS.keys()),
                device=device,
                verbose=False,
            )[0]
        )

    candidates = []
    if results.boxes is not None and len(results.boxes):
        boxes = results.boxes.xyxy.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        row["detections_raw"] = len(boxes)

        for box, cls_id, conf_v in zip(boxes, cls_ids, confs):
            coco_name = VEHICLE_CLASS_IDS.get(int(cls_id), f"class_{int(cls_id)}")
            frac = box_area_frac(box, img_w, img_h)
            if frac < min_area_frac:
                row["skipped_small"] += 1
                row["skipped_boxes"].append(
                    {
                        "reason": "too_small",
                        "name": coco_name,
                        "conf": round(float(conf_v), 4),
                        "area_frac": round(frac, 4),
                        "xyxy": [round(float(x), 1) for x in box.tolist()],
                    }
                )
                continue
            candidates.append(
                {
                    "xyxy": box.tolist(),
                    "coco_name": coco_name,
                    "conf": float(conf_v),
                    "area_frac": frac,
                }
            )

    # NO BOX: nothing usable detected.
    if not candidates:
        return reject("no_box")

    # Too many vehicles -> treat as not a clean single-vehicle rear shot.
    if len(candidates) > max_vehicles:
        row["num_candidates"] = len(candidates)
        return reject("too_many_vehicles")

    label_lines: list[str] = []
    debug_img = bgr.copy() if debug_dir is not None else None

    for cand in candidates:
        crop = crop_box(bgr, cand["xyxy"])
        if crop is None:
            return reject("no_box")  # invalid/empty crop == no proper box

        # NO PLATE: reject the whole image if any vehicle has no visible plate.
        visible, info = await asyncio.to_thread(anpr_plate_visible, session, crop)
        cand["plate_info"] = info
        if not visible:
            row["plate_info"] = info
            return reject("no_plate")

        vehicle_type = "bike" if is_bike(cand["coco_name"]) else "non_bike"
        class_name = f"{vehicle_type}_back"  # rear-only set
        class_id = CLASS_TO_ID[class_name]

        cx, cy, bw, bh = xyxy_to_yolo(cand["xyxy"], img_w, img_h)
        cx = min(1.0, max(0.0, cx))
        cy = min(1.0, max(0.0, cy))
        bw = min(1.0, max(0.0, bw))
        bh = min(1.0, max(0.0, bh))

        label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        row["kept"].append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "coco_name": cand["coco_name"],
                "side": "back",
                "conf": round(cand["conf"], 4),
                "area_frac": round(cand["area_frac"], 4),
                "plate": info,
                "xyxy": [round(float(x), 1) for x in cand["xyxy"]],
                "yolo": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
            }
        )

        if debug_img is not None:
            draw_label_inside_box(debug_img, cand["xyxy"], class_name, cand["conf"])

    if not label_lines:
        return reject("no_box")

    # Accept: write label + copy image (+ debug overlay).
    label_path = out_labels / f"{img_path.stem}.txt"
    write_text_flush(label_path, "\n".join(label_lines) + "\n")
    row["label_file"] = str(label_path)
    row["status"] = "labeled"

    img_out = out_images / img_path.name
    copy_image_flush(img_path, img_out)
    row["image_file"] = str(img_out)

    if debug_img is not None and debug_dir is not None:
        debug_path = debug_dir / img_path.name
        write_image_flush(debug_path, debug_img)
        row["debug_file"] = str(debug_path)

    return row


async def process_one_worker(
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
    log_lock: asyncio.Lock,
    fout,
    counts: dict,
    **kwargs,
) -> dict:
    async with sem:
        row = await process_image(**kwargs)
        status = row.get("status", "ok")
        kept_n = len(row.get("kept") or [])

        async with log_lock:
            counts[status] = counts.get(status, 0) + 1
            counts["boxes_kept"] += kept_n
            counts["boxes_skipped_small"] += int(row.get("skipped_small") or 0)
            if status != "labeled":
                counts["deleted"] += 1
            fout.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

        saved_bits = []
        if row.get("label_file"):
            saved_bits.append(f"txt={row['label_file']}")
        if row.get("image_file"):
            saved_bits.append(f"img={row['image_file']}")
        if row.get("deleted_file"):
            saved_bits.append(f"del={row['deleted_file']}")
        saved_msg = (" | " + ", ".join(saved_bits)) if saved_bits else ""
        print(
            f"[{idx}/{total}] {kwargs['img_path'].name} -> {status} "
            f"(kept={kept_n}, small={row.get('skipped_small', 0)}, "
            f"raw={row.get('detections_raw', 0)}){saved_msg}",
            flush=True,
        )
        return row


async def run_async(args) -> None:
    images = collect_source_images(args.source)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.source}")

    args.out_labels.mkdir(parents=True, exist_ok=True)
    args.out_images.mkdir(parents=True, exist_ok=True)
    args.deleted_dir.mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.data_yaml)

    debug_dir = args.debug_dir if args.save_debug else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO: {args.weights}")
    model = YOLO(str(args.weights))

    session = build_anpr_session(args.workers)
    sem = asyncio.Semaphore(args.workers)
    yolo_lock = asyncio.Lock()
    log_lock = asyncio.Lock()

    print(f"Images: {len(images)} (workers={args.workers})")
    print(f"Source     -> {args.source}")
    print(f"Dataset    -> {args.dataset_dir}")
    print(f"Labels     -> {args.out_labels}")
    print(f"Images     -> {args.out_images}")
    print(f"Deleted    -> {args.deleted_dir}")
    print(f"data.yaml  -> {args.data_yaml}")
    print(f"Log        -> {args.log}")
    if debug_dir is not None:
        print(f"Debug      -> {debug_dir}")
    print(f"ANPR       -> {ANPR_API_URL} (plate_conf>={PLATE_CONF_THRESHOLD})")
    print(f"Filters: min_area_frac={args.min_area_frac}, max_vehicles={args.max_vehicles}")

    counts = {
        "labeled": 0,
        "no_box": 0,
        "no_plate": 0,
        "too_many_vehicles": 0,
        "read_failed": 0,
        "deleted": 0,
        "boxes_kept": 0,
        "boxes_skipped_small": 0,
    }

    with args.log.open("w", encoding="utf-8") as fout:
        tasks = []
        for i, img_path in enumerate(images, start=1):
            tasks.append(
                process_one_worker(
                    sem=sem,
                    idx=i,
                    total=len(images),
                    log_lock=log_lock,
                    fout=fout,
                    counts=counts,
                    session=session,
                    model=model,
                    yolo_lock=yolo_lock,
                    img_path=img_path,
                    out_labels=args.out_labels,
                    out_images=args.out_images,
                    deleted_dir=args.deleted_dir,
                    conf=args.conf,
                    iou=args.iou,
                    min_area_frac=args.min_area_frac,
                    max_vehicles=args.max_vehicles,
                    device=args.device or None,
                    debug_dir=debug_dir,
                )
            )
        await asyncio.gather(*tasks)

    session.close()
    print("\nDone.")
    print(json.dumps(counts, indent=2, default=_json_default))


def main() -> None:
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent  # project root (weights, etc.)

    default_source = Path("/mnt/datadisk/avani/front_back/main/night_images")
    default_dataset_dir = Path("/mnt/datadisk/avani/front_back/main/night_dataset")

    ap = argparse.ArgumentParser(
        description="Rear-only YOLO labels (bike_back / non_bike_back) with "
                    "plate-visibility filtering."
    )
    ap.add_argument("--source", type=Path, default=default_source)
    ap.add_argument("--weights", type=Path, default=root / "yolo26x.pt")
    ap.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)
    ap.add_argument("--out-labels", type=Path, default=None)
    ap.add_argument("--out-images", type=Path, default=None)
    ap.add_argument("--deleted-dir", type=Path, default=None)
    ap.add_argument("--data-yaml", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--min-area-frac", type=float, default=0.05)
    ap.add_argument("--max-vehicles", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--workers", type=int, default=16,
                    help="Concurrent images in flight (default: 16)")
    ap.add_argument("--device", type=str, default="", help="cuda / cpu / empty=auto")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--save-debug", action="store_true", default=True)
    ap.add_argument("--no-save-debug", action="store_false", dest="save_debug")
    args = ap.parse_args()

    args.out_labels = args.out_labels or (args.dataset_dir / "labels")
    args.out_images = args.out_images or (args.dataset_dir / "images")
    args.deleted_dir = args.deleted_dir or (args.dataset_dir / "deleted_images")
    args.data_yaml = args.data_yaml or (args.dataset_dir / "data.yaml")
    args.log = args.log or (args.dataset_dir / "label_gen_log.jsonl")
    args.debug_dir = args.debug_dir or (args.dataset_dir / "debug")
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
