#!/usr/bin/env python3
"""split_car_other.py

Convert the 5-class label set into the 7-class one by splitting
non_bike_front / non_bike_back into car_* and other_*.

  OLD (5)                     NEW (7)
  0 bike_front         ->     0 bike_front
  1 bike_back          ->     1 bike_back
  2 non_bike_front     ->     2 car_front   OR  4 other_front
  3 non_bike_back      ->     3 car_back    OR  5 other_back
  4 reg_plate          ->     6 reg_plate

Only class 2 and 3 boxes require a decision. For each one the box region is
CROPPED from the image (with a little padding for context) and run through a
COCO-pretrained detector. If the crop reads as COCO 'car' it becomes car_*,
anything else becomes other_*. Bike boxes and plate boxes are pure digit
remaps -- no inference, no risk.

Boxes, coordinates and front/back orientation are never modified.

Original labels are NOT touched: results go to a new --out-labels directory.

Run:
    python split_car_other.py \
        --images  Dataset3_merged/images \
        --labels  Dataset3_merged/labels \
        --out-labels Dataset3_merged/labels_7class \
        --weights /mnt/datadisk/avani/front_back/yolo26x.pt
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm
from ultralytics import YOLO

# --- old -> new mapping for the classes that need no decision ---------------
DIRECT_MAP = {0: 0, 1: 1, 4: 6}          # bike_front, bike_back, reg_plate
DECIDE = {2: (2, 4), 3: (3, 5)}          # old: (car_id, other_id)

NEW_NAMES = {
    0: "bike_front", 1: "bike_back",
    2: "car_front", 3: "car_back",
    4: "other_front", 5: "other_back",
    6: "reg_plate",
}

# COCO ids. 2 = car -> car_*. Everything else that is a vehicle -> other_*.
COCO_CAR = 2
COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_labels(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() == ".txt")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Split non_bike into car / other")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out-labels", type=Path, required=True)
    ap.add_argument("--weights", type=Path,
                    default=Path("/mnt/datadisk/avani/front_back/yolo26x.pt"))
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Detection conf inside the crop")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="Fraction of box size added as context padding")
    ap.add_argument("--batch", type=int, default=64, help="Crops per GPU call")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--fallback", choices=["car", "other"], default="car",
                    help="Class to use when the crop yields no vehicle detection")
    ap.add_argument("--save-crops", type=int, default=0,
                    help="Save this many sample crops per decision for review")
    ap.add_argument("--limit", type=int, default=0, help="0 = all label files")
    args = ap.parse_args()

    args.out_labels.mkdir(parents=True, exist_ok=True)
    review_dir = args.out_labels.parent / "car_other_review"
    if args.save_crops:
        for sub in ("car", "other", "fallback"):
            (review_dir / sub).mkdir(parents=True, exist_ok=True)

    labels = iter_labels(args.labels)
    if args.limit and args.limit > 0:
        labels = labels[: args.limit]
    if not labels:
        raise SystemExit(f"No .txt labels in {args.labels}")

    print(f"Loading COCO detector: {args.weights}")
    model = YOLO(str(args.weights))

    counts = {k: 0 for k in NEW_NAMES}
    stats = {"decided_car": 0, "decided_other": 0, "fallback": 0,
             "no_image": 0, "bad_crop": 0, "files": 0}
    saved = {"car": 0, "other": 0, "fallback": 0}
    log_rows: List[list] = []
    rng = random.Random(42)

    progress = tqdm(total=len(labels), desc="Relabelling", unit="file")
    for lbl in labels:
        try:
            lines = lbl.read_text(encoding="utf-8").splitlines()
        except OSError:
            progress.update(1)
            continue

        parsed: List[Tuple[int, float, float, float, float]] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            parsed.append((cid, cx, cy, bw, bh))

        needs = [i for i, p in enumerate(parsed) if p[0] in DECIDE]

        image = None
        if needs:
            img_path = find_image(args.images, lbl.stem)
            if img_path is None:
                stats["no_image"] += 1
            else:
                image = cv2.imread(str(img_path))
                if image is None:
                    stats["no_image"] += 1

        decisions: Dict[int, int] = {}
        if needs and image is not None:
            h, w = image.shape[:2]
            crops, crop_idx = [], []
            for i in needs:
                _cid, cx, cy, bw, bh = parsed[i]
                px, py = bw * args.pad, bh * args.pad
                x1 = int(max(0.0, (cx - bw / 2 - px)) * w)
                y1 = int(max(0.0, (cy - bh / 2 - py)) * h)
                x2 = int(min(1.0, (cx + bw / 2 + px)) * w)
                y2 = int(min(1.0, (cy + bh / 2 + py)) * h)
                if x2 - x1 < 8 or y2 - y1 < 8:
                    stats["bad_crop"] += 1
                    continue
                crops.append(image[y1:y2, x1:x2])
                crop_idx.append(i)

            for s in range(0, len(crops), args.batch):
                chunk = crops[s:s + args.batch]
                idxs = crop_idx[s:s + args.batch]
                res = model.predict(source=chunk, conf=args.conf,
                                    classes=list(COCO_VEHICLES),
                                    device=args.device, verbose=False)
                for r, i, crop in zip(res, idxs, chunk):
                    best_cls, best_conf = None, 0.0
                    if r.boxes is not None and len(r.boxes):
                        cl = r.boxes.cls.cpu().numpy()
                        cf = r.boxes.conf.cpu().numpy()
                        for c, v in zip(cl, cf):
                            if v > best_conf:
                                best_conf, best_cls = float(v), int(c)
                    old = parsed[i][0]
                    car_id, other_id = DECIDE[old]
                    if best_cls is None:
                        new_id = car_id if args.fallback == "car" else other_id
                        stats["fallback"] += 1
                        bucket = "fallback"
                    elif best_cls == COCO_CAR:
                        new_id = car_id
                        stats["decided_car"] += 1
                        bucket = "car"
                    else:
                        new_id = other_id
                        stats["decided_other"] += 1
                        bucket = "other"
                    decisions[i] = new_id
                    log_rows.append([lbl.stem, old, new_id,
                                     COCO_VEHICLES.get(best_cls, "none"),
                                     f"{best_conf:.3f}"])
                    if args.save_crops and saved[bucket] < args.save_crops \
                            and rng.random() < 0.5:
                        cv2.imwrite(str(review_dir / bucket /
                                        f"{lbl.stem}_{i}_{bucket}.jpg"), crop)
                        saved[bucket] += 1

        out_lines: List[str] = []
        for i, (cid, cx, cy, bw, bh) in enumerate(parsed):
            if cid in DIRECT_MAP:
                new_id = DIRECT_MAP[cid]
            elif cid in DECIDE:
                new_id = decisions.get(
                    i, DECIDE[cid][0] if args.fallback == "car" else DECIDE[cid][1])
                if i not in decisions:
                    stats["fallback"] += 1
            else:
                continue  # unknown class id, drop
            counts[new_id] = counts.get(new_id, 0) + 1
            out_lines.append(f"{new_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (args.out_labels / lbl.name).write_text(
            ("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")
        stats["files"] += 1
        progress.update(1)
    progress.close()

    csv_path = args.out_labels.parent / "car_other_decisions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["stem", "old_class", "new_class", "coco_class", "conf"])
        wcsv.writerows(log_rows)

    total_boxes = sum(counts.values())
    print("\n==================== 7-CLASS RELABEL SUMMARY ====================")
    print(f"Label files written : {stats['files']}  -> {args.out_labels}")
    print(f"Total boxes         : {total_boxes}")
    print("-" * 64)
    for k in sorted(NEW_NAMES):
        pct = counts.get(k, 0) / max(total_boxes, 1) * 100
        print(f"  {k} {NEW_NAMES[k]:<14}: {counts.get(k, 0):>8}  ({pct:5.2f}%)")
    print("-" * 64)
    print(f"Decided car         : {stats['decided_car']}")
    print(f"Decided other       : {stats['decided_other']}")
    print(f"Fallback ({args.fallback})   : {stats['fallback']}  <-- review these")
    print(f"Missing image       : {stats['no_image']}")
    print(f"Crop too small      : {stats['bad_crop']}")
    print(f"Decisions CSV       : {csv_path}")
    if args.save_crops:
        print(f"Sample crops        : {review_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
