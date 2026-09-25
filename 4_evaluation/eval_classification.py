#!/usr/bin/env python3
"""eval_classification.py

Measure how well the model *identifies the class* of each vehicle, ignoring
bounding-box quality. For every ground-truth box we find the model's overlapping
prediction (loose IoU, since we don't care about box precision) and check whether
its predicted class matches. Output is a 4x4 confusion matrix plus per-class
classification accuracy and overall accuracy.

Classes:
    0 bike_front   1 bike_back   2 non_bike_front   3 non_bike_back

A ground-truth vehicle with no overlapping prediction is counted as "missed"
(the model didn't find it at all) and reported separately, so it doesn't distort
the classification accuracy of the vehicles it *did* find.

Run:
    python eval_classification.py \
        --weights /path/to/best.pt \
        --images  night_dataset/images \
        --labels  night_dataset/labels \
        --yolov5-dir /mnt/datadisk/avani/yolov5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "non_bike_front",
    3: "non_bike_back",
}
NUM_CLASSES = 4
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_YOLOV5_DIR = Path("/mnt/datadisk/avani/yolov5")


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_gt(label_path: Path, w: int, h: int) -> List[Tuple[int, np.ndarray]]:
    boxes: List[Tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return boxes
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return boxes
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                              float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        boxes.append((cls, np.array(
            [(cx - bw / 2) * w, (cy - bh / 2) * h,
             (cx + bw / 2) * w, (cy + bh / 2) * h], dtype=np.float32)))
    return boxes


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return float(inter / ua) if ua > 0 else 0.0


def best_pred_class(
    gbox: np.ndarray,
    preds: Sequence[Tuple[int, float, np.ndarray]],
    match_iou: float,
) -> Optional[int]:
    """Return the class of the highest-IoU prediction overlapping gbox, or None."""
    best_iou = match_iou
    best_cls: Optional[int] = None
    for pcls, _conf, pbox in preds:
        iou = iou_xyxy(gbox, pbox)
        if iou >= best_iou:
            best_iou = iou
            best_cls = pcls
    return best_cls


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-vehicle classification accuracy")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--yolov5-dir", type=Path, default=DEFAULT_YOLOV5_DIR)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Min prediction confidence to consider")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.3,
                    help="Loose IoU to link a GT vehicle to a prediction")
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV path for the confusion matrix (optional)")
    args = ap.parse_args()

    images = iter_images(args.images)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    if not (args.yolov5_dir / "hubconf.py").is_file():
        raise SystemExit(f"yolov5 repo not found at {args.yolov5_dir}")
    device = ("cuda:" + args.device) if args.device not in ("cpu", "") else "cpu"
    print(f"Loading model via {args.yolov5_dir}")
    model = torch.hub.load(str(args.yolov5_dir), "custom",
                           path=str(args.weights), source="local", verbose=False)
    model.to(device)
    model.conf = args.conf
    model.iou = args.iou

    # confusion[gt][pred]; plus a "missed" tally per gt class.
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    missed = np.zeros(NUM_CLASSES, dtype=np.int64)

    for img_path in tqdm(images, desc="Classifying", unit="img"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        det = model(rgb, size=args.img).xyxy[0].cpu().numpy()
        preds = [(int(r[5]), float(r[4]),
                  np.array(r[:4], dtype=np.float32)) for r in det]

        for gcls, gbox in load_gt(args.labels / (img_path.stem + ".txt"), w, h):
            if not (0 <= gcls < NUM_CLASSES):
                continue
            pcls = best_pred_class(gbox, preds, args.match_iou)
            if pcls is None or not (0 <= pcls < NUM_CLASSES):
                missed[gcls] += 1
            else:
                confusion[gcls][pcls] += 1

    # ---- report ----
    names = [CLASS_NAMES[i] for i in range(NUM_CLASSES)]
    total_matched = int(confusion.sum())
    total_correct = int(np.trace(confusion))

    print("\n================ CONFUSION MATRIX (rows=truth, cols=predicted) ==========")
    header = "truth \\ pred".ljust(16) + "".join(n[:13].rjust(15) for n in names) + "  missed"
    print(header)
    for i in range(NUM_CLASSES):
        row = CLASS_NAMES[i].ljust(16)
        row += "".join(str(confusion[i][j]).rjust(15) for j in range(NUM_CLASSES))
        row += str(missed[i]).rjust(9)
        print(row)

    print("\n================ PER-CLASS CLASSIFICATION ACCURACY ======================")
    print("{:<16}{:>10}{:>10}{:>10}{:>12}{:>10}".format(
        "class", "gt_total", "found", "correct", "class_acc", "found%"))
    print("-" * 70)
    for i in range(NUM_CLASSES):
        found = int(confusion[i].sum())
        gt_total = found + int(missed[i])
        correct = int(confusion[i][i])
        cacc = (correct / found * 100.0) if found else 0.0
        frate = (found / gt_total * 100.0) if gt_total else 0.0
        print("{:<16}{:>10}{:>10}{:>10}{:>11.1f}%{:>9.1f}%".format(
            CLASS_NAMES[i], gt_total, found, correct, cacc, frate))
    print("-" * 70)
    overall = (total_correct / total_matched * 100.0) if total_matched else 0.0
    print(f"\nOVERALL classification accuracy (over vehicles the model found): "
          f"{overall:.2f}%  ({total_correct}/{total_matched})")
    print(f"Total ground-truth vehicles: {total_matched + int(missed.sum())} "
          f"| found: {total_matched} | missed: {int(missed.sum())}")

    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(["truth\\pred"] + names + ["missed"])
            for i in range(NUM_CLASSES):
                wtr.writerow([CLASS_NAMES[i]] +
                             [int(confusion[i][j]) for j in range(NUM_CLASSES)] +
                             [int(missed[i])])
        print(f"Confusion matrix CSV -> {args.out}")


if __name__ == "__main__":
    main()
