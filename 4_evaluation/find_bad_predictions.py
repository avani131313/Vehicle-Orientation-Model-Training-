#!/usr/bin/env python3
"""find_bad_predictions.py

Error analysis for a trained YOLOv5/YOLO model: find the images where the
model's predictions disagree most with the labels, copy them into a folder,
and draw a GT-vs-prediction overlay so you can see *why* each one failed.

For every image it matches predicted boxes to ground-truth boxes by IoU +
class and scores the image:

    badness = false_negatives (missed GT)
            + false_positives (extra predictions)
            + class_mismatches (right place, wrong class)

Images with badness >= --min-score are copied to <out>/images, an annotated
overlay is written to <out>/overlays (GREEN = ground truth, RED = prediction),
and a sorted CSV (<out>/bad_predictions.csv) lists the worst offenders first.

Run:
    python find_bad_predictions.py \
        --weights /path/to/best.pt \
        --images  /mnt/datadisk/avani/front_back/main/night_dataset/images \
        --labels  /mnt/datadisk/avani/front_back/main/night_dataset/labels
"""

from __future__ import annotations

import argparse
import csv
import shutil
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

# Classic ultralytics/yolov5 repo (has hubconf.py). Used to load YOLOv5 .pt.
DEFAULT_YOLOV5_DIR = Path("/mnt/datadisk/avani/yolov5")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

GT_COLOR = (0, 200, 0)      # green  = ground truth
PRED_COLOR = (0, 0, 220)    # red    = prediction


def iter_images(images_dir: Path) -> List[Path]:
    """Return all image paths under a directory."""
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_gt(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
    """Load ground-truth boxes as (class_id, xyxy_pixels)."""
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
        x1 = (cx - bw / 2.0) * img_w
        y1 = (cy - bh / 2.0) * img_h
        x2 = (cx + bw / 2.0) * img_w
        y2 = (cy + bh / 2.0) * img_h
        boxes.append((cls, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return boxes


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def score_image(
    preds: Sequence[Tuple[int, float, np.ndarray]],
    gts: Sequence[Tuple[int, np.ndarray]],
    match_iou: float,
) -> Dict[str, int]:
    """Greedily match preds to GTs and count error types."""
    pred_used = [False] * len(preds)
    gt_used = [False] * len(gts)

    # Candidate pairs by IoU, matched greedily from highest overlap.
    pairs: List[Tuple[float, int, int]] = []
    for pi, (_pc, _conf, pbox) in enumerate(preds):
        for gi, (_gc, gbox) in enumerate(gts):
            iou = iou_xyxy(pbox, gbox)
            if iou >= match_iou:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True)

    tp = 0
    class_mismatch = 0
    for iou, pi, gi in pairs:
        if pred_used[pi] or gt_used[gi]:
            continue
        pred_used[pi] = True
        gt_used[gi] = True
        if preds[pi][0] == gts[gi][0]:
            tp += 1
        else:
            class_mismatch += 1

    false_neg = gt_used.count(False)   # GTs with no matched prediction
    false_pos = pred_used.count(False) # predictions matching no GT
    badness = false_neg + false_pos + class_mismatch
    return {
        "tp": tp,
        "false_neg": false_neg,
        "false_pos": false_pos,
        "class_mismatch": class_mismatch,
        "badness": badness,
    }


def draw_overlay(
    image: np.ndarray,
    preds: Sequence[Tuple[int, float, np.ndarray]],
    gts: Sequence[Tuple[int, np.ndarray]],
) -> np.ndarray:
    """Draw GT (green) and predictions (red) on a copy of the image."""
    out = image.copy()
    for cls, box in gts:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), GT_COLOR, 2)
        cv2.putText(out, f"GT:{CLASS_NAMES.get(cls, cls)}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1, cv2.LINE_AA)
    for cls, conf, box in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), PRED_COLOR, 2)
        cv2.putText(out, f"P:{CLASS_NAMES.get(cls, cls)} {conf:.2f}",
                    (x1, min(out.shape[0] - 2, y2 + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, PRED_COLOR, 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Find worst-predicted images")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--yolov5-dir", type=Path, default=DEFAULT_YOLOV5_DIR,
                    help="Path to the classic yolov5 repo (with hubconf.py)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <images>/../bad_predictions)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Prediction confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.5,
                    help="IoU to consider a pred and GT the same object")
    ap.add_argument("--img", type=int, default=320, help="Inference image size")
    ap.add_argument("--device", type=str, default="0", help="cuda index or cpu")
    ap.add_argument("--min-score", type=int, default=1,
                    help="Copy images with badness >= this (default 1)")
    ap.add_argument("--top-n", type=int, default=0,
                    help="Only keep the N worst images (0 = all flagged)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    out_dir = args.out or (args.images.parent / "bad_predictions")
    out_images = out_dir / "images"
    out_overlays = out_dir / "overlays"
    out_images.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bad_predictions.csv"

    images = iter_images(args.images)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    print(f"Loading model via yolov5 repo: {args.yolov5_dir}")
    if not (args.yolov5_dir / "hubconf.py").is_file():
        raise SystemExit(
            f"yolov5 repo not found at {args.yolov5_dir} "
            "(no hubconf.py). Pass --yolov5-dir /path/to/yolov5")
    device = ("cuda:" + args.device) if args.device not in ("cpu", "") else "cpu"
    model = torch.hub.load(
        str(args.yolov5_dir), "custom", path=str(args.weights),
        source="local", verbose=False,
    )
    model.to(device)
    model.conf = args.conf   # confidence threshold
    model.iou = args.iou     # NMS IoU threshold

    rows: List[dict] = []
    for img_path in tqdm(images, desc="Analyzing", unit="img"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]

        # yolov5 AutoShape handles loading + letterbox; RGB in.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = model(rgb, size=args.img)
        det = result.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

        preds: List[Tuple[int, float, np.ndarray]] = []
        for row_det in det:
            x1, y1, x2, y2, cf, c = row_det[:6]
            preds.append((int(c), float(cf),
                          np.array([x1, y1, x2, y2], dtype=np.float32)))

        label_path = args.labels / (img_path.stem + ".txt")
        gts = load_gt(label_path, w, h)

        s = score_image(preds, gts, args.match_iou)
        if s["badness"] < args.min_score:
            continue

        rows.append({
            "image": img_path.name,
            "badness": s["badness"],
            "false_neg": s["false_neg"],
            "false_pos": s["false_pos"],
            "class_mismatch": s["class_mismatch"],
            "n_gt": len(gts),
            "n_pred": len(preds),
            "_src": img_path,
            "_image": image,
            "_preds": preds,
            "_gts": gts,
        })

    # Worst first; optionally keep only the top-N.
    rows.sort(key=lambda r: r["badness"], reverse=True)
    if args.top_n and args.top_n > 0:
        rows = rows[: args.top_n]

    for r in rows:
        shutil.copy2(str(r["_src"]), str(out_images / r["_src"].name))
        overlay = draw_overlay(r["_image"], r["_preds"], r["_gts"])
        cv2.imwrite(str(out_overlays / r["_src"].name), overlay)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "badness", "false_neg", "false_pos",
                         "class_mismatch", "n_gt", "n_pred"])
        for r in rows:
            writer.writerow([r["image"], r["badness"], r["false_neg"],
                             r["false_pos"], r["class_mismatch"],
                             r["n_gt"], r["n_pred"]])

    print(f"\nFlagged {len(rows)} bad images "
          f"(min_score={args.min_score}).")
    print(f"Images   -> {out_images}")
    print(f"Overlays -> {out_overlays}  (GREEN=GT, RED=prediction)")
    print(f"CSV      -> {csv_path}")


if __name__ == "__main__":
    main()
