#!/usr/bin/env python3
"""eval_metrics.py

Proper detection metrics (precision, recall, F1, AP@0.5, AP@0.5:0.95) for a
folder of images + labels that does NOT follow YOLOv5's images/labels naming
convention -- so it works on external datasets without copying or renaming
anything.

Why not val.py? val.py locates labels by string-replacing '/images/' with
'/labels/' in the image path and needs a data.yaml. This script takes the two
folders explicitly and can remap ground-truth class ids in memory, so the
source dataset is never modified.

How AP is computed (standard COCO-style):
  * run the model at a very low confidence (--min-conf) so the full
    precision-recall curve is visible,
  * pool every prediction across the dataset, sort by confidence descending,
  * greedily match each prediction to an unmatched GT box in the same image
    at a given IoU threshold -> TP, otherwise FP,
  * walk the sorted list accumulating TP/FP to trace the PR curve,
  * AP = area under that curve (all-point interpolation),
  * repeat at IoU 0.50, 0.55 ... 0.95 and average -> AP@0.5:0.95.

Precision/recall/F1 are additionally reported at your chosen operating
threshold (--conf), because that is what you would actually deploy at.

Run:
    python eval_metrics.py \
        --weights /mnt/datadisk/avani/front_back/main/BESTPT/4_August_new_model/best.pt \
        --images  /root/DATASETS/plate_roi_dataset/test/reg_plate \
        --labels  /root/DATASETS/plate_roi_dataset/test_labels \
        --only-class 4 --gt-class-map 0:4 --img 320 --batch 128
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "non_bike_front",
    3: "non_bike_back",
    4: "reg_plate",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def parse_class_map(spec: str | None) -> Dict[int, int]:
    if not spec:
        return {}
    out: Dict[int, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        try:
            a, b = pair.split(":")
            out[int(a)] = int(b)
        except ValueError:
            raise SystemExit(f"Bad --gt-class-map entry: {pair!r}")
    return out


def load_gt(label_path: Path, w: int, h: int,
            class_map: Dict[int, int]) -> List[Tuple[int, float, float, float, float]]:
    out = []
    if not label_path.exists():
        return out
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            if class_map:
                cid = class_map.get(cid, cid)
            out.append((cid, (cx - bw / 2) * w, (cy - bh / 2) * h,
                        (cx + bw / 2) * w, (cy + bh / 2) * h))
    except OSError:
        pass
    return out


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def compute_ap(preds: List[Tuple[float, int, tuple]],
               gt_by_img: Dict[int, List[tuple]],
               n_gt: int, iou_thr: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return (AP, precision_curve, recall_curve) at one IoU threshold.

    preds: (confidence, image_index, box) sorted descending by confidence.
    """
    if n_gt == 0 or not preds:
        return 0.0, np.array([]), np.array([])

    matched: Dict[int, set] = {}
    tp = np.zeros(len(preds), dtype=np.float64)
    fp = np.zeros(len(preds), dtype=np.float64)

    for i, (_conf, img_idx, box) in enumerate(preds):
        gts = gt_by_img.get(img_idx, [])
        used = matched.setdefault(img_idx, set())
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou_xyxy(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            tp[i] = 1.0
        else:
            fp[i] = 1.0

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)

    # All-point interpolation: make precision monotonically decreasing.
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, precision, recall


def main() -> None:
    ap_ = argparse.ArgumentParser(description="Detection metrics on a folder pair")
    ap_.add_argument("--weights", type=Path, required=True)
    ap_.add_argument("--images", type=Path, required=True)
    ap_.add_argument("--labels", type=Path, required=True)
    ap_.add_argument("--yolov5-dir", type=Path,
                     default=Path("/mnt/datadisk/avani/yolov5"))
    ap_.add_argument("--only-class", type=int, default=None,
                     help="Evaluate only this class id (e.g. 4 = reg_plate)")
    ap_.add_argument("--gt-class-map", type=str, default=None,
                     help="Remap GT class ids in memory, e.g. '0:4'")
    ap_.add_argument("--conf", type=float, default=0.65,
                     help="Operating threshold for the reported P/R/F1")
    ap_.add_argument("--min-conf", type=float, default=0.001,
                     help="Low threshold used to trace the full PR curve for AP")
    ap_.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    ap_.add_argument("--img", type=int, default=320)
    ap_.add_argument("--batch", type=int, default=128)
    ap_.add_argument("--device", type=str, default="0")
    ap_.add_argument("--n", type=int, default=0, help="0 = all images")
    ap_.add_argument("--csv", type=Path, default=None,
                     help="Optional path to write the PR curve as CSV")
    args = ap_.parse_args()

    class_map = parse_class_map(args.gt_class_map)
    if class_map:
        print(f"Remapping GT classes (in memory): {class_map}")

    images = iter_images(args.images)
    if args.n and args.n > 0:
        images = images[: args.n]
    if not images:
        raise SystemExit(f"No images in {args.images}")

    if not (args.yolov5_dir / "hubconf.py").is_file():
        raise SystemExit(f"yolov5 repo not found at {args.yolov5_dir}")
    device = ("cuda:" + args.device) if args.device not in ("cpu", "") else "cpu"
    print(f"Loading model via {args.yolov5_dir}")
    model = torch.hub.load(str(args.yolov5_dir), "custom",
                           path=str(args.weights), source="local", verbose=False)
    model.to(device)
    model.conf = args.min_conf     # low, so the PR curve is complete
    model.iou = args.iou
    raw = getattr(model, "names", None)
    if isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(raw)}
    else:
        names = dict(CLASS_NAMES)
    print("Model classes:", names)
    target = args.only_class
    if target is not None:
        print(f"Evaluating class {target} ({names.get(target, '?')}) only")

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    all_preds: List[Tuple[float, int, tuple]] = []
    gt_by_img: Dict[int, List[tuple]] = {}
    n_gt = 0
    n_missing_label = 0

    progress = tqdm(total=len(images), desc="Inferring", unit="img")
    img_idx = 0
    for chunk in batched(images, args.batch):
        loaded = []
        for p in chunk:
            im = cv2.imread(str(p))
            if im is None:
                progress.update(1)
                continue
            loaded.append((p, im.shape[0], im.shape[1],
                           cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if not loaded:
            continue

        results = model([lo[3] for lo in loaded], size=args.img)

        for i, (p, h, w, _rgb) in enumerate(loaded):
            lbl = args.labels / (p.stem + ".txt")
            if not lbl.exists():
                n_missing_label += 1
            gt = load_gt(lbl, w, h, class_map)
            if target is not None:
                gt = [g for g in gt if g[0] == target]
            if gt:
                gt_by_img[img_idx] = [g[1:5] for g in gt]
                n_gt += len(gt)

            det = results.xyxy[i].cpu().numpy()
            for x1, y1, x2, y2, conf, cls in det:
                if target is not None and int(cls) != target:
                    continue
                all_preds.append((float(conf), img_idx, (x1, y1, x2, y2)))

            img_idx += 1
            progress.update(1)
    progress.close()

    all_preds.sort(key=lambda t: -t[0])

    if n_gt == 0:
        raise SystemExit("No ground-truth boxes found -- check --gt-class-map "
                         "and that label filenames match image filenames.")

    # AP across IoU 0.50 : 0.05 : 0.95
    thresholds = [round(0.5 + 0.05 * k, 2) for k in range(10)]
    aps = []
    pr50 = (np.array([]), np.array([]))
    for t in thresholds:
        a, prec, rec = compute_ap(all_preds, gt_by_img, n_gt, t)
        aps.append(a)
        if abs(t - 0.5) < 1e-9:
            pr50 = (prec, rec)
    ap50 = aps[0]
    ap5095 = float(np.mean(aps))

    # P / R / F1 at the operating threshold, IoU 0.5
    op_preds = [p for p in all_preds if p[0] >= args.conf]
    matched: Dict[int, set] = {}
    tp = 0
    for _conf, idx, box in op_preds:
        gts = gt_by_img.get(idx, [])
        used = matched.setdefault(idx, set())
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou_xyxy(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= 0.5:
            used.add(best_j)
            tp += 1
    fp = len(op_preds) - tp
    fn = n_gt - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    if args.csv:
        prec, rec = pr50
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["recall", "precision"])
            for r, pv in zip(rec, prec):
                wcsv.writerow([f"{r:.6f}", f"{pv:.6f}"])
        print(f"PR curve written -> {args.csv}")

    cname = names.get(target, "all") if target is not None else "all"
    print("\n==================== DETECTION METRICS ====================")
    print(f"Images evaluated     : {img_idx}")
    if n_missing_label:
        print(f"  !! images with NO label file : {n_missing_label}")
    print(f"Class evaluated      : {cname}")
    print(f"GT boxes             : {n_gt}")
    print(f"Predictions (>= {args.min_conf}) : {len(all_preds)}")
    print("-" * 58)
    print(f"AP@0.5               : {ap50 * 100:.2f}%")
    print(f"AP@0.5:0.95          : {ap5095 * 100:.2f}%")
    print("-" * 58)
    print(f"At operating conf = {args.conf} (IoU 0.5):")
    print(f"  Precision          : {precision * 100:.2f}%   (TP {tp} / TP+FP {tp + fp})")
    print(f"  Recall             : {recall * 100:.2f}%   (TP {tp} / TP+FN {tp + fn})")
    print(f"  F1                 : {f1 * 100:.2f}%")
    print(f"  False positives    : {fp}")
    print(f"  False negatives    : {fn}")
    print("-" * 58)
    print("AP by IoU threshold:")
    for t, a in zip(thresholds, aps):
        print(f"  IoU {t:.2f} : {a * 100:6.2f}%")
    print("=" * 58)


if __name__ == "__main__":
    main()
