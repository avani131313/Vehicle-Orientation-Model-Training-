#!/usr/bin/env python3
"""find_plate_errors.py

Find and visualise the WORST predictions for ONE class (default: reg_plate,
class 4) so you can see exactly where plate detection is failing.

For each image it compares ground-truth boxes of the target class against
predicted boxes of the target class and buckets the failures:

  missed        - a GT plate with no matching prediction        (false negative)
  spurious      - a predicted plate with no matching GT         (false positive)
  loose_box     - matched, but IoU below --good-iou             (bad localisation)
  wrong_class   - a GT plate whose best overlap was predicted
                  as a DIFFERENT class (e.g. called a vehicle)

Badness score = 3*missed + 2*spurious + 1*wrong_class + 1*loose_box
(missing a plate is the worst outcome for an ANPR pipeline, so it weighs most.)

Images are sorted worst-first. The top --n are written out with overlays:
  GREEN  = ground-truth plate
  RED    = predicted plate (label shows confidence)
  YELLOW = prediction of a different class that overlapped a GT plate

Also writes plate_errors.csv with per-image counts, and optionally splits the
overlays into per-failure-type subfolders with --split-by-type.

Run:
    python find_plate_errors.py \
        --weights /mnt/datadisk/avani/front_back/main/BESTPT/4_August_new_model/best.pt \
        --images  /mnt/datadisk/avani/front_back/main/Dataset3_merged/yolo_split/test/images \
        --labels  /mnt/datadisk/avani/front_back/main/Dataset3_merged/yolo_split/test/labels \
        --yolov5-dir /mnt/datadisk/avani/yolov5 --img 320
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

GT_COLOR = (0, 200, 0)        # green  - ground truth
PRED_COLOR = (0, 0, 220)      # red    - predicted target class
OTHER_COLOR = (0, 200, 255)   # yellow - other-class prediction over a GT plate

# Badness weights: a missed plate hurts an ANPR pipeline most.
W_MISSED, W_SPURIOUS, W_WRONG_CLASS, W_LOOSE = 3.0, 2.0, 1.0, 1.0


def iter_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_gt(label_path: Path, w: int, h: int
            ) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO label -> [(cls, x1, y1, x2, y2), ...] in pixels."""
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
            out.append((cid,
                        (cx - bw / 2) * w, (cy - bh / 2) * h,
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
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def draw(img, x1, y1, x2, y2, label: str, color) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), b = cv2.getTextSize(label, f, s, t)
    ty = max(y1, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 2, ty + b - 2), color, -1)
    cv2.putText(img, label, (x1 + 1, ty - 2), f, s, (0, 0, 0), t, cv2.LINE_AA)


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Worst predictions for one class")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--yolov5-dir", type=Path,
                    default=Path("/mnt/datadisk/avani/yolov5"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output folder (default: <images>/../plate_errors)")
    ap.add_argument("--class-id", type=int, default=4,
                    help="Target class to analyse (default 4 = reg_plate)")
    ap.add_argument("--n", type=int, default=300,
                    help="How many worst images to write out")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.3,
                    help="IoU above which a pred is considered to match a GT")
    ap.add_argument("--good-iou", type=float, default=0.5,
                    help="Matched below this IoU counts as loose_box")
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--split-by-type", action="store_true",
                    help="Write overlays into missed/ spurious/ loose/ subfolders")
    args = ap.parse_args()

    target = args.class_id
    target_name = CLASS_NAMES.get(target, f"class_{target}")

    out_dir = args.out or (args.images.parent / "plate_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.split_by_type:
        for sub in ("missed", "spurious", "loose_box", "wrong_class"):
            (out_dir / sub).mkdir(parents=True, exist_ok=True)

    images = iter_images(args.images)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images in {args.images}")

    if not (args.yolov5_dir / "hubconf.py").is_file():
        raise SystemExit(f"yolov5 repo not found at {args.yolov5_dir}")
    device = ("cuda:" + args.device) if args.device not in ("cpu", "") else "cpu"
    print(f"Loading model via {args.yolov5_dir}")
    model = torch.hub.load(str(args.yolov5_dir), "custom",
                           path=str(args.weights), source="local", verbose=False)
    model.to(device)
    model.conf = args.conf
    model.iou = args.iou
    print("Model classes:", getattr(model, "names", None))
    print(f"Analysing class {target} ({target_name})\n")

    rows: List[dict] = []
    totals = {"missed": 0, "spurious": 0, "loose_box": 0, "wrong_class": 0,
              "matched_good": 0, "gt_total": 0, "pred_total": 0}

    progress = tqdm(total=len(images), desc="Scoring", unit="img")
    for batch in batched(images, args.batch):
        loaded = []
        for p in batch:
            im = cv2.imread(str(p))
            if im is None:
                progress.update(1)
                continue
            loaded.append((p, im, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if not loaded:
            continue

        results = model([lo[2] for lo in loaded], size=args.img)

        for i, (img_path, image, _rgb) in enumerate(loaded):
            det = results.xyxy[i].cpu().numpy()   # [x1,y1,x2,y2,conf,cls]
            h, w = image.shape[:2]
            gt_all = load_gt(args.labels / (img_path.stem + ".txt"), w, h)

            gt_t = [g for g in gt_all if g[0] == target]
            pred_t = [d for d in det if int(d[5]) == target]
            pred_other = [d for d in det if int(d[5]) != target]

            totals["gt_total"] += len(gt_t)
            totals["pred_total"] += len(pred_t)

            used_pred = set()
            missed = loose = wrong_class = 0
            notes: List[Tuple] = []   # (kind, box, label)

            # Match each GT plate to its best unused predicted plate.
            for g in gt_t:
                gbox = g[1:5]
                best_j, best_iou = -1, 0.0
                for j, d in enumerate(pred_t):
                    if j in used_pred:
                        continue
                    v = iou_xyxy(gbox, d[:4])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0 and best_iou >= args.match_iou:
                    used_pred.add(best_j)
                    d = pred_t[best_j]
                    if best_iou < args.good_iou:
                        loose += 1
                        notes.append(("loose_box", d[:4],
                                      f"{target_name} {d[4]:.2f} IoU{best_iou:.2f}"))
                    else:
                        totals["matched_good"] += 1
                        notes.append(("ok", d[:4],
                                      f"{target_name} {d[4]:.2f} IoU{best_iou:.2f}"))
                else:
                    # No plate prediction matched. Did another class cover it?
                    ob_j, ob_iou = -1, 0.0
                    for j, d in enumerate(pred_other):
                        v = iou_xyxy(gbox, d[:4])
                        if v > ob_iou:
                            ob_iou, ob_j = v, j
                    if ob_j >= 0 and ob_iou >= args.match_iou:
                        wrong_class += 1
                        d = pred_other[ob_j]
                        cn = CLASS_NAMES.get(int(d[5]), f"class_{int(d[5])}")
                        notes.append(("wrong_class", d[:4],
                                      f"pred {cn} {d[4]:.2f}"))
                    else:
                        missed += 1
                notes.append(("gt", gbox, f"GT {target_name}"))

            # Predicted plates that matched no GT plate.
            spurious = 0
            for j, d in enumerate(pred_t):
                if j in used_pred:
                    continue
                spurious += 1
                notes.append(("spurious", d[:4],
                              f"FP {target_name} {d[4]:.2f}"))

            totals["missed"] += missed
            totals["spurious"] += spurious
            totals["loose_box"] += loose
            totals["wrong_class"] += wrong_class

            badness = (W_MISSED * missed + W_SPURIOUS * spurious
                       + W_WRONG_CLASS * wrong_class + W_LOOSE * loose)
            if badness > 0:
                rows.append({
                    "image": img_path.name,
                    "path": str(img_path),
                    "badness": badness,
                    "missed": missed,
                    "spurious": spurious,
                    "loose_box": loose,
                    "wrong_class": wrong_class,
                    "gt_plates": len(gt_t),
                    "pred_plates": len(pred_t),
                    "notes": notes,
                    "shape": (h, w),
                })
            progress.update(1)
    progress.close()

    rows.sort(key=lambda r: (-r["badness"], -r["missed"]))

    # Write overlays for the worst N.
    print(f"\nWriting {min(args.n, len(rows))} worst images -> {out_dir}")
    for r in tqdm(rows[: args.n], desc="Drawing", unit="img"):
        image = cv2.imread(r["path"])
        if image is None:
            continue
        for kind, box, label in r["notes"]:
            if kind == "gt":
                draw(image, *box, label, GT_COLOR)
            elif kind == "wrong_class":
                draw(image, *box, label, OTHER_COLOR)
            else:
                draw(image, *box, label, PRED_COLOR)
        tag = (f"m{r['missed']}_s{r['spurious']}"
               f"_l{r['loose_box']}_w{r['wrong_class']}")
        name = f"{r['badness']:05.1f}_{tag}_{r['image']}"
        if args.split_by_type:
            if r["missed"]:
                sub = "missed"
            elif r["wrong_class"]:
                sub = "wrong_class"
            elif r["spurious"]:
                sub = "spurious"
            else:
                sub = "loose_box"
            cv2.imwrite(str(out_dir / sub / name), image)
        else:
            cv2.imwrite(str(out_dir / name), image)

    # CSV of every bad image (not just the ones drawn).
    csv_path = out_dir / "plate_errors.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["image", "badness", "missed", "spurious", "loose_box",
                       "wrong_class", "gt_plates", "pred_plates"])
        for r in rows:
            wcsv.writerow([r["image"], f"{r['badness']:.1f}", r["missed"],
                           r["spurious"], r["loose_box"], r["wrong_class"],
                           r["gt_plates"], r["pred_plates"]])

    gt_n = max(1, totals["gt_total"])
    print(f"\n============ {target_name.upper()} ERROR SUMMARY ============")
    print(f"Images analysed        : {len(images)}")
    print(f"Images with any error  : {len(rows)}")
    print(f"GT {target_name} boxes      : {totals['gt_total']}")
    print(f"Predicted {target_name}     : {totals['pred_total']}")
    print("-" * 52)
    print(f"Matched well (IoU>={args.good_iou}) : {totals['matched_good']} "
          f"({totals['matched_good'] / gt_n * 100:.2f}% of GT)")
    print(f"Loose boxes (IoU<{args.good_iou})   : {totals['loose_box']} "
          f"({totals['loose_box'] / gt_n * 100:.2f}%)")
    print(f"MISSED (no detection)        : {totals['missed']} "
          f"({totals['missed'] / gt_n * 100:.2f}%)")
    print(f"Wrong class over a GT plate  : {totals['wrong_class']} "
          f"({totals['wrong_class'] / gt_n * 100:.2f}%)")
    print(f"Spurious (false positives)   : {totals['spurious']}")
    print("-" * 52)
    print(f"Overlays -> {out_dir}")
    print(f"CSV      -> {csv_path}")
    print("Legend: GREEN = ground truth, RED = predicted plate, "
          "YELLOW = other class over a GT plate")
    print("=" * 52)


if __name__ == "__main__":
    main()
