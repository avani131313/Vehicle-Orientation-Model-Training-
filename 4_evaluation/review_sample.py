#!/usr/bin/env python3
"""review_sample.py

Score the SAME random sample that infer_draw.py drew, count how many images
have bad predictions, and split them into good/ and bad/ folders.

Uses identical sampling logic to infer_draw.py (sorted listing -> seeded
random.sample), so passing the same --n / --seed reviews exactly the images
you just looked at.

An image is "bad" if, comparing predictions to ground truth, it has any of:
  missed        - a GT box with no matching prediction        (false negative)
  spurious      - a prediction matching no GT box             (false positive)
  wrong_class   - matched a GT box but predicted the wrong class
  loose_box     - matched with correct class but IoU < --good-iou

Output:
  <out>/good/   overlaid images with no errors
  <out>/bad/    overlaid images with >=1 error, filename prefixed by badness
  <out>/review_sample.csv   per-image counts for every sampled image

Overlay colours: GREEN = ground truth, RED = prediction.

Run:
    python review_sample.py \
        --weights /mnt/datadisk/avani/front_back/main/BESTPT/4_August_new_model/best.pt \
        --images  Dataset3_merged/yolo_split/test/images \
        --labels  Dataset3_merged/yolo_split/test/labels \
        --n 1000 --seed 42 --conf 0.65 --img 320
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
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

GT_COLOR = (0, 200, 0)      # green
PRED_COLOR = (0, 0, 220)    # red

W_MISSED, W_SPURIOUS, W_WRONG, W_LOOSE = 3.0, 2.0, 2.0, 1.0


def iter_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_gt(label_path: Path, w: int, h: int,
            class_map: Dict[int, int] | None = None
            ) -> List[Tuple[int, float, float, float, float]]:
    """Read a YOLO label file. class_map remaps GT class ids IN MEMORY only,
    so an external dataset never has to be modified on disk."""
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


def parse_class_map(spec: str | None) -> Dict[int, int]:
    """Parse '0:4' or '0:4,1:2' into {0: 4} / {0: 4, 1: 2}."""
    if not spec:
        return {}
    out: Dict[int, int] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        try:
            src, dst = pair.split(":")
            out[int(src)] = int(dst)
        except ValueError:
            raise SystemExit(f"Bad --gt-class-map entry: {pair!r} (expected 'src:dst')")
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


def draw(img, x1, y1, x2, y2, label: str, color) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), b = cv2.getTextSize(label, f, s, t)
    ty = max(y1, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 2, ty + b - 2), color, -1)
    cv2.putText(img, label, (x1 + 1, ty - 2), f, s, (0, 0, 0), t, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Count bad predictions in a sample")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--yolov5-dir", type=Path,
                    default=Path("/mnt/datadisk/avani/yolov5"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output folder (default: <images>/../REVIEW_<n>)")
    ap.add_argument("--n", type=int, default=1000, help="Sample size")
    ap.add_argument("--seed", type=int, default=42,
                    help="Must match infer_draw.py to review the same images")
    ap.add_argument("--conf", type=float, default=0.65)
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    ap.add_argument("--match-iou", type=float, default=0.3,
                    help="IoU above which a prediction is linked to a GT box")
    ap.add_argument("--good-iou", type=float, default=0.5,
                    help="Linked below this IoU counts as a loose box")
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--only-class", type=int, default=None,
                    help="Score only this class id (e.g. 4 for reg_plate)")
    ap.add_argument("--gt-class-map", type=str, default=None,
                    help="Remap GT class ids in memory, e.g. '0:4'. The label "
                         "files on disk are never modified.")
    args = ap.parse_args()

    class_map = parse_class_map(args.gt_class_map)
    if class_map:
        print(f"Remapping GT classes (in memory): {class_map}")

    out_dir = args.out or (args.images.parent / f"REVIEW_{args.n}")
    good_dir, bad_dir = out_dir / "good", out_dir / "bad"
    good_dir.mkdir(parents=True, exist_ok=True)
    bad_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.images)
    if not images:
        raise SystemExit(f"No images in {args.images}")
    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.n, len(images)))

    if not (args.yolov5_dir / "hubconf.py").is_file():
        raise SystemExit(f"yolov5 repo not found at {args.yolov5_dir}")
    device = ("cuda:" + args.device) if args.device not in ("cpu", "") else "cpu"
    print(f"Loading model via {args.yolov5_dir}")
    model = torch.hub.load(str(args.yolov5_dir), "custom",
                           path=str(args.weights), source="local", verbose=False)
    model.to(device)
    model.conf = args.conf
    model.iou = args.iou
    raw = getattr(model, "names", None)
    if isinstance(raw, dict):
        names = {int(k): str(v) for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        names = {i: str(v) for i, v in enumerate(raw)}
    else:
        names = dict(CLASS_NAMES)
    print("Model classes:", names)
    if args.only_class is not None:
        print(f"Scoring ONLY class {args.only_class} "
              f"({names.get(args.only_class, '?')})")

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    rows: List[dict] = []
    tot = {"missed": 0, "spurious": 0, "wrong_class": 0, "loose_box": 0,
           "correct": 0, "gt": 0, "pred": 0}
    n_bad = n_good = 0

    progress = tqdm(total=len(sample), desc="Reviewing", unit="img")
    for chunk in batched(sample, args.batch):
        loaded = []
        for p in chunk:
            im = cv2.imread(str(p))
            if im is None:
                progress.update(1)
                continue
            loaded.append((p, im, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if not loaded:
            continue

        results = model([lo[2] for lo in loaded], size=args.img)

        for i, (img_path, image, _rgb) in enumerate(loaded):
            det = results.xyxy[i].cpu().numpy()
            h, w = image.shape[:2]
            gt = load_gt(args.labels / (img_path.stem + ".txt"), w, h, class_map)
            preds = [d for d in det]

            if args.only_class is not None:
                gt = [g for g in gt if g[0] == args.only_class]
                preds = [d for d in preds if int(d[5]) == args.only_class]

            tot["gt"] += len(gt)
            tot["pred"] += len(preds)

            used = set()
            missed = wrong = loose = 0
            overlay = image.copy()

            for g in gt:
                gcls, gbox = g[0], g[1:5]
                draw(overlay, *gbox, f"GT {names.get(gcls, gcls)}", GT_COLOR)
                best_j, best_iou = -1, 0.0
                for j, d in enumerate(preds):
                    if j in used:
                        continue
                    v = iou_xyxy(gbox, d[:4])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_j < 0 or best_iou < args.match_iou:
                    missed += 1
                    continue
                used.add(best_j)
                d = preds[best_j]
                if int(d[5]) != gcls:
                    wrong += 1
                elif best_iou < args.good_iou:
                    loose += 1
                else:
                    tot["correct"] += 1

            spurious = 0
            for j, d in enumerate(preds):
                cn = names.get(int(d[5]), f"class_{int(d[5])}")
                draw(overlay, *d[:4], f"{cn} {d[4]:.2f}", PRED_COLOR)
                if j not in used:
                    spurious += 1

            tot["missed"] += missed
            tot["spurious"] += spurious
            tot["wrong_class"] += wrong
            tot["loose_box"] += loose

            badness = (W_MISSED * missed + W_SPURIOUS * spurious
                       + W_WRONG * wrong + W_LOOSE * loose)
            rows.append({"image": img_path.name, "badness": badness,
                         "missed": missed, "spurious": spurious,
                         "wrong_class": wrong, "loose_box": loose,
                         "gt": len(gt), "pred": len(preds)})

            if badness > 0:
                n_bad += 1
                tag = f"m{missed}_s{spurious}_w{wrong}_l{loose}"
                cv2.imwrite(str(bad_dir /
                                f"{badness:05.1f}_{tag}_{img_path.name}"), overlay)
            else:
                n_good += 1
                cv2.imwrite(str(good_dir / img_path.name), overlay)
            progress.update(1)
    progress.close()

    rows.sort(key=lambda r: -r["badness"])
    csv_path = out_dir / "review_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["image", "badness", "missed", "spurious",
                       "wrong_class", "loose_box", "gt_boxes", "pred_boxes"])
        for r in rows:
            wcsv.writerow([r["image"], f"{r['badness']:.1f}", r["missed"],
                           r["spurious"], r["wrong_class"], r["loose_box"],
                           r["gt"], r["pred"]])

    n = max(1, len(rows))
    gt_n = max(1, tot["gt"])
    print("\n==================== SAMPLE REVIEW ====================")
    print(f"Images reviewed   : {len(rows)}   (seed {args.seed}, conf {args.conf})")
    print(f"CLEAN images      : {n_good}  ({n_good / n * 100:.1f}%)")
    print(f"BAD images        : {n_bad}  ({n_bad / n * 100:.1f}%)")
    print("-" * 55)
    print(f"GT boxes          : {tot['gt']}")
    print(f"Predicted boxes   : {tot['pred']}")
    print(f"Correct           : {tot['correct']}  ({tot['correct'] / gt_n * 100:.2f}% of GT)")
    print(f"Missed (FN)       : {tot['missed']}  ({tot['missed'] / gt_n * 100:.2f}%)")
    print(f"Wrong class       : {tot['wrong_class']}  ({tot['wrong_class'] / gt_n * 100:.2f}%)")
    print(f"Loose box         : {tot['loose_box']}  ({tot['loose_box'] / gt_n * 100:.2f}%)")
    print(f"Spurious (FP)     : {tot['spurious']}")
    print("-" * 55)
    print(f"good/ -> {good_dir}")
    print(f"bad/  -> {bad_dir}   (worst sort to top by filename)")
    print(f"CSV   -> {csv_path}")
    print("=" * 55)


if __name__ == "__main__":
    main()
