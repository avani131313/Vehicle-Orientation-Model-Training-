#!/usr/bin/env python3
"""draw_labels.py

Draw GROUND-TRUTH label boxes (from .txt files) onto their images so you can
visually verify a labelling or relabelling step. No model, no inference.

Iterates the LABEL folder (not the image folder), so pointing it at a partial
label set -- e.g. a 200-file smoke test -- only renders those images.

Run:
    python draw_labels.py \
        --images Dataset3_merged/images \
        --labels Dataset3_merged/labels_7class_TEST \
        --out    Dataset3_merged/LABELS_7CLASS_VIEW
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
from tqdm import tqdm

CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "car_front",
    3: "car_back",
    4: "other_front",
    5: "other_back",
    6: "reg_plate",
}
# BGR, chosen to be easy to tell apart at a glance
CLASS_COLORS: Dict[int, tuple] = {
    0: (0, 200, 0),       # green      bike_front
    1: (0, 140, 255),     # orange     bike_back
    2: (255, 100, 0),     # blue       car_front
    3: (255, 0, 200),     # magenta    car_back
    4: (0, 255, 255),     # yellow     other_front
    5: (0, 0, 220),       # red        other_back
    6: (255, 255, 255),   # white      reg_plate
}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None


def draw_box(img, x1, y1, x2, y2, label: str, color) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), b = cv2.getTextSize(label, f, s, t)
    ty = max(y1, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 2, ty + b - 2), color, -1)
    cv2.putText(img, label, (x1 + 1, ty - 2), f, s, (0, 0, 0), t, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw ground-truth labels on images")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all label files")
    ap.add_argument("--sort-by-class", type=int, default=None,
                    help="Only render images containing this class id")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels = sorted(p for p in args.labels.iterdir() if p.suffix.lower() == ".txt")
    if args.limit and args.limit > 0:
        labels = labels[: args.limit]
    if not labels:
        raise SystemExit(f"No .txt files in {args.labels}")

    counts: Dict[int, int] = {}
    written = missing = 0

    for lbl in tqdm(labels, desc="Drawing", unit="file"):
        try:
            lines = lbl.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue

        boxes = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append((cid, cx, cy, bw, bh))

        if args.sort_by_class is not None and \
                not any(b[0] == args.sort_by_class for b in boxes):
            continue

        img_path = find_image(args.images, lbl.stem)
        if img_path is None:
            missing += 1
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            missing += 1
            continue

        h, w = image.shape[:2]
        for cid, cx, cy, bw, bh in boxes:
            counts[cid] = counts.get(cid, 0) + 1
            draw_box(image,
                     (cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h,
                     CLASS_NAMES.get(cid, f"class_{cid}"),
                     CLASS_COLORS.get(cid, (200, 200, 200)))
        cv2.imwrite(str(args.out / img_path.name), image)
        written += 1

    total = sum(counts.values())
    print(f"\nRendered {written} images ({missing} missing/unreadable) -> {args.out}")
    print("Boxes drawn per class:")
    for k in sorted(counts):
        print(f"  {k} {CLASS_NAMES.get(k, '?'):<14}: {counts[k]:>7} "
              f"({counts[k] / max(total, 1) * 100:5.2f}%)")
    print("\nColour key: bike_front=green  bike_back=orange  car_front=blue")
    print("            car_back=magenta  other_front=yellow  other_back=red")
    print("            reg_plate=white")


if __name__ == "__main__":
    main()
