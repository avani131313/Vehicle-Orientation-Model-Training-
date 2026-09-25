#!/usr/bin/env python3
"""filter_brightness.py

Separate extremely dark images (where front/back is undecidable) from usable
ones, using average image brightness.

Two modes:

  ANALYZE (default): score every image's brightness (0-255), write a CSV, print
  the distribution, and copy a few sample images from each brightness band into
  <base>/brightness_review/<band>/ so you can SEE where "too dark to classify"
  begins and choose a threshold.

  APPLY (--apply --threshold T): move images with brightness < T into
  <base>/too_dark/, leaving the rest in place.

Brightness is the mean of a 1/8-scale grayscale read (fast, low memory), so it
scales to hundreds of thousands of images.

Run:
    python filter_brightness.py --images dataset3_images
    python filter_brightness.py --images dataset3_images --apply --threshold 40
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Review bands (upper bounds); the last catches everything brighter.
BANDS = [10, 20, 30, 40, 50, 70, 100, 256]
SAMPLES_PER_BAND = 25


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def brightness(img_path: Path) -> float:
    """Mean brightness (0-255) from a fast 1/8-scale grayscale read; -1 on fail."""
    im = cv2.imread(str(img_path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if im is None:
        im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return -1.0
    return float(im.mean())


def band_label(value: float) -> str:
    lo = 0
    for hi in BANDS:
        if value < hi:
            return f"{lo:03d}-{hi:03d}"
        lo = hi
    return f"{BANDS[-2]:03d}-plus"


def analyze(images: List[Path], base: Path, csv_path: Path) -> None:
    """Score brightness, write CSV, print distribution, copy review samples."""
    review = base / "brightness_review"
    scores: List[Tuple[Path, float]] = []
    band_counts: dict = {}
    band_copied: dict = {}

    for p in tqdm(images, desc="Scoring brightness", unit="img"):
        b = brightness(p)
        scores.append((p, b))
        if b < 0:
            continue
        lbl = band_label(b)
        band_counts[lbl] = band_counts.get(lbl, 0) + 1
        if band_copied.get(lbl, 0) < SAMPLES_PER_BAND:
            d = review / lbl
            d.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(p), str(d / p.name))
                band_copied[lbl] = band_copied.get(lbl, 0) + 1
            except OSError:
                pass

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "brightness"])
        for p, b in scores:
            w.writerow([p.name, f"{b:.2f}"])

    valid = np.array([b for _, b in scores if b >= 0])
    print("\n==================== BRIGHTNESS DISTRIBUTION ====================")
    print(f"Images scored : {len(valid)}  (unreadable: {len(scores) - len(valid)})")
    if len(valid):
        for pct in (1, 5, 10, 25, 50, 75, 90):
            print(f"  {pct:>2}th percentile: {np.percentile(valid, pct):6.1f}")
        print(f"  mean: {valid.mean():.1f}   min: {valid.min():.1f}   max: {valid.max():.1f}")
    print("\nCount per band (mean brightness):")
    lo = 0
    for hi in BANDS:
        lbl = f"{lo:03d}-{hi:03d}"
        print(f"  {lbl}: {band_counts.get(lbl, 0)}")
        lo = hi
    print(f"\nReview samples -> {review}")
    print(f"CSV            -> {csv_path}")
    print("Open the low bands (e.g. 000-010, 010-020, 020-030) to see where")
    print("front/back becomes undecidable, then pick a --threshold and --apply.")
    print("================================================================")


def apply_split(images: List[Path], base: Path, threshold: float) -> None:
    """Move images dimmer than threshold into <base>/too_dark/."""
    dark_dir = base / "too_dark"
    dark_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in tqdm(images, desc=f"Splitting < {threshold}", unit="img"):
        b = brightness(p)
        if 0 <= b < threshold:
            try:
                shutil.move(str(p), str(dark_dir / p.name))
                moved += 1
            except (OSError, shutil.Error):
                pass
    print(f"\nMoved {moved} too-dark images -> {dark_dir}")
    print(f"Remaining in {base.name}: brighter images stay for labeling.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter images by brightness")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Move too-dark images (requires --threshold)")
    ap.add_argument("--threshold", type=float, default=40.0,
                    help="Mean-brightness cutoff for --apply (0-255)")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    images = iter_images(args.images)
    if not images:
        raise SystemExit(f"No images in {args.images}")
    base = args.images if args.images.is_dir() else args.images.parent
    csv_path = args.csv or (base.parent / f"{base.name}_brightness.csv")

    if args.apply:
        apply_split(images, base, args.threshold)
    else:
        analyze(images, base, csv_path)


if __name__ == "__main__":
    main()
