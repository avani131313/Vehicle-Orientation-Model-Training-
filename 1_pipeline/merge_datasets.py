#!/usr/bin/env python3
"""merge_datasets.py

Merge the original cleaned dataset with the auto-labeled night sets into one
combined dataset, ready for adding the plate class and re-splitting.

For each source (images_dir, labels_dir) it takes every image that has a
matching non-empty label and adds it to <out>/images + <out>/labels:

  * images are SYMLINKED  (no data duplication; hundreds of GB saved)
  * labels are COPIED     (real files, so plate boxes can be appended later
                           without touching the originals)

Filename collisions across sources are skipped and reported.

Default sources:
    Dataset            (original cleaned 4-class set)
    night_dataset      (back)
    night_dataset_2    (back, after the front->back remap)

Run:
    python merge_datasets.py
    python merge_datasets.py --copy-images   # copy instead of symlink
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

BASE = Path("/mnt/datadisk/avani/front_back/main")

SOURCES: List[Tuple[Path, Path]] = [
    (BASE / "Dataset" / "images", BASE / "Dataset" / "labels"),
    (BASE / "night_dataset" / "images", BASE / "night_dataset" / "labels"),
]

OUT_DIR = BASE / "Dataset_merged"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def label_nonempty(label_path: Path) -> bool:
    """True if the label file exists and has at least one non-blank line."""
    if not label_path.exists():
        return False
    try:
        return any(line.strip() for line in label_path.read_text(
            encoding="utf-8").splitlines())
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge datasets into one folder")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--copy-images", action="store_true",
                    help="Copy images instead of symlinking (uses more disk)")
    args = ap.parse_args()

    out_images = args.out / "images"
    out_labels = args.out / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    total_added = 0
    total_skipped_nolabel = 0
    total_collisions = 0

    for images_dir, labels_dir in SOURCES:
        if not images_dir.exists():
            print(f"[skip] source missing: {images_dir}")
            continue
        imgs = [p for p in sorted(images_dir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        added = skipped = collide = 0
        for img in tqdm(imgs, desc=f"Merging {images_dir.parent.name}", unit="img"):
            label = labels_dir / (img.stem + ".txt")
            if not label_nonempty(label):
                skipped += 1
                continue

            dst_img = out_images / img.name
            dst_lbl = out_labels / label.name
            if dst_img.exists() or dst_lbl.exists():
                collide += 1
                continue

            try:
                if args.copy_images:
                    shutil.copy2(str(img), str(dst_img))
                else:
                    os.symlink(img.resolve(), dst_img)
                shutil.copy2(str(label), str(dst_lbl))
                added += 1
            except (OSError, shutil.Error):
                collide += 1

        print(f"  {images_dir.parent.name}: added {added}, "
              f"no-label {skipped}, collisions {collide}")
        total_added += added
        total_skipped_nolabel += skipped
        total_collisions += collide

    print("\n==================== MERGE SUMMARY ====================")
    print(f"Merged images : {total_added}")
    print(f"Skipped (no label): {total_skipped_nolabel}")
    print(f"Collisions skipped: {total_collisions}")
    print(f"Output        : {args.out}")
    print(f"  images/ (symlinks unless --copy-images), labels/ (copies)")
    print("======================================================")


if __name__ == "__main__":
    main()
