#!/usr/bin/env python3
"""split_dataset3.py

Split the final merged 5-class dataset (dataset3_merged) into
train/val/test = 75/10/15.

IMPORTANT: dataset3_merged's files are currently in MERGE ORDER (all of
source A, then all of source B, etc.) — NOT random. Splitting on that raw
order would put entire sources into a single split (e.g. test set being
100% one source). This script does a SEEDED SHUFFLE first so every split
gets a representative mix of every source.

Images are SYMLINKED into the split folders (dataset is large; symlinks
avoid duplicating ~500k+ image files on disk). Labels are COPIED (they are
tiny text files and some pipelines mutate them in place, so each split
needs its own independent copy).

Classes (5): bike_front, bike_back, non_bike_front, non_bike_back, reg_plate

Run:
    python split_dataset3.py --base /mnt/datadisk/avani/front_back/main/dataset3_merged
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "non_bike_front",
    3: "non_bike_back",
    4: "reg_plate",
}

TRAIN_RATIO = 0.75
VAL_RATIO = 0.10
TEST_RATIO = 0.15
RANDOM_SEED = 42


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def gather_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Only keep images that have a matching, non-empty label file."""
    pairs: List[Tuple[Path, Path]] = []
    skipped_no_label = 0
    skipped_empty = 0
    for img in iter_images(images_dir):
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            skipped_no_label += 1
            continue
        if lbl.stat().st_size == 0:
            skipped_empty += 1
            continue
        pairs.append((img, lbl))
    print(f"Found {len(pairs)} valid image/label pairs "
          f"(skipped {skipped_no_label} missing-label, "
          f"{skipped_empty} empty-label)")
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]]
                ) -> Dict[str, List[Tuple[Path, Path]]]:
    rng = random.Random(RANDOM_SEED)
    shuffled = pairs[:]
    rng.shuffle(shuffled)  # <-- de-correlates merge order from split

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # test takes the remainder so rounding never drops an image
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def materialize_split(split_name: str, pairs: List[Tuple[Path, Path]],
                      out_dir: Path, symlink_images: bool) -> None:
    img_out = out_dir / split_name / "images"
    lbl_out = out_dir / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img, lbl in pairs:
        dst_img = img_out / img.name
        if symlink_images:
            try:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                dst_img.symlink_to(img.resolve())
            except OSError:
                shutil.copy2(img, dst_img)  # fallback if symlinks unsupported
        else:
            shutil.copy2(img, dst_img)
        shutil.copy2(lbl, lbl_out / lbl.name)


def write_data_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    lines = [
        f"train: {out_dir / 'train' / 'images'}",
        f"val: {out_dir / 'val' / 'images'}",
        f"test: {out_dir / 'test' / 'images'}",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i in range(len(CLASS_NAMES)):
        lines.append(f"  - {CLASS_NAMES[i]}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description="75/10/15 split of dataset3_merged")
    ap.add_argument("--base", type=Path, required=True,
                    help="dataset3_merged folder (must contain images/ and labels/)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <base>/yolo_split)")
    ap.add_argument("--copy-images", action="store_true",
                    help="Copy images instead of symlinking (uses far more disk)")
    args = ap.parse_args()

    base = args.base.resolve()
    images_dir = base / "images"
    labels_dir = base / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to exist")

    out_dir = (args.out.resolve() if args.out else (base / "yolo_split"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = gather_pairs(images_dir, labels_dir)
    if not pairs:
        raise SystemExit("No valid image/label pairs found")

    splits = split_pairs(pairs)

    for name, split_pairs_list in splits.items():
        print(f"Writing {name}: {len(split_pairs_list)} images "
              f"({len(split_pairs_list) / len(pairs) * 100:.1f}%)")
        materialize_split(name, split_pairs_list, out_dir, symlink_images=not args.copy_images)

    yaml_path = write_data_yaml(out_dir)

    print("\n==================== SPLIT SUMMARY ====================")
    print(f"Total pairs : {len(pairs)}")
    print(f"Train       : {len(splits['train'])} ({TRAIN_RATIO*100:.0f}%)")
    print(f"Val         : {len(splits['val'])} ({VAL_RATIO*100:.0f}%)")
    print(f"Test        : {len(splits['test'])} ({TEST_RATIO*100:.0f}%)")
    print(f"Output      : {out_dir}")
    print(f"data.yaml   : {yaml_path}")
    print("=========================================================")


if __name__ == "__main__":
    main()
