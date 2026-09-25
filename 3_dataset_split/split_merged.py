#!/usr/bin/env python3
"""split_merged.py

Build a YOLOv5 train/val/test split from the merged 5-class dataset
(Dataset_merged), after plate labels (class 4) have been added.

Reads:
    Dataset_merged/images/*.jpg   (symlinks or files)
    Dataset_merged/labels/*.txt   (vehicle boxes 0-3 + plate boxes 4)

Writes a 70/20/10 split (files COPIED; symlinked images are materialized into
real files) plus a 5-class data.yaml:
    Dataset_merged/yolo_split/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml

Run:
    python split_merged.py
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm

BASE = Path("/mnt/datadisk/avani/front_back/main")
SRC_IMAGES = BASE / "Dataset_merged" / "images"
SRC_LABELS = BASE / "Dataset_merged" / "labels"
OUTPUT_DIR = BASE / "Dataset_merged" / "yolo_split"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RANDOM_SEED = 42
SPLITS = ("train", "val", "test")

CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "non_bike_front",
    3: "non_bike_back",
    4: "reg_plate",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def label_nonempty_classes(label_path: Path, seen: Set[int]) -> bool:
    """Return True if the label has >=1 valid box; record class ids into seen."""
    if not label_path.exists():
        return False
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    valid = False
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            seen.add(int(float(parts[0])))
            valid = True
        except ValueError:
            continue
    return valid


def make_dirs() -> None:
    for s in SPLITS:
        (OUTPUT_DIR / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / s).mkdir(parents=True, exist_ok=True)


def write_data_yaml(class_ids: Set[int]) -> Path:
    max_id = max(class_ids | set(CLASS_NAMES))
    names = [CLASS_NAMES.get(i, f"class_{i}") for i in range(max_id + 1)]
    block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    yaml = OUTPUT_DIR / "data.yaml"
    yaml.write_text(
        f"path: {OUTPUT_DIR}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(names)}\nnames:\n{block}\n",
        encoding="utf-8",
    )
    return yaml


def main() -> None:
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        raise SystemExit("ratios must sum to 1.0")
    if not SRC_IMAGES.exists():
        raise SystemExit(f"Missing {SRC_IMAGES} (run merge_datasets.py first)")

    make_dirs()

    images = iter_images(SRC_IMAGES)
    class_ids: Set[int] = set()
    pairs: List[Path] = []
    missing = 0
    for img in tqdm(images, desc="Scanning pairs", unit="img"):
        lbl = SRC_LABELS / (img.stem + ".txt")
        if label_nonempty_classes(lbl, class_ids):
            pairs.append(img)
        else:
            missing += 1

    if not pairs:
        raise SystemExit("No valid image/label pairs found")

    random.Random(RANDOM_SEED).shuffle(pairs)
    n = len(pairs)
    n_tr = int(n * TRAIN_RATIO)
    n_va = int(n * VAL_RATIO)
    assign = {
        "train": pairs[:n_tr],
        "val": pairs[n_tr:n_tr + n_va],
        "test": pairs[n_tr + n_va:],
    }

    counts = {}
    for s in SPLITS:
        di = OUTPUT_DIR / "images" / s
        dl = OUTPUT_DIR / "labels" / s
        c = 0
        for img in tqdm(assign[s], desc=f"Copying {s}", unit="img"):
            lbl = SRC_LABELS / (img.stem + ".txt")
            try:
                shutil.copy2(str(img), str(di / img.name))   # follows symlink
                shutil.copy2(str(lbl), str(dl / lbl.name))
                c += 1
            except (OSError, shutil.Error):
                continue
        counts[s] = c

    yaml = write_data_yaml(class_ids)
    names = [CLASS_NAMES.get(i, f"class_{i}")
             for i in range(max(class_ids | set(CLASS_NAMES)) + 1)]

    print("\n==================== MERGED SPLIT SUMMARY ====================")
    print(f"Valid pairs     : {len(pairs)}")
    print(f"Skipped (no/empty label): {missing}")
    print(f"train/val/test  : {counts['train']} / {counts['val']} / {counts['test']}")
    print(f"Classes seen    : {sorted(class_ids)}")
    print(f"nc={len(names)} names={names}")
    print(f"data.yaml       : {yaml}")
    print("=============================================================")


if __name__ == "__main__":
    main()
