"""split_dataset.py

Build a YOLOv5-style train/val/test split from the cleaned flat dataset.

Source layout (produced by the preprocessing pipeline)::

    <ROOT>/images/*.jpg
    <ROOT>/labels/*.txt

Output layout (created here, YOLOv5 convention)::

    <ROOT>/yolo_split/
        images/{train,val,test}/
        labels/{train,val,test}/
        data.yaml

Files are COPIED (originals untouched). The split is 70/20/10 by default, made
reproducible with a fixed random seed. Only image/label pairs where the label
exists and is non-empty are included; anything else is skipped and reported.

Run:

    python split_dataset.py
"""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from tqdm import tqdm

import config
import utils


# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = config.ROOT / "yolo_split"

# Ratios must sum to 1.0. (train, val, test)
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.20
TEST_RATIO: float = 0.10

# Fixed seed -> reproducible shuffling/splitting.
RANDOM_SEED: int = 42

SPLITS: Tuple[str, str, str] = ("train", "val", "test")

# Human-readable class names, keyed by YOLO class ID (nc = 4).
CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "non_bike_front",
    3: "non_bike_back",
}


@dataclass
class SplitStats:
    """Counters describing the split result."""

    total_images: int = 0
    paired: int = 0
    missing_label: int = 0
    empty_label: int = 0
    per_split: Dict[str, int] = field(default_factory=dict)
    class_ids: Set[int] = field(default_factory=set)


def _make_output_dirs() -> None:
    """Create images/ and labels/ subfolders for every split."""
    for split in SPLITS:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def _collect_pairs(stats: SplitStats) -> List[Path]:
    """Return image paths that have a non-empty matching label file.

    Also records every class ID seen, so ``data.yaml`` can be generated with
    the correct ``nc``. Streams the image directory to keep memory flat.
    """
    total = utils.count_images(config.IMAGE_DIR)
    stats.total_images = total

    pairs: List[Path] = []
    for image_path in tqdm(
        utils.iter_image_paths(config.IMAGE_DIR),
        total=total,
        desc="Scanning pairs",
        unit="img",
    ):
        label_path = utils.label_path_for_image(image_path)
        if not label_path.exists():
            stats.missing_label += 1
            continue

        boxes = utils.parse_yolo_label(label_path)
        if not boxes:
            stats.empty_label += 1
            continue

        for box in boxes:
            stats.class_ids.add(box.cls)
        pairs.append(image_path)

    stats.paired = len(pairs)
    return pairs


def _assign_splits(pairs: Sequence[Path]) -> Dict[str, List[Path]]:
    """Shuffle deterministically and partition into train/val/test."""
    shuffled = list(pairs)
    random.Random(RANDOM_SEED).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    # test takes the remainder so the counts always sum exactly to n_total.
    assignment: Dict[str, List[Path]] = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
    return assignment


def _copy_split(split: str, image_paths: Sequence[Path]) -> int:
    """Copy every image + its label into the split's folders. Returns count."""
    dest_images = OUTPUT_DIR / "images" / split
    dest_labels = OUTPUT_DIR / "labels" / split

    copied = 0
    for image_path in tqdm(image_paths, desc=f"Copying {split}", unit="img"):
        label_path = utils.label_path_for_image(image_path)
        try:
            shutil.copy2(str(image_path), str(dest_images / image_path.name))
            shutil.copy2(str(label_path), str(dest_labels / label_path.name))
            copied += 1
        except (OSError, shutil.Error):
            # Skip unreadable/vanished files rather than aborting the whole run.
            continue
    return copied


def _resolve_class_names(class_ids: Set[int]) -> List[str]:
    """Build an ordered class-name list covering IDs 0..max(class_ids).

    Uses CLASS_NAMES where provided, otherwise falls back to ``class_<id>``.
    A contiguous 0-based list is required by YOLOv5's ``names`` field.
    """
    if not class_ids:
        return ["class_0"]
    max_id = max(class_ids)
    return [CLASS_NAMES.get(i, f"class_{i}") for i in range(max_id + 1)]


def _write_data_yaml(class_names: Sequence[str]) -> Path:
    """Write ``data.yaml`` for YOLOv5 and return its path."""
    yaml_path = OUTPUT_DIR / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        f"# Auto-generated by split_dataset.py\n"
        f"path: {OUTPUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    """Scan pairs, split 70/20/10, copy files, and emit data.yaml."""
    logger = utils.setup_logger("split")
    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    logger.info("Splitting dataset from %s into %s", config.IMAGE_DIR, OUTPUT_DIR)
    _make_output_dirs()

    stats = SplitStats()
    pairs = _collect_pairs(stats)
    if not pairs:
        logger.error("No valid image/label pairs found under %s", config.IMAGE_DIR)
        print("No valid image/label pairs found. Nothing to split.")
        return

    assignment = _assign_splits(pairs)
    for split in SPLITS:
        count = _copy_split(split, assignment[split])
        stats.per_split[split] = count
        logger.info("Copied %d images into %s split", count, split)

    class_names = _resolve_class_names(stats.class_ids)
    yaml_path = _write_data_yaml(class_names)

    summary = (
        "\n==================== SPLIT SUMMARY ====================\n"
        f"Images scanned      : {stats.total_images}\n"
        f"Valid pairs         : {stats.paired}\n"
        f"Skipped (no label)  : {stats.missing_label}\n"
        f"Skipped (empty)     : {stats.empty_label}\n"
        f"train / val / test  : "
        f"{stats.per_split.get('train', 0)} / "
        f"{stats.per_split.get('val', 0)} / "
        f"{stats.per_split.get('test', 0)}\n"
        f"Classes (nc={len(class_names)}): {class_names}\n"
        f"data.yaml           : {yaml_path}\n"
        "======================================================"
    )
    logger.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
