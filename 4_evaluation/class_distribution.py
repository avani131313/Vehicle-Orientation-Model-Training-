#!/usr/bin/env python3
"""class_distribution.py

Report the per-class distribution of a YOLO dataset.

For a folder of YOLO label .txt files it counts, per class:
  * BOXES  - total bounding-box instances of that class
  * IMAGES - how many images contain at least one box of that class
            (an image with 2 cars + 1 bike counts once for car, once for bike)

Also prints totals and how many images/labels were empty or unreadable.

Default target is the cleaned training set's labels:
    /mnt/datadisk/avani/front_back/main/Dataset/labels

Run:
    python class_distribution.py
    python class_distribution.py --labels /path/to/labels
    python class_distribution.py --labels Dataset/yolo_split/labels/train
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Class id -> name (matches your 4-class front/back set).
CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "non_bike_front",
    3: "non_bike_back",
}

DEFAULT_LABELS = Path("/mnt/datadisk/avani/front_back/main/Dataset/labels")


def iter_label_files(labels_dir: Path) -> List[Path]:
    """Return all .txt label files under a directory (recursive)."""
    return sorted(labels_dir.rglob("*.txt"))


def class_id_of(line: str) -> int | None:
    """Parse the class id from a YOLO label line, or None if malformed."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None


def analyze(labels_dir: Path) -> None:
    """Scan labels_dir and print the per-class box/image distribution."""
    if not labels_dir.exists():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    files = iter_label_files(labels_dir)
    if not files:
        raise SystemExit(f"No .txt label files found under {labels_dir}")

    box_counts: Counter = Counter()      # class_id -> total boxes
    image_counts: Counter = Counter()    # class_id -> images containing it
    total_boxes = 0
    empty_labels = 0
    unreadable = 0
    total_images = len(files)

    for label_path in files:
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            unreadable += 1
            continue

        classes_in_image = set()
        boxes_in_image = 0
        for line in lines:
            if not line.strip():
                continue
            cid = class_id_of(line)
            if cid is None:
                continue
            box_counts[cid] += 1
            classes_in_image.add(cid)
            boxes_in_image += 1

        if boxes_in_image == 0:
            empty_labels += 1
        total_boxes += boxes_in_image
        for cid in classes_in_image:
            image_counts[cid] += 1

    # All class ids seen, plus any known names, in sorted order.
    all_ids = sorted(set(box_counts) | set(CLASS_NAMES))

    print(f"\nLabels dir : {labels_dir}")
    print(f"Label files: {total_images}")
    print(f"Total boxes: {total_boxes}")
    if empty_labels:
        print(f"Empty labels (no boxes): {empty_labels}")
    if unreadable:
        print(f"Unreadable label files : {unreadable}")

    print("\n{:<4} {:<16} {:>12} {:>14} {:>10}".format(
        "id", "class", "boxes", "images", "img %"))
    print("-" * 60)
    for cid in all_ids:
        name = CLASS_NAMES.get(cid, f"class_{cid}")
        boxes = box_counts.get(cid, 0)
        imgs = image_counts.get(cid, 0)
        pct = (imgs / total_images * 100.0) if total_images else 0.0
        print("{:<4} {:<16} {:>12} {:>14} {:>9.1f}%".format(
            cid, name, boxes, imgs, pct))
    print("-" * 60)
    print("{:<4} {:<16} {:>12} {:>14}".format(
        "", "TOTAL", total_boxes, total_images))
    print("\nNote: image % sums to >100 because one image can contain "
          "multiple classes.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO dataset class distribution")
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                    help=f"Labels directory (default: {DEFAULT_LABELS})")
    args = ap.parse_args()
    analyze(args.labels)


if __name__ == "__main__":
    main()
