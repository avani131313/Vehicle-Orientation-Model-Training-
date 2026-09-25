#!/usr/bin/env python3
"""add_dataset.py

One-command orchestrator to fold a NEW data batch into the merged dataset.

Given a transactions CSV (or an already-downloaded image folder), it runs:

    1. extract image URLs from the CSV        (-> <name>_urls.txt)
    2. download them in parallel, resumable   (-> <name>_images/)
    3. label with the correct orientation      (-> <name>_dataset/ via make_yolo*)
    4. merge into Dataset_merged/              (symlink images, copy labels)
    5. (optional) append plate boxes (class 4) to the merged labels

After it finishes you just run:  split_merged.py  then  train.

Examples:
    # new BACK batch from a CSV
    python add_dataset.py --csv batch3.csv --name batch3 --orientation back

    # new FRONT batch from an existing folder, skip re-download
    python add_dataset.py --images-dir /path/to/imgs --name batch4 --orientation front

    # skip the plate step (run add_plate_labels.py yourself later)
    python add_dataset.py --csv batch5.csv --name batch5 --orientation back --no-plates
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Fixed project locations
# ---------------------------------------------------------------------------
BASE = Path("/mnt/datadisk/avani/front_back/main")
MERGED_DIR = BASE / "Dataset_merged"
LABEL_SCRIPT_BACK = BASE / "make_yolo.py"
LABEL_SCRIPT_FRONT = BASE / "make_yolo_front.py"
PLATE_SCRIPT = BASE / "add_plate_labels.py"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CSV_URL_COLS = ("entry_image_url", "exit_image_url")


# ===========================================================================
# Step 1: URL extraction
# ===========================================================================
def extract_urls(csv_path: Path, out_txt: Path) -> int:
    """Write unique image URLs from the transactions CSV. Returns the count."""
    seen = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in CSV_URL_COLS:
                u = (row.get(col) or "").strip()
                if u and u.lower() != "none" and u.startswith("http"):
                    seen.add(u)
    with out_txt.open("w", encoding="utf-8") as g:
        for u in sorted(seen):
            g.write(u + "\n")
    return len(seen)


# ===========================================================================
# Step 2: download
# ===========================================================================
def _build_session(pool: int) -> requests.Session:
    s = requests.Session()
    a = HTTPAdapter(pool_connections=pool, pool_maxsize=pool * 2, max_retries=0)
    s.mount("http://", a)
    s.mount("https://", a)
    s.headers.update({"User-Agent": "add-dataset/1.0"})
    return s


def _download_one(session: requests.Session, url: str, dest: Path) -> str:
    name = os.path.basename(urlparse(url).path) or url.rsplit("/", 1)[-1]
    target = dest / name
    if target.exists() and target.stat().st_size > 0:
        return "skipped"
    tmp = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with session.get(url, timeout=30, stream=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            out.write(chunk)
            if tmp.stat().st_size == 0:
                raise IOError("empty body")
            os.replace(tmp, target)
            return "downloaded"
        except (requests.RequestException, OSError):
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if attempt < 3:
                time.sleep(attempt)
    return "failed"


def download_images(url_txt: Path, out_dir: Path, workers: int) -> None:
    """Download every URL in url_txt into out_dir (parallel, resumable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in url_txt.read_text(encoding="utf-8").splitlines()
            if u.strip().startswith("http")]
    session = _build_session(workers)
    dl = sk = fa = 0
    progress = tqdm(total=len(urls), desc="Downloading", unit="img")
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_download_one, session, u, out_dir): u for u in urls}
            for fut in as_completed(futs):
                r = fut.result()
                dl += r == "downloaded"; sk += r == "skipped"; fa += r == "failed"
                progress.update(1)
    finally:
        progress.close()
        session.close()
    print(f"  downloaded {dl}, skipped {sk}, failed {fa}")


# ===========================================================================
# Step 3: label (delegates to the existing make_yolo* scripts)
# ===========================================================================
def label_dataset(orientation: str, src_images: Path,
                  dataset_dir: Path, conf: float) -> None:
    """Run the correct labeling script for this orientation."""
    script = LABEL_SCRIPT_BACK if orientation == "back" else LABEL_SCRIPT_FRONT
    if not script.is_file():
        raise SystemExit(f"Labeling script not found: {script}")
    cmd = [
        sys.executable, str(script),
        "--source", str(src_images),
        "--dataset-dir", str(dataset_dir),
        "--conf", str(conf),
    ]
    print("  running:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(BASE)).returncode
    if rc != 0:
        raise SystemExit(f"Labeling failed (exit {rc})")


# ===========================================================================
# Step 4: merge into Dataset_merged
# ===========================================================================
def merge_into_master(dataset_dir: Path) -> None:
    """Symlink new images and copy new labels into Dataset_merged."""
    import shutil
    src_images = dataset_dir / "images"
    src_labels = dataset_dir / "labels"
    out_images = MERGED_DIR / "images"
    out_labels = MERGED_DIR / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in sorted(src_images.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    added = collide = nolabel = 0
    for img in tqdm(imgs, desc="Merging", unit="img"):
        lbl = src_labels / (img.stem + ".txt")
        if not lbl.exists():
            nolabel += 1
            continue
        di = out_images / img.name
        dl = out_labels / lbl.name
        if di.exists() or dl.exists():
            collide += 1
            continue
        try:
            os.symlink(img.resolve(), di)
            shutil.copy2(str(lbl), str(dl))
            added += 1
        except (OSError, shutil.Error):
            collide += 1
    print(f"  merged {added}, collisions {collide}, no-label {nolabel}")


# ===========================================================================
# Step 5: plate labels (optional; delegates to add_plate_labels.py)
# ===========================================================================
def add_plate_labels(workers: int) -> None:
    """Append plate boxes to the merged set (resumable; only new images)."""
    if not PLATE_SCRIPT.is_file():
        raise SystemExit(f"Plate script not found: {PLATE_SCRIPT}")
    cmd = [
        sys.executable, str(PLATE_SCRIPT),
        "--images", str(MERGED_DIR / "images"),
        "--labels", str(MERGED_DIR / "labels"),
        "--workers", str(workers),
    ]
    print("  running:", " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(BASE)).returncode
    if rc != 0:
        raise SystemExit(f"Plate labeling failed (exit {rc})")


# ===========================================================================
# Orchestration
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Fold a new data batch into Dataset_merged")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Transactions CSV to download from")
    src.add_argument("--images-dir", type=Path,
                     help="Existing folder of images (skip download)")
    ap.add_argument("--name", required=True,
                    help="Batch name (used for folder names)")
    ap.add_argument("--orientation", required=True, choices=["front", "back"])
    ap.add_argument("--conf", type=float, default=0.4, help="YOLO conf for labeling")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-plates", action="store_true",
                    help="Skip the plate-labeling step")
    ap.add_argument("--skip-download", action="store_true",
                    help="With --csv: reuse an existing <name>_images folder")
    args = ap.parse_args()

    name = args.name
    images_dir = BASE / f"{name}_images"
    dataset_dir = BASE / f"{name}_dataset"

    print(f"\n=== add_dataset: batch '{name}' ({args.orientation}) ===")

    # ---- 1 & 2: get images ----
    if args.images_dir:
        images_dir = args.images_dir
        print(f"[1-2] Using existing images: {images_dir}")
    else:
        url_txt = BASE / f"{name}_urls.txt"
        print(f"[1] Extracting URLs from {args.csv}")
        n = extract_urls(args.csv, url_txt)
        print(f"    {n} unique URLs -> {url_txt}")
        if args.skip_download and images_dir.exists():
            print(f"[2] Skipping download; using {images_dir}")
        else:
            print(f"[2] Downloading into {images_dir}")
            download_images(url_txt, images_dir, args.workers)

    # ---- 3: label ----
    print(f"[3] Labeling ({args.orientation}) -> {dataset_dir}")
    label_dataset(args.orientation, images_dir, dataset_dir, args.conf)

    # ---- 4: merge ----
    print(f"[4] Merging {dataset_dir} into {MERGED_DIR}")
    merge_into_master(dataset_dir)

    # ---- 5: plates ----
    if args.no_plates:
        print("[5] Skipping plate labeling (--no-plates)")
    else:
        print("[5] Adding plate boxes to merged set")
        add_plate_labels(args.workers)

    print("\n==================== DONE ====================")
    print("Next, run:")
    print("  python split_merged.py")
    print("  # then train.py transferring from your latest best.pt")
    print("=============================================")


if __name__ == "__main__":
    main()
