#!/usr/bin/env python3
"""infer_draw.py

Run the trained YOLOv5 model on a random sample of images and draw the
predicted boxes + class labels on top, for manual inspection. No ground-truth
labels are needed or used.

Output: annotated copies of the sampled images in <out> (default TEST_500).

Run:
    python infer_draw.py \
        --weights /path/to/best.pt \
        --images  /mnt/datadisk/avani/front_back/main/night_images_2 \
        --yolov5-dir /mnt/datadisk/avani/yolov5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

CLASS_NAMES: Dict[int, str] = {
    0: "bike_front",
    1: "bike_back",
    2: "non_bike_front",
    3: "non_bike_back",
}
CLASS_COLORS: Dict[int, tuple] = {
    0: (0, 200, 0),      # green
    1: (0, 140, 255),    # orange
    2: (255, 180, 0),    # cyan-ish
    3: (0, 0, 220),      # red
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_YOLOV5_DIR = Path("/mnt/datadisk/avani/yolov5")


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def draw_box(img, x1, y1, x2, y2, label: str, color) -> None:
    """Draw one box with its label on a bar above the top-left corner."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(label, font, scale, thick)
    ty = max(y1, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 2, ty + base - 2), color, -1)
    cv2.putText(img, label, (x1 + 1, ty - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw model predictions on a sample")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--yolov5-dir", type=Path, default=DEFAULT_YOLOV5_DIR)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output folder (default: <images>/../TEST_500)")
    ap.add_argument("--n", type=int, default=500, help="Number of images to sample")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--batch", type=int, default=64,
                    help="Images per GPU forward pass (local model, not an API)")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = args.out or (args.images.parent / "TEST_500")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.images)
    if not images:
        raise SystemExit(f"No images found in {args.images}")
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

    # Prefer the model's own class names (works for any model, not just the
    # 5-class vehicle one). model.names may be a dict or a list.
    raw_names = getattr(model, "names", None)
    if isinstance(raw_names, dict):
        model_names = {int(k): str(v) for k, v in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        model_names = {i: str(v) for i, v in enumerate(raw_names)}
    else:
        model_names = dict(CLASS_NAMES)
    print("Model classes:", model_names)

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    n_boxes = 0
    progress = tqdm(total=len(sample), desc="Predicting", unit="img")
    for batch in batched(sample, args.batch):
        loaded = []  # (path, bgr_image, rgb_image)
        for p in batch:
            im = cv2.imread(str(p))
            if im is None:
                progress.update(1)
                continue
            loaded.append((p, im, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if not loaded:
            continue
        # One GPU forward pass for the whole batch.
        results = model([lo[2] for lo in loaded], size=args.img)
        for i, (p, image, _rgb) in enumerate(loaded):
            det = results.xyxy[i].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
            for x1, y1, x2, y2, conf, cls in det:
                c = int(cls)
                name = model_names.get(c, f"class_{c}")
                color = CLASS_COLORS.get(c, (0, 255, 255))
                draw_box(image, x1, y1, x2, y2, f"{name} {conf:.2f}", color)
                n_boxes += 1
            cv2.imwrite(str(out_dir / p.name), image)
            progress.update(1)
    progress.close()

    print(f"\nDone. Annotated {len(sample)} images ({n_boxes} boxes drawn).")
    print(f"Output -> {out_dir}")


if __name__ == "__main__":
    main()
