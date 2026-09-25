#!/usr/bin/env python3
"""recover_plates_local.py

Second-chance plate detection for the 'no_plate' images using a LOCAL YOLOv5
plate model (class 0 = reg_plate), instead of the ANPR HTTP service.

For every image in <images>:
  * run the plate model (batched on GPU),
  * if it finds a plate (conf >= --conf):
      - append '4 cx cy w h' to the label (model class 0 -> dataset class 4),
      - MOVE image + updated label into  <base>/recovered/{images,labels},
      - save an overlay (box + confidence) into <base>/recovered/overlays,
  * if it finds no plate:
      - MOVE image + label into <base>/rejected/{images,labels}
        (final: plate not visible).

<base> defaults to the parent of --images (e.g. Dataset_merged/no_plate).

Run:
    python recover_plates_local.py \
        --weights /mnt/datadisk/avani/front_back/main/plate_model_320.pt \
        --images  Dataset_merged/no_plate/images \
        --labels  Dataset_merged/no_plate/labels \
        --yolov5-dir /mnt/datadisk/avani/yolov5
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

PLATE_CLASS_ID = 4               # reg_plate in the merged dataset
MODEL_PLATE_CLASS = 0            # plate model outputs class 0 = reg_plate
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VEHICLE_COLOR = (0, 200, 0)      # green
PLATE_COLOR = (0, 0, 220)        # red


def iter_images(images_dir: Path) -> List[Path]:
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def read_vehicle_lines(label_path: Path) -> List[str]:
    """Return existing non-plate label lines (classes 0-3), plate lines dropped."""
    lines: List[str] = []
    if not label_path.exists():
        return lines
    try:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if cid == PLATE_CLASS_ID:
                continue
            lines.append(line.strip())
    except OSError:
        pass
    return lines


def draw(img, x1, y1, x2, y2, label, color) -> None:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    f, s, t = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), b = cv2.getTextSize(label, f, s, t)
    ty = max(y1, th + 4)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 2, ty + b - 2), color, -1)
    cv2.putText(img, label, (x1 + 1, ty - 2), f, s, (0, 0, 0), t, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover plates with a local model")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--yolov5-dir", type=Path,
                    default=Path("/mnt/datadisk/avani/yolov5"))
    ap.add_argument("--base", type=Path, default=None,
                    help="Output base (default: parent of --images)")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base = args.base or args.images.parent
    rec_images = base / "recovered" / "images"
    rec_labels = base / "recovered" / "labels"
    rec_overlays = base / "recovered" / "overlays"
    rej_images = base / "rejected" / "images"
    rej_labels = base / "rejected" / "labels"
    for d in (rec_images, rec_labels, rec_overlays, rej_images, rej_labels):
        d.mkdir(parents=True, exist_ok=True)

    images = iter_images(args.images)
    if args.limit and args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images in {args.images}")

    if not (args.yolov5_dir / "hubconf.py").is_file():
        raise SystemExit(f"yolov5 repo not found at {args.yolov5_dir}")
    device = ("cuda:" + args.device) if args.device not in ("cpu", "") else "cpu"
    print(f"Loading plate model via {args.yolov5_dir}")
    model = torch.hub.load(str(args.yolov5_dir), "custom",
                           path=str(args.weights), source="local", verbose=False)
    model.to(device)
    model.conf = args.conf
    model.iou = args.iou
    print("Model classes:", getattr(model, "names", None))

    def batched(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    recovered = rejected = read_fail = 0
    progress = tqdm(total=len(images), desc="Recovering", unit="img")

    for batch in batched(images, args.batch):
        loaded: List[Tuple[Path, np.ndarray, np.ndarray]] = []
        for p in batch:
            im = cv2.imread(str(p))
            if im is None:
                read_fail += 1
                progress.update(1)
                continue
            loaded.append((p, im, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if not loaded:
            continue

        results = model([lo[2] for lo in loaded], size=args.img)

        for i, (img_path, image, _rgb) in enumerate(loaded):
            det = results.xyxy[i].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]
            h, w = image.shape[:2]
            label_path = args.labels / (img_path.stem + ".txt")
            vehicle_lines = read_vehicle_lines(label_path)

            plate_lines: List[str] = []
            plate_boxes: List[Tuple[float, float, float, float, float]] = []
            for x1, y1, x2, y2, conf, cls in det:
                if int(cls) != MODEL_PLATE_CLASS:
                    continue
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                if bw <= 0 or bh <= 0:
                    continue
                cx = min(1.0, max(0.0, cx)); cy = min(1.0, max(0.0, cy))
                bw = min(1.0, max(0.0, bw)); bh = min(1.0, max(0.0, bh))
                plate_lines.append(
                    f"{PLATE_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                plate_boxes.append((x1, y1, x2, y2, float(conf)))

            if plate_lines:
                # recovered: write label (vehicles + plate), overlay, move image
                new_label = vehicle_lines + plate_lines
                (rec_labels / label_path.name).write_text(
                    "\n".join(new_label) + "\n", encoding="utf-8")
                overlay = image.copy()
                for vl in vehicle_lines:
                    pp = vl.split()
                    try:
                        cx, cy, bw, bh = map(float, pp[1:5])
                    except ValueError:
                        continue
                    draw(overlay, (cx-bw/2)*w, (cy-bh/2)*h,
                         (cx+bw/2)*w, (cy+bh/2)*h, pp[0], VEHICLE_COLOR)
                for (x1, y1, x2, y2, conf) in plate_boxes:
                    draw(overlay, x1, y1, x2, y2, f"reg_plate {conf:.2f}", PLATE_COLOR)
                cv2.imwrite(str(rec_overlays / img_path.name), overlay)
                shutil.move(str(img_path), str(rec_images / img_path.name))
                if label_path.exists():
                    try:
                        label_path.unlink()
                    except OSError:
                        pass
                recovered += 1
            else:
                # rejected: no plate found
                shutil.move(str(img_path), str(rej_images / img_path.name))
                if label_path.exists():
                    shutil.move(str(label_path), str(rej_labels / label_path.name))
                rejected += 1
            progress.update(1)

    progress.close()
    print("\n==================== RECOVERY SUMMARY ====================")
    print(f"Recovered (plate found) : {recovered}")
    print(f"Rejected  (no plate)    : {rejected}")
    print(f"Read failures           : {read_fail}")
    print(f"Recovered  -> {rec_images.parent}  (images/, labels/, overlays/)")
    print(f"Rejected   -> {rej_images.parent}  (images/, labels/)")
    print("=========================================================")


if __name__ == "__main__":
    main()
