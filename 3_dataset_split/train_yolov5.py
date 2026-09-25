"""train_yolov5.py

Launch YOLOv5-nano training on the 70/20/10 split produced by
``split_dataset.py``.

This is a thin, transparent wrapper around the Ultralytics YOLOv5 repo's
``train.py``. It locates the yolov5 checkout, verifies the generated
``data.yaml`` exists, and runs training with settings tuned for a single
NVIDIA A100 at 320x320 input. Every knob is a constant at the top of the file.

Run:

    python train_yolov5.py

Override the yolov5 location without editing this file:

    YOLOV5_DIR=/path/to/yolov5 python train_yolov5.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import split_dataset  # for OUTPUT_DIR / data.yaml location (imports config too)


# ---------------------------------------------------------------------------
# Training hyperparameters (YOLOv5-nano @ 320, single A100)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS: str = "yolov5n.pt"   # nano; auto-downloaded if absent
IMG_SIZE: int = 320
BATCH_SIZE: int = 256               # A100 handles this comfortably at 320/nano
EPOCHS: int = 100
DEVICE: str = "0"                   # GPU index; "0,1" for multi-GPU
WORKERS: int = 16                   # dataloader workers
RUN_NAME: str = "front_back_yolov5n_320"
PROJECT_DIR: Path = split_dataset.OUTPUT_DIR / "runs"
# Cache mode: "" (none), "ram", or "disk". RAM caching 400k images will OOM;
# leave empty, or use "disk" for a speed-up if you have spare disk.
CACHE_MODE: str = ""
PATIENCE: int = 30                  # early-stopping patience (epochs)

DATA_YAML: Path = split_dataset.OUTPUT_DIR / "data.yaml"

# Candidate locations for the yolov5 repo, checked in order.
_YOLOV5_CANDIDATES: List[Path] = [
    Path("/mnt/datadisk/avani/yolov5"),
    Path("/mnt/datadisk/avani/front_back/yolov5"),
    Path("/mnt/datadisk/avani/front_back/main/yolov5"),
    Path.cwd() / "yolov5",
]


def find_yolov5_dir() -> Optional[Path]:
    """Locate the yolov5 repo (dir containing train.py).

    Honors the ``YOLOV5_DIR`` environment variable first, then a list of
    common locations on this server.
    """
    env = os.environ.get("YOLOV5_DIR")
    candidates: List[Path] = []
    if env:
        candidates.append(Path(env))
    candidates.extend(_YOLOV5_CANDIDATES)

    for candidate in candidates:
        if (candidate / "train.py").is_file():
            return candidate
    return None


def build_command(yolov5_dir: Path) -> List[str]:
    """Assemble the ``python train.py ...`` command line as an argv list."""
    cmd: List[str] = [
        sys.executable,
        str(yolov5_dir / "train.py"),
        "--img", str(IMG_SIZE),
        "--batch", str(BATCH_SIZE),
        "--epochs", str(EPOCHS),
        "--data", str(DATA_YAML),
        "--weights", MODEL_WEIGHTS,
        "--device", DEVICE,
        "--workers", str(WORKERS),
        "--project", str(PROJECT_DIR),
        "--name", RUN_NAME,
        "--patience", str(PATIENCE),
        "--exist-ok",
    ]
    if CACHE_MODE:
        cmd.extend(["--cache", CACHE_MODE])
    return cmd


def main() -> int:
    """Validate prerequisites and run YOLOv5 training. Returns exit code."""
    if not DATA_YAML.is_file():
        print(f"ERROR: {DATA_YAML} not found. Run split_dataset.py first.",
              file=sys.stderr)
        return 2

    yolov5_dir = find_yolov5_dir()
    if yolov5_dir is None:
        print(
            "ERROR: could not locate the yolov5 repo (no train.py found).\n"
            "Set YOLOV5_DIR=/path/to/yolov5 or clone it:\n"
            "    git clone https://github.com/ultralytics/yolov5\n"
            "    pip install -r yolov5/requirements.txt",
            file=sys.stderr,
        )
        return 3

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = build_command(yolov5_dir)

    print("Launching YOLOv5 training:")
    print("  yolov5 dir :", yolov5_dir)
    print("  data.yaml  :", DATA_YAML)
    print("  command    :", " ".join(cmd))
    print()

    # Run from inside the yolov5 dir so its relative imports/paths resolve.
    completed = subprocess.run(cmd, cwd=str(yolov5_dir))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
