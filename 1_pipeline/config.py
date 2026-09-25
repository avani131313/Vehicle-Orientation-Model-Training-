"""config.py

Central configuration for the front/back YOLOv5 dataset cleaning pipeline.

All paths, thresholds, threading and API settings live here. Importing this
module has the side effect of creating every output directory that the
pipeline needs, so that the rest of the code can assume they exist.

The module is intentionally free of heavy logic: it only defines constants
and guarantees the on-disk directory layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, FrozenSet, Tuple


# ---------------------------------------------------------------------------
# Root / input directories
# ---------------------------------------------------------------------------
# ROOT points at the dataset root ("front_back"). Override by editing this
# single line (or set it from an environment variable if you prefer).
ROOT: Final[Path] = Path("/mnt/datadisk/avani/front_back/main/Dataset").resolve()

IMAGE_DIR: Final[Path] = ROOT / "images"
LABEL_DIR: Final[Path] = ROOT / "labels"
DEBUG_DIR: Final[Path] = ROOT / "debug"

# ---------------------------------------------------------------------------
# Output directories (created automatically)
# ---------------------------------------------------------------------------
REMOVED_DUPLICATES_DIR: Final[Path] = ROOT / "removed_duplicates"
REMOVED_DUPLICATES_IMAGES: Final[Path] = REMOVED_DUPLICATES_DIR / "images"
REMOVED_DUPLICATES_LABELS: Final[Path] = REMOVED_DUPLICATES_DIR / "labels"

REMOVED_NO_PLATE_DIR: Final[Path] = ROOT / "removed_no_plate"
REMOVED_NO_PLATE_IMAGES: Final[Path] = REMOVED_NO_PLATE_DIR / "images"
REMOVED_NO_PLATE_LABELS: Final[Path] = REMOVED_NO_PLATE_DIR / "labels"

LOG_DIR: Final[Path] = ROOT / "logs"

# CSV report of everything the pipeline removed.
REMOVED_CSV: Final[Path] = LOG_DIR / "removed_images.csv"
# Plain-text run log.
RUN_LOG: Final[Path] = LOG_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Resume checkpoints
# ---------------------------------------------------------------------------
# Stage 1 writes one JSONL line per KEPT image (name, sha256, phash). On
# restart the pipeline reloads these to rebuild the dedup index and skip work
# already done. Stage 2 writes one line per image that PASSED validation, so a
# restart never re-calls the ANPR API for images already accepted.
DEDUP_CHECKPOINT: Final[Path] = LOG_DIR / "dedup_checkpoint.jsonl"
VALIDATED_CHECKPOINT: Final[Path] = LOG_DIR / "validated_checkpoint.txt"

# ---------------------------------------------------------------------------
# ANPR API
# ---------------------------------------------------------------------------
API_URL: Final[str] = "http://216.48.185.18:5505/upload"
API_TIMEOUT_SECONDS: Final[int] = 30
API_MAX_RETRIES: Final[int] = 3
API_RETRY_BACKOFF_SECONDS: Final[float] = 1.0  # multiplied by attempt number

# Multipart field name used when uploading a single image.
API_FILE_FIELD: Final[str] = "file"

# ---------------------------------------------------------------------------
# ANPR response keys.
# Kept as constants so a future API schema change only touches this block.
# The pipeline reads plate results out of a list under RESP_RESULTS_KEY; each
# element is expected to expose the three keys below.
# ---------------------------------------------------------------------------
RESP_RESULTS_KEY: Final[str] = "plates"
KEY_PLATE_CONFIDENCE: Final[str] = "confidence"
KEY_OCR_CONFIDENCE: Final[str] = "ocr_confidence"
KEY_PLATE_TEXT: Final[str] = "plate"

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
PLATE_DETECTION_THRESHOLD: Final[float] = 0.65
OCR_THRESHOLD: Final[float] = 0.80
MIN_PLATE_TEXT_LENGTH: Final[int] = 5

# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
PHASH_THRESHOLD: Final[int] = 6          # max Hamming distance for a near dup
PHASH_SIZE: Final[int] = 8               # imagehash phash hash_size (8 -> 64 bit)

# ---------------------------------------------------------------------------
# Throughput / batching / threading
# ---------------------------------------------------------------------------
BATCH_SIZE: Final[int] = 512
NUM_THREADS: Final[int] = 20

# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
# Vehicle crops smaller than this in either dimension are considered invalid.
MIN_CROP_SIZE: Final[int] = 8

# ---------------------------------------------------------------------------
# Valid image extensions (lower case, with leading dot)
# ---------------------------------------------------------------------------
VALID_EXTENSIONS: Final[FrozenSet[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

# Directories that must exist before the pipeline runs.
_DIRECTORIES_TO_CREATE: Final[Tuple[Path, ...]] = (
    IMAGE_DIR,
    LABEL_DIR,
    DEBUG_DIR,
    REMOVED_DUPLICATES_DIR,
    REMOVED_DUPLICATES_IMAGES,
    REMOVED_DUPLICATES_LABELS,
    REMOVED_NO_PLATE_DIR,
    REMOVED_NO_PLATE_IMAGES,
    REMOVED_NO_PLATE_LABELS,
    LOG_DIR,
)


def ensure_directories() -> None:
    """Create every output directory required by the pipeline.

    Idempotent: safe to call multiple times. Input directories (images /
    labels) are created too so the pipeline never crashes on a fresh checkout,
    but they are expected to already contain the dataset.
    """
    for directory in _DIRECTORIES_TO_CREATE:
        directory.mkdir(parents=True, exist_ok=True)


# Guarantee the directory layout at import time.
ensure_directories()
