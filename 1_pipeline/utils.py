"""utils.py

Reusable building blocks for the dataset cleaning pipeline:

* logging setup
* a BK-tree for scalable Hamming-distance (perceptual-hash) lookups
* exact (SHA256) and perceptual (pHash) hashing helpers
* YOLO label parsing and vehicle cropping
* a thin, swappable ANPR API client (single function to change for batching)
* per-vehicle plate validation
* a thread-safe CSV logger
* safe file "move" helpers (never delete)

Nothing here loads the whole dataset into memory; images are opened one at a
time and released promptly.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import imagehash
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter

import config


# ===========================================================================
# Logging
# ===========================================================================
def setup_logger(name: str = "preprocess") -> logging.Logger:
    """Create (or return) a configured logger that writes to file and stdout.

    The logger is safe to fetch repeatedly; handlers are only attached once.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)-14s | %(message)s"
    )

    file_handler = logging.FileHandler(config.RUN_LOG, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


# ===========================================================================
# BK-tree for perceptual-hash near-duplicate detection
# ===========================================================================
def hamming_distance(a: int, b: int) -> int:
    """Return the Hamming distance between two integers (bit difference)."""
    return int((a ^ b).bit_count())


class _BKNode:
    """Internal BK-tree node holding one hash value and distance-keyed kids."""

    __slots__ = ("value", "children")

    def __init__(self, value: int) -> None:
        self.value: int = value
        self.children: Dict[int, "_BKNode"] = {}


class BKTree:
    """Burkhard-Keller tree over integer perceptual hashes.

    Provides O(log n)-ish inserts and radius queries under the Hamming
    metric, which keeps near-duplicate detection well away from the naive
    O(n^2) all-pairs comparison. Suitable for hundreds of thousands of
    64-bit hashes.
    """

    def __init__(self) -> None:
        self._root: Optional[_BKNode] = None
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    def add(self, value: int) -> None:
        """Insert a hash value into the tree."""
        if self._root is None:
            self._root = _BKNode(value)
            self._size = 1
            return

        node = self._root
        while True:
            distance = hamming_distance(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value)
                self._size += 1
                return
            node = child

    def find_within(self, value: int, threshold: int) -> Optional[int]:
        """Return any stored hash within ``threshold`` of ``value``.

        Returns the first match found (we only need existence for dedup), or
        ``None`` if nothing is close enough. The triangle inequality prunes
        the search so only a small fraction of nodes are visited.
        """
        if self._root is None:
            return None

        stack: List[_BKNode] = [self._root]
        while stack:
            node = stack.pop()
            distance = hamming_distance(value, node.value)
            if distance <= threshold:
                return node.value
            low = distance - threshold
            high = distance + threshold
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)
        return None


# ===========================================================================
# Hashing helpers
# ===========================================================================
def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> Optional[str]:
    """Return the hex SHA256 of a file, streaming it in chunks.

    Returns ``None`` if the file cannot be read.
    """
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return None


def phash_of_file(path: Path) -> Optional[int]:
    """Return the perceptual hash of an image as a 64-bit integer.

    Uses Pillow so the image is decoded lazily and released immediately.
    Returns ``None`` for corrupt or unreadable images.
    """
    try:
        with Image.open(path) as img:
            img.load()
            phash = imagehash.phash(img, hash_size=config.PHASH_SIZE)
    except (OSError, ValueError, Image.DecompressionBombError):
        return None

    bits = phash.hash.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


# ===========================================================================
# YOLO label parsing + cropping
# ===========================================================================
@dataclass(frozen=True)
class YoloBox:
    """A single normalized YOLO bounding box (class + center/size in [0, 1])."""

    cls: int
    x_center: float
    y_center: float
    width: float
    height: float


def label_path_for_image(image_path: Path) -> Path:
    """Return the expected label file path for a given image path."""
    return config.LABEL_DIR / (image_path.stem + ".txt")


def parse_yolo_label(label_path: Path) -> List[YoloBox]:
    """Parse a YOLO label file into a list of :class:`YoloBox`.

    Malformed lines are skipped rather than raising. Missing or empty files
    yield an empty list; the caller decides how to treat that.
    """
    boxes: List[YoloBox] = []
    try:
        with label_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                except ValueError:
                    continue
                boxes.append(YoloBox(cls, x_center, y_center, width, height))
    except OSError:
        return []
    return boxes


def yolo_box_to_pixels(
    box: YoloBox, img_width: int, img_height: int
) -> Tuple[int, int, int, int]:
    """Convert a normalized YOLO box to integer pixel (x1, y1, x2, y2).

    The result is clamped to the image bounds.
    """
    bw = box.width * img_width
    bh = box.height * img_height
    cx = box.x_center * img_width
    cy = box.y_center * img_height

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    return x1, y1, x2, y2


def crop_vehicles(
    image: "np.ndarray", boxes: Sequence[YoloBox]
) -> List[Optional["np.ndarray"]]:
    """Crop every vehicle box out of a decoded BGR image.

    Returns one entry per box, in order. An entry is ``None`` when the crop is
    invalid (degenerate or below the minimum size), which the validator treats
    as a failure.
    """
    height, width = image.shape[:2]
    crops: List[Optional["np.ndarray"]] = []
    for box in boxes:
        x1, y1, x2, y2 = yolo_box_to_pixels(box, width, height)
        if (x2 - x1) < config.MIN_CROP_SIZE or (y2 - y1) < config.MIN_CROP_SIZE:
            crops.append(None)
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crops.append(None)
            continue
        crops.append(np.ascontiguousarray(crop))
    return crops


def encode_crop_to_jpeg(crop: "np.ndarray") -> Optional[bytes]:
    """Encode a BGR crop to in-memory JPEG bytes for multipart upload.

    Returns ``None`` if encoding fails. Keeps memory bounded (no temp files).
    """
    try:
        success, buffer = cv2.imencode(".jpg", crop)
    except cv2.error:
        return None
    if not success:
        return None
    return buffer.tobytes()


# ===========================================================================
# ANPR API client
# ===========================================================================
def build_session() -> requests.Session:
    """Create a pooled, retrying :class:`requests.Session`.

    Connection pooling is sized for the worker count so all threads can reuse
    keep-alive connections without contention.
    """
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=config.NUM_THREADS,
        pool_maxsize=config.NUM_THREADS * 2,
        max_retries=0,  # retries handled explicitly for finer logging control
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass(frozen=True)
class PlateResult:
    """Normalized single-plate result extracted from the ANPR response."""

    plate_confidence: float
    ocr_confidence: float
    plate_text: str


def _parse_anpr_payload(payload: Dict) -> List[PlateResult]:
    """Translate a raw ANPR JSON payload into a list of :class:`PlateResult`.

    Isolated so that adapting to a new response schema only touches this
    function and the key constants in ``config``.
    """
    results_raw = payload.get(config.RESP_RESULTS_KEY, [])
    if not isinstance(results_raw, list):
        return []

    parsed: List[PlateResult] = []
    for item in results_raw:
        if not isinstance(item, dict):
            continue
        try:
            plate_conf = float(item.get(config.KEY_PLATE_CONFIDENCE, 0.0) or 0.0)
            ocr_conf = float(item.get(config.KEY_OCR_CONFIDENCE, 0.0) or 0.0)
        except (TypeError, ValueError):
            plate_conf, ocr_conf = 0.0, 0.0
        text = str(item.get(config.KEY_PLATE_TEXT, "") or "")
        parsed.append(PlateResult(plate_conf, ocr_conf, text))
    return parsed


def call_anpr_api(
    session: requests.Session,
    images: Sequence[bytes],
    logger: logging.Logger,
) -> List[List[PlateResult]]:
    """Run ANPR on a group of encoded images.

    This is the ONLY function to modify when the API starts accepting true
    batches. It takes a sequence of JPEG byte blobs and returns, for each input
    image (in order), a list of :class:`PlateResult`. On any per-image failure
    (timeout, HTTP error, bad JSON) that image's entry is an empty list, which
    the validator interprets as "no readable plate".

    Current implementation: one HTTP request per image using multipart upload
    (``files={"file": image}``). To switch to batched inference, replace the
    body with a single request that posts all ``images`` at once and fan the
    parsed results back out into the same ``List[List[PlateResult]]`` shape.
    """
    outputs: List[List[PlateResult]] = []
    for index, blob in enumerate(images):
        outputs.append(_call_anpr_single(session, blob, index, logger))
    return outputs


def _call_anpr_single(
    session: requests.Session,
    blob: bytes,
    index: int,
    logger: logging.Logger,
) -> List[PlateResult]:
    """Post one image to the ANPR endpoint with bounded retries.

    Returns parsed results, or an empty list if every attempt fails.
    """
    last_error: Optional[str] = None
    for attempt in range(1, config.API_MAX_RETRIES + 1):
        try:
            response = session.post(
                config.API_URL,
                files={config.API_FILE_FIELD: (f"crop_{index}.jpg", blob, "image/jpeg")},
                timeout=config.API_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.Timeout:
            last_error = "timeout"
        except requests.RequestException as exc:
            last_error = f"request_error:{exc.__class__.__name__}"
        except ValueError:
            last_error = "invalid_json"
        else:
            if isinstance(payload, dict):
                return _parse_anpr_payload(payload)
            last_error = "unexpected_payload_type"

        if attempt < config.API_MAX_RETRIES:
            _sleep_backoff(attempt)

    logger.warning("ANPR failed for crop %d after %d attempts (%s)",
                   index, config.API_MAX_RETRIES, last_error)
    return []


def _sleep_backoff(attempt: int) -> None:
    """Sleep for a linear backoff based on the attempt number."""
    import time

    time.sleep(config.API_RETRY_BACKOFF_SECONDS * attempt)


# ===========================================================================
# Plate validation
# ===========================================================================
def is_plate_valid(result: PlateResult) -> bool:
    """Return True if a single plate result passes all thresholds."""
    return (
        result.plate_confidence >= config.PLATE_DETECTION_THRESHOLD
        and result.ocr_confidence >= config.OCR_THRESHOLD
        and len(result.plate_text.strip()) >= config.MIN_PLATE_TEXT_LENGTH
    )


def best_plate(results: Sequence[PlateResult]) -> Optional[PlateResult]:
    """Return the highest-OCR-confidence plate from a list, or None if empty."""
    if not results:
        return None
    return max(results, key=lambda r: r.ocr_confidence)


# ===========================================================================
# Thread-safe CSV logging of removals
# ===========================================================================
class RemovalLogger:
    """Thread-safe writer for the ``removed_images.csv`` report.

    A single lock serializes writes across all worker threads, so rows never
    interleave. The header is written once on construction.
    """

    _FIELDS = ("image", "reason", "vehicle_id", "plate_confidence", "ocr_confidence")

    def __init__(self, csv_path: Path) -> None:
        self._lock = threading.Lock()
        self._path = csv_path
        # Append so a resumed run keeps prior removal rows; write the header
        # only when the file is new/empty.
        is_new = (not csv_path.exists()) or csv_path.stat().st_size == 0
        self._handle = csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        if is_new:
            self._writer.writerow(self._FIELDS)
            self._handle.flush()

    def log(
        self,
        image: str,
        reason: str,
        vehicle_id: Optional[int] = None,
        plate_confidence: Optional[float] = None,
        ocr_confidence: Optional[float] = None,
    ) -> None:
        """Append one removal row. Safe to call from many threads."""
        row = [
            image,
            reason,
            "" if vehicle_id is None else vehicle_id,
            "" if plate_confidence is None else f"{plate_confidence:.4f}",
            "" if ocr_confidence is None else f"{ocr_confidence:.4f}",
        ]
        with self._lock:
            self._writer.writerow(row)
            self._handle.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self) -> "RemovalLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ===========================================================================
# Resume checkpoints
# ===========================================================================
class DedupCheckpoint:
    """Append-only JSONL checkpoint of KEPT images for Stage 1 resume.

    Each kept image contributes one line ``{"name", "sha", "phash"}``. On a
    restart :meth:`load` rebuilds the exact-hash set and pHash BK-tree, and the
    set of names to skip, so dedup does not re-hash work already done.
    """

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._handle = path.open("a", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> Dict[str, Tuple[str, int]]:
        """Load prior kept records as ``{name: (sha_hex, phash_int)}``."""
        records: Dict[str, Tuple[str, int]] = {}
        if not path.exists():
            return records
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        records[str(obj["name"])] = (str(obj["sha"]), int(obj["phash"]))
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            pass
        return records

    def add(self, name: str, sha: str, phash: int) -> None:
        """Append one kept-image record and flush to the OS buffer."""
        record = json.dumps({"name": name, "sha": sha, "phash": phash})
        with self._lock:
            self._handle.write(record + "\n")
            self._handle.flush()

    def close(self) -> None:
        """Flush and close the checkpoint file."""
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self) -> "DedupCheckpoint":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class NameCheckpoint:
    """Thread-safe append-only set of image names, for Stage 2 resume.

    One name per line. Records images that PASSED validation so a restart skips
    them and never re-calls the ANPR API for already-accepted images.
    """

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._handle = path.open("a", encoding="utf-8")

    @staticmethod
    def load(path: Path) -> set:
        """Load the set of previously recorded names."""
        names: set = set()
        if not path.exists():
            return names
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        names.add(stripped)
        except OSError:
            pass
        return names

    def add(self, name: str) -> None:
        """Append one name and flush to the OS buffer."""
        with self._lock:
            self._handle.write(name + "\n")
            self._handle.flush()

    def close(self) -> None:
        """Flush and close the checkpoint file."""
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self) -> "NameCheckpoint":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ===========================================================================
# Safe file movement (never delete)
# ===========================================================================
def move_pair(
    image_path: Path,
    dest_image_dir: Path,
    dest_label_dir: Path,
) -> None:
    """Move an image and its sibling label into destination directories.

    Never deletes: uses ``shutil.move``. If a name collision occurs at the
    destination, a numeric suffix is appended so nothing is overwritten. A
    missing label is tolerated (the image is still moved).
    """
    _safe_move(image_path, dest_image_dir)

    label_path = label_path_for_image(image_path)
    if label_path.exists():
        _safe_move(label_path, dest_label_dir)


def _safe_move(source: Path, dest_dir: Path) -> None:
    """Move ``source`` into ``dest_dir`` without overwriting existing files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / source.name
    if target.exists():
        stem, suffix = source.stem, source.suffix
        counter = 1
        while target.exists():
            target = dest_dir / f"{stem}__{counter}{suffix}"
            counter += 1
    try:
        shutil.move(str(source), str(target))
    except (OSError, shutil.Error):
        # As a last resort, copy; still never deletes the original silently.
        try:
            shutil.copy2(str(source), str(target))
        except OSError:
            pass


# ===========================================================================
# Image discovery / decoding
# ===========================================================================
def iter_image_paths(image_dir: Path) -> Iterable[Path]:
    """Yield image paths with valid extensions, one at a time (no big list).

    Streaming keeps memory flat even for 400k+ files.
    """
    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in config.VALID_EXTENSIONS:
            yield path


def count_images(image_dir: Path) -> int:
    """Count valid images without materializing the list of paths."""
    total = 0
    for _ in iter_image_paths(image_dir):
        total += 1
    return total


def decode_image(path: Path) -> Optional["np.ndarray"]:
    """Decode an image to a BGR ndarray, or return None if corrupt.

    Uses ``cv2.imdecode`` on raw bytes so unusual paths and formats are handled
    uniformly, and the buffer is freed as soon as the function returns.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    return image
