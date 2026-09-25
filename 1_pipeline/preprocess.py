"""preprocess.py

End-to-end cleaning pipeline for the front/back YOLOv5 dataset.

Stage 1 — Duplicate removal
    Exact duplicates via SHA256, near duplicates via pHash indexed in a
    BK-tree (Hamming metric). Both are moved to ``removed_duplicates/``.

Stage 2 — Plate validation
    Remaining images are processed in batches of 512 across 20 worker threads.
    Every vehicle box is cropped and sent to the ANPR API. If ANY vehicle in an
    image fails validation (or has no readable plate), the whole image + label
    is moved to ``removed_no_plate/``.

Nothing is ever deleted; files are only moved. Images are decoded one at a time
so memory stays flat across the full ~400k dataset. Run:

    python preprocess.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

import config
import utils


# ===========================================================================
# Stage 1: duplicate removal
# ===========================================================================
@dataclass
class DuplicateStats:
    """Counters produced by the duplicate-removal stage."""

    exact: int = 0
    near: int = 0
    unreadable: int = 0

    @property
    def total_removed(self) -> int:
        return self.exact + self.near


def remove_duplicates(
    logger: "utils.logging.Logger",
    removal_logger: utils.RemovalLogger,
) -> Tuple[DuplicateStats, List[Path]]:
    """Remove exact and near-duplicate images, returning the survivors.

    Single-threaded by design: hashing is CPU-cheap relative to disk I/O and
    keeping it sequential guarantees deterministic "first occurrence wins"
    behavior and avoids any shared-structure locking on the BK-tree.

    Returns the stats and the list of kept image paths (for stage 2).
    """
    stats = DuplicateStats()
    kept: List[Path] = []

    # Rebuild dedup state from any prior checkpoint so an interrupted run
    # resumes instead of re-hashing everything from scratch.
    records = utils.DedupCheckpoint.load(config.DEDUP_CHECKPOINT)
    seen_sha: Set[str] = {sha for sha, _ in records.values()}
    tree = utils.BKTree()
    for _sha, phash_value in records.values():
        tree.add(phash_value)
    processed: Set[str] = set(records.keys())
    if processed:
        logger.info("Resuming Stage 1 from checkpoint: %d images already kept",
                    len(processed))

    total = utils.count_images(config.IMAGE_DIR)
    logger.info("Stage 1: scanning %d images for duplicates", total)

    checkpoint = utils.DedupCheckpoint(config.DEDUP_CHECKPOINT)
    try:
        for image_path in tqdm(
            utils.iter_image_paths(config.IMAGE_DIR),
            total=total,
            desc="Deduplicating",
            unit="img",
        ):
            # Already handled in a previous run -> skip (fast).
            if image_path.name in processed:
                continue

            sha = utils.sha256_of_file(image_path)
            if sha is None:
                stats.unreadable += 1
                removal_logger.log(image_path.name, "unreadable_file")
                utils.move_pair(
                    image_path,
                    config.REMOVED_DUPLICATES_IMAGES,
                    config.REMOVED_DUPLICATES_LABELS,
                )
                continue

            if sha in seen_sha:
                stats.exact += 1
                removal_logger.log(image_path.name, "exact_duplicate")
                utils.move_pair(
                    image_path,
                    config.REMOVED_DUPLICATES_IMAGES,
                    config.REMOVED_DUPLICATES_LABELS,
                )
                continue

            phash = utils.phash_of_file(image_path)
            if phash is None:
                # File hashed but cannot be decoded as an image -> corrupt.
                stats.unreadable += 1
                removal_logger.log(image_path.name, "corrupt_image")
                utils.move_pair(
                    image_path,
                    config.REMOVED_DUPLICATES_IMAGES,
                    config.REMOVED_DUPLICATES_LABELS,
                )
                continue

            if tree.find_within(phash, config.PHASH_THRESHOLD) is not None:
                stats.near += 1
                removal_logger.log(image_path.name, "near_duplicate")
                utils.move_pair(
                    image_path,
                    config.REMOVED_DUPLICATES_IMAGES,
                    config.REMOVED_DUPLICATES_LABELS,
                )
                continue

            # Unique: record, checkpoint, and keep.
            seen_sha.add(sha)
            tree.add(phash)
            processed.add(image_path.name)
            checkpoint.add(image_path.name, sha, phash)
            kept.append(image_path)
    finally:
        checkpoint.close()

    # Fold in images kept during earlier runs (skipped above) that still exist
    # on disk, so Stage 2 sees the full survivor set.
    kept_names = {path.name for path in kept}
    for name in processed:
        if name in kept_names:
            continue
        prior_path = config.IMAGE_DIR / name
        if prior_path.exists():
            kept.append(prior_path)

    logger.info(
        "Stage 1 complete: %d exact, %d near, %d unreadable removed; %d kept",
        stats.exact, stats.near, stats.unreadable, len(kept),
    )
    return stats, kept


# ===========================================================================
# Stage 2: plate validation
# ===========================================================================
@dataclass
class ValidationOutcome:
    """Result of validating a single image."""

    image_path: Path
    accepted: bool
    reason: str = "ok"
    vehicle_id: Optional[int] = None
    plate_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None


def _validate_image(
    image_path: Path,
    session: "utils.requests.Session",
    logger: "utils.logging.Logger",
) -> ValidationOutcome:
    """Validate one image: every vehicle must contain a readable plate.

    Returns a :class:`ValidationOutcome`. Never raises; all error conditions
    map to a rejection reason so a single bad file cannot crash a worker.
    """
    label_path = utils.label_path_for_image(image_path)
    if not label_path.exists():
        return ValidationOutcome(image_path, False, "missing_label")

    boxes = utils.parse_yolo_label(label_path)
    if not boxes:
        return ValidationOutcome(image_path, False, "empty_label")

    image = utils.decode_image(image_path)
    if image is None:
        return ValidationOutcome(image_path, False, "corrupt_image")

    crops = utils.crop_vehicles(image, boxes)
    del image  # release the full frame ASAP; keep only the small crops

    # Encode all valid crops; a None crop is an immediate rejection.
    encoded: List[bytes] = []
    encoded_vehicle_ids: List[int] = []
    for vehicle_id, crop in enumerate(crops):
        if crop is None:
            return ValidationOutcome(image_path, False, "invalid_crop", vehicle_id)
        blob = utils.encode_crop_to_jpeg(crop)
        if blob is None:
            return ValidationOutcome(image_path, False, "invalid_crop", vehicle_id)
        encoded.append(blob)
        encoded_vehicle_ids.append(vehicle_id)

    # One ANPR call covering this image's vehicles. call_anpr_api is the single
    # function to swap for true batch inference.
    per_vehicle_results = utils.call_anpr_api(session, encoded, logger)

    for vehicle_id, results in zip(encoded_vehicle_ids, per_vehicle_results):
        plate = utils.best_plate(results)
        if plate is None:
            return ValidationOutcome(
                image_path, False, "no_plate_detected", vehicle_id, 0.0, 0.0
            )
        if not utils.is_plate_valid(plate):
            return ValidationOutcome(
                image_path,
                False,
                "plate_below_threshold",
                vehicle_id,
                plate.plate_confidence,
                plate.ocr_confidence,
            )

    return ValidationOutcome(image_path, True)


def _batched(paths: Sequence[Path], size: int) -> Iterator[List[Path]]:
    """Yield successive ``size``-length slices of ``paths``."""
    for start in range(0, len(paths), size):
        yield list(paths[start : start + size])


def validate_dataset(
    kept: Sequence[Path],
    logger: "utils.logging.Logger",
    removal_logger: utils.RemovalLogger,
) -> int:
    """Validate all kept images in batches; move failures aside.

    Uses one shared, connection-pooled session across a ``ThreadPoolExecutor``
    of ``NUM_THREADS`` workers, processing ``BATCH_SIZE`` images per batch. File
    moves happen on the main thread as results arrive, so only the CSV logger
    needs locking (it has its own).

    Returns the number of images rejected for plate failures.
    """
    session = utils.build_session()
    rejected = 0

    # Skip images that already passed validation in a previous run so the ANPR
    # API is never re-hit for accepted images.
    already_validated = utils.NameCheckpoint.load(config.VALIDATED_CHECKPOINT)
    pending = [path for path in kept if path.name not in already_validated]
    skipped = len(kept) - len(pending)
    if skipped:
        logger.info("Resuming Stage 2 from checkpoint: %d images already validated",
                    skipped)

    total = len(pending)
    logger.info("Stage 2: validating %d images (batch=%d, threads=%d)",
                total, config.BATCH_SIZE, config.NUM_THREADS)

    validated_checkpoint = utils.NameCheckpoint(config.VALIDATED_CHECKPOINT)
    progress = tqdm(total=total, desc="Validating", unit="img")
    try:
        with ThreadPoolExecutor(max_workers=config.NUM_THREADS) as executor:
            for batch in _batched(pending, config.BATCH_SIZE):
                future_map = {
                    executor.submit(_validate_image, path, session, logger): path
                    for path in batch
                }
                for future in as_completed(future_map):
                    path = future_map[future]
                    try:
                        outcome = future.result()
                    except Exception as exc:  # defensive: never crash pipeline
                        logger.error("Unexpected error on %s: %s", path.name, exc)
                        outcome = ValidationOutcome(path, False, "worker_exception")

                    if outcome.accepted:
                        validated_checkpoint.add(outcome.image_path.name)
                    else:
                        rejected += 1
                        removal_logger.log(
                            outcome.image_path.name,
                            outcome.reason,
                            outcome.vehicle_id,
                            outcome.plate_confidence,
                            outcome.ocr_confidence,
                        )
                        utils.move_pair(
                            outcome.image_path,
                            config.REMOVED_NO_PLATE_IMAGES,
                            config.REMOVED_NO_PLATE_LABELS,
                        )
                    progress.update(1)
    finally:
        progress.close()
        session.close()
        validated_checkpoint.close()

    logger.info("Stage 2 complete: %d images rejected for plate failures", rejected)
    return rejected


# ===========================================================================
# Orchestration
# ===========================================================================
def main() -> None:
    """Run the full two-stage cleaning pipeline and print a summary."""
    config.ensure_directories()
    logger = utils.setup_logger()

    start = time.perf_counter()
    logger.info("Pipeline started. Dataset root: %s", config.ROOT)

    with utils.RemovalLogger(config.REMOVED_CSV) as removal_logger:
        dup_stats, kept = remove_duplicates(logger, removal_logger)
        plate_failures = validate_dataset(kept, logger, removal_logger)

    remaining = len(kept) - plate_failures
    elapsed = time.perf_counter() - start

    summary = (
        "\n==================== PIPELINE SUMMARY ====================\n"
        f"Duplicates removed      : {dup_stats.total_removed} "
        f"(exact={dup_stats.exact}, near={dup_stats.near})\n"
        f"Unreadable/corrupt moved: {dup_stats.unreadable}\n"
        f"Plate-validation failures: {plate_failures}\n"
        f"Remaining images         : {remaining}\n"
        f"Processing time          : {elapsed:.1f}s "
        f"({elapsed / 3600.0:.2f}h)\n"
        f"Removal report           : {config.REMOVED_CSV}\n"
        "=========================================================="
    )
    logger.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
