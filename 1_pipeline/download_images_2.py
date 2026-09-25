"""download_images.py

Download all ANPR image URLs into a local ``night_images`` folder.

Reads a newline-delimited list of image URLs (``image_urls.txt``) and downloads
them in parallel. Designed for tens of thousands of files:

* Resumable  - a file that already exists (non-empty) is skipped, so re-running
  after an interruption only fetches what's missing.
* Parallel   - a thread pool with a shared, connection-pooled requests session.
* Robust     - per-URL retries with backoff; failures are logged, never fatal.
* Progress   - a tqdm bar plus a summary at the end.

Run:

    python download_images.py

Adjust the constants at the top for paths / concurrency.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Newline-delimited URL list produced from the transactions CSV.
URL_LIST: Path = Path("image_urls_2.txt")

# Destination folder for the downloaded images.
OUTPUT_DIR: Path = Path("/mnt/datadisk/avani/front_back/main/night_images_2")

# CSV of URLs that failed all retries (for a later re-run).
FAILED_LOG: Path = OUTPUT_DIR / "failed_downloads.csv"

NUM_THREADS: int = 32
TIMEOUT_SECONDS: int = 30
MAX_RETRIES: int = 3
RETRY_BACKOFF_SECONDS: float = 1.0  # multiplied by attempt number
CHUNK_SIZE: int = 1 << 16           # 64 KB streaming chunks


def build_session() -> requests.Session:
    """Create a pooled requests session sized for the worker count."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=NUM_THREADS,
        pool_maxsize=NUM_THREADS * 2,
        max_retries=0,  # retries handled explicitly below
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "night-images-downloader/1.0"})
    return session


def load_urls(path: Path) -> List[str]:
    """Read, strip, and de-duplicate URLs from the list file (order-preserving)."""
    if not path.exists():
        raise FileNotFoundError(f"URL list not found: {path}")
    seen = set()
    urls: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            url = line.strip()
            if url and url.startswith("http") and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def filename_for_url(url: str) -> str:
    """Derive a safe local filename from a URL's last path segment."""
    name = os.path.basename(urlparse(url).path)
    return name if name else url.rsplit("/", 1)[-1]


def download_one(
    session: requests.Session, url: str, dest_dir: Path
) -> Tuple[str, str, Optional[str]]:
    """Download a single URL to ``dest_dir``.

    Returns ``(url, status, error)`` where status is one of
    ``"skipped"``, ``"downloaded"``, or ``"failed"``.
    """
    filename = filename_for_url(url)
    target = dest_dir / filename

    # Resumable: already present and non-empty -> skip.
    if target.exists() and target.stat().st_size > 0:
        return url, "skipped", None

    tmp = target.with_suffix(target.suffix + ".part")
    last_error: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, timeout=TIMEOUT_SECONDS, stream=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            out.write(chunk)
            if tmp.stat().st_size == 0:
                raise IOError("empty response body")
            os.replace(tmp, target)  # atomic move into place
            return url, "downloaded", None
        except (requests.RequestException, OSError) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return url, "failed", last_error


def main() -> None:
    """Download every URL in the list into OUTPUT_DIR, with resume + logging."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = load_urls(URL_LIST)
    print(f"Loaded {len(urls)} unique URLs")
    print(f"Downloading into {OUTPUT_DIR} with {NUM_THREADS} threads")

    session = build_session()
    downloaded = skipped = failed = 0
    failed_rows: List[Tuple[str, str]] = []
    lock = threading.Lock()

    progress = tqdm(total=len(urls), desc="Downloading", unit="img")
    try:
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {
                executor.submit(download_one, session, url, OUTPUT_DIR): url
                for url in urls
            }
            for future in as_completed(futures):
                url = futures[future]
                try:
                    _, status, error = future.result()
                except Exception as exc:  # defensive: never crash the run
                    status, error = "failed", f"worker_exception: {exc}"

                with lock:
                    if status == "downloaded":
                        downloaded += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        failed += 1
                        failed_rows.append((url, error or "unknown"))
                progress.update(1)
    finally:
        progress.close()
        session.close()

    if failed_rows:
        with FAILED_LOG.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["url", "error"])
            writer.writerows(failed_rows)

    summary = (
        "\n==================== DOWNLOAD SUMMARY ====================\n"
        f"Total URLs   : {len(urls)}\n"
        f"Downloaded   : {downloaded}\n"
        f"Skipped      : {skipped} (already present)\n"
        f"Failed       : {failed}\n"
        f"Output dir   : {OUTPUT_DIR}\n"
        + (f"Failed log   : {FAILED_LOG}\n" if failed_rows else "")
        + "========================================================="
    )
    print(summary)


if __name__ == "__main__":
    main()
