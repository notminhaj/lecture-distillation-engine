"""
Media downloader using yt-dlp.

Supports any URL yt-dlp handles (YouTube, Vimeo, direct video links).
Downloads the best available video+audio stream and caches locally.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from distillation.config import Config

logger = logging.getLogger(__name__)


class Downloader:
    """
    Downloads remote media to the local cache directory.

    Why yt-dlp instead of direct HTTP?
      - Handles cookie-gated content, live streams, age gates.
      - Automatically selects the best quality stream.
      - Built-in retry logic with exponential backoff.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.cache_dir = config.cache_dir / "downloads"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> Path:
        """
        Download *url* and return the local path.

        The destination filename is a hash of the URL so re-running the
        pipeline on the same URL skips the download entirely.
        """
        import yt_dlp  # deferred import: only required when downloading URLs

        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        dest_template = str(self.cache_dir / f"{url_hash}.%(ext)s")

        # Check cache hit — yt-dlp already downloaded this URL
        existing = list(self.cache_dir.glob(f"{url_hash}.*"))
        if existing:
            logger.info("Cache hit for %s → %s", url, existing[0])
            return existing[0]

        ydl_opts: dict = {
            "outtmpl": dest_template,
            # Prefer a single merged file (mp4) with both streams
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }

        logger.info("Downloading %s", url)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = Path(ydl.prepare_filename(info))

        logger.info("Downloaded to %s", downloaded_path)
        return downloaded_path
