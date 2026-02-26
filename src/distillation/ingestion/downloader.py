"""
Media downloader using yt-dlp.

Supports any URL yt-dlp handles (YouTube, Vimeo, direct video links).
Downloads the best available video+audio stream and caches locally.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
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

        ydl_opts = self._build_ydl_opts(dest_template)

        # First attempt
        try:
            return self._do_download(ydl_opts, url)
        except Exception as first_err:
            if "Sign in to confirm" not in str(first_err) and "bot" not in str(first_err).lower():
                raise

        # Bot detection — try with cookies
        logger.warning("Bot detection triggered. Attempting cookie-based auth...")
        ydl_opts = self._build_ydl_opts(dest_template, force_cookies=True)
        return self._do_download(ydl_opts, url)

    def _build_ydl_opts(self, dest_template: str, force_cookies: bool = False) -> dict:
        """Construct yt-dlp options dict."""
        ydl_opts: dict = {
            "outtmpl": dest_template,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }

        # Enable Node.js runtime for PoToken generation (YouTube anti-bot).
        if shutil.which("node"):
            ydl_opts["js_runtimes"] = {"node": {}}

        # Cookie strategy:
        # 1. Explicit cookies file (most reliable on Windows)
        # 2. Auto-extracted cookies via Selenium (fallback)
        # 3. Browser cookies via yt-dlp (works on Linux/macOS)
        cookie_file = self._resolve_cookie_file(force_extract=force_cookies)
        if cookie_file:
            ydl_opts["cookiefile"] = str(cookie_file)
        elif self.cfg.cookies_from_browser:
            ydl_opts["cookiesfrombrowser"] = (self.cfg.cookies_from_browser,)

        return ydl_opts

    def _resolve_cookie_file(self, force_extract: bool = False) -> Path | None:
        """Find or create a cookies.txt file."""
        # Check explicit config path
        if self.cfg.cookies_file:
            p = Path(self.cfg.cookies_file)
            if p.exists():
                return p
            logger.warning("cookies_file %s not found", p)

        # Check default location
        default_path = self.cfg.cache_dir / "cookies.txt"
        if default_path.exists() and not force_extract:
            return default_path

        # Auto-extract via Selenium
        if force_extract or not default_path.exists():
            try:
                from distillation.ingestion.cookie_helper import extract_cookies_to_file
                logger.info("Auto-extracting browser cookies via Selenium...")
                extract_cookies_to_file(default_path)
                if default_path.exists():
                    return default_path
            except Exception as e:
                logger.debug("Cookie auto-extraction failed: %s", e)

        return None

    @staticmethod
    def _do_download(ydl_opts: dict, url: str) -> Path:
        """Execute the yt-dlp download and return the output path."""
        import yt_dlp

        logger.info("Downloading %s", url)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded_path = Path(ydl.prepare_filename(info))

        logger.info("Downloaded to %s", downloaded_path)
        return downloaded_path
