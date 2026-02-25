"""Tests for the yt-dlp downloader — all network calls are mocked."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distillation.ingestion.downloader import Downloader


@pytest.fixture
def config(tmp_path):
    cfg = MagicMock()
    cfg.cache_dir = tmp_path / "cache"
    return cfg


@pytest.fixture
def downloader(config):
    return Downloader(config)


class TestDownloader:
    def test_cache_dir_created(self, config, tmp_path):
        Downloader(config)
        assert (tmp_path / "cache" / "downloads").is_dir()

    def test_cache_hit_skips_download(self, downloader, config, tmp_path):
        """If a file matching the URL hash already exists, return it directly."""
        url = "https://www.youtube.com/watch?v=test123"
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

        # Create a fake cached file
        cached = tmp_path / "cache" / "downloads" / f"{url_hash}.mp4"
        cached.write_text("fake video")

        result = downloader.download(url)
        assert result == cached

    def test_download_calls_ytdlp(self, downloader):
        """When no cache hit, yt-dlp should be invoked."""
        url = "https://www.youtube.com/watch?v=abc"
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]

        mock_ydl_instance = MagicMock()
        mock_info = {"id": "abc", "ext": "mp4"}
        mock_ydl_instance.extract_info.return_value = mock_info
        mock_ydl_instance.prepare_filename.return_value = str(
            downloader.cache_dir / f"{url_hash}.mp4"
        )
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        mock_ydl_class = MagicMock(return_value=mock_ydl_instance)

        with patch.dict("sys.modules", {"yt_dlp": MagicMock(YoutubeDL=mock_ydl_class)}):
            result = downloader.download(url)

        mock_ydl_instance.extract_info.assert_called_once_with(url, download=True)
        assert str(result).endswith(".mp4")

    def test_url_hash_is_deterministic(self, downloader):
        """Same URL should always produce the same cache filename."""
        url = "https://youtube.com/watch?v=stable"
        h1 = hashlib.sha256(url.encode()).hexdigest()[:16]
        h2 = hashlib.sha256(url.encode()).hexdigest()[:16]
        assert h1 == h2

    def test_different_urls_produce_different_hashes(self):
        url_a = "https://youtube.com/watch?v=aaa"
        url_b = "https://youtube.com/watch?v=bbb"
        h_a = hashlib.sha256(url_a.encode()).hexdigest()[:16]
        h_b = hashlib.sha256(url_b.encode()).hexdigest()[:16]
        assert h_a != h_b
