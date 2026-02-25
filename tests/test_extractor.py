"""Tests for the audio extractor — ffmpeg calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distillation.ingestion.extractor import (
    AudioExtractor,
    CHANNELS,
    CODEC,
    CONTAINER,
    SAMPLE_RATE,
)


@pytest.fixture
def config(tmp_path):
    cfg = MagicMock()
    cfg.cache_dir = tmp_path / "cache"
    return cfg


@pytest.fixture
def extractor(config):
    return AudioExtractor(config)


class TestAudioExtractor:
    def test_cache_dir_created(self, config, tmp_path):
        AudioExtractor(config)
        assert (tmp_path / "cache" / "audio").is_dir()

    def test_cache_hit_returns_existing(self, extractor, tmp_path):
        """If WAV already exists, skip ffmpeg and return it."""
        audio_dir = tmp_path / "cache" / "audio"
        cached_wav = audio_dir / "lecture.wav"
        cached_wav.write_text("fake audio")

        result = extractor.extract(Path("/some/path/lecture.mp4"))
        assert result == cached_wav

    @patch("distillation.ingestion.extractor.subprocess.run")
    def test_extract_calls_ffmpeg(self, mock_run, extractor, tmp_path):
        """When no cache hit, ffmpeg should be called with correct args."""
        mock_run.return_value = MagicMock(returncode=0)
        video_path = Path("/videos/sermon.mp4")

        result = extractor.extract(video_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert str(video_path) in cmd
        assert "-vn" in cmd
        assert CODEC in cmd
        assert str(SAMPLE_RATE) in cmd
        assert str(CHANNELS) in cmd
        assert str(result) == str(tmp_path / "cache" / "audio" / "sermon.wav")

    @patch("distillation.ingestion.extractor.subprocess.run")
    def test_extract_raises_on_ffmpeg_failure(self, mock_run, extractor):
        """RuntimeError should be raised if ffmpeg exits non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stderr="codec error")

        with pytest.raises(RuntimeError, match="ffmpeg audio extraction failed"):
            extractor.extract(Path("/videos/bad.mp4"))

    def test_output_has_wav_extension(self, extractor, tmp_path):
        """Output path should always be .wav regardless of input extension."""
        audio_dir = tmp_path / "cache" / "audio"
        # Pre-create so cache hit triggers — just verify the path
        wav = audio_dir / "video.wav"
        wav.write_text("fake")
        result = extractor.extract(Path("/input/video.mkv"))
        assert result.suffix == ".wav"

    def test_constants_match_whisper_requirements(self):
        """Verify the constants match what Whisper expects."""
        assert SAMPLE_RATE == 16_000
        assert CHANNELS == 1
        assert CODEC == "pcm_s16le"
        assert CONTAINER == "wav"
