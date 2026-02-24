"""
Audio extraction from video files using ffmpeg.

Whisper works on 16kHz mono WAV.  This module normalises any input
(mp4, mkv, mp3, m4a, webm …) to exactly that format so the transcription
layer never has to handle format variance.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from distillation.config import Config

logger = logging.getLogger(__name__)

# Whisper's required audio format
SAMPLE_RATE = 16_000
CHANNELS = 1
CODEC = "pcm_s16le"
CONTAINER = "wav"


class AudioExtractor:
    """Extracts and normalises audio from any ffmpeg-supported container."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.cache_dir = config.cache_dir / "audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_path: Path) -> Path:
        """
        Extract audio from *video_path* and return path to 16kHz mono WAV.

        Idempotent: if the WAV already exists, return it without re-running ffmpeg.
        """
        out_path = self.cache_dir / f"{video_path.stem}.wav"
        if out_path.exists():
            logger.info("Audio cache hit: %s", out_path)
            return out_path

        logger.info("Extracting audio from %s", video_path)
        cmd = [
            "ffmpeg",
            "-y",                   # overwrite without prompt
            "-i", str(video_path),
            "-vn",                  # no video
            "-acodec", CODEC,
            "-ar", str(SAMPLE_RATE),
            "-ac", str(CHANNELS),
            str(out_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed:\n{result.stderr}"
            )

        logger.info("Audio written to %s", out_path)
        return out_path
