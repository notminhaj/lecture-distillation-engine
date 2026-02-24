"""
Video export using ffmpeg.

For each selected Clip:
  1. Cut the source video with stream-copy for speed (no re-encode).
  2. Optionally burn in subtitles using ffmpeg's subtitles filter.
  3. Write SRT alongside the clip for platform upload (YouTube auto-subs suck).
  4. Write metadata JSON for downstream automation (scheduling bots, APIs).

Stream-copy vs re-encode:
  - We use `-c copy` by default for speed.  This is lossless and fast but
    can produce slightly inaccurate cuts (snaps to nearest keyframe).
  - For publication-quality cuts, set `force_reencode=True` to use
    `-c:v libx264 -c:a aac` with exact seek.  ~10× slower.

Output structure:
  outputs/
    clip_000/
      clip_000.mp4
      clip_000.srt
      clip_000_metadata.json
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from distillation.config import Config
from distillation.models import Clip

logger = logging.getLogger(__name__)


class VideoExporter:
    """Cuts clips from source video and writes all associated assets."""

    def __init__(self, config: Config, force_reencode: bool = False) -> None:
        self.cfg = config
        self.force_reencode = force_reencode
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, clip: Clip, source_video: str) -> Path:
        """
        Cut *clip* from *source_video* and write all assets.
        Sets clip.output_path and returns the clip video path.
        """
        clip_dir = self.output_dir / f"clip_{clip.clip_id:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        video_path = clip_dir / f"clip_{clip.clip_id:03d}.mp4"
        srt_path = clip_dir / f"clip_{clip.clip_id:03d}.srt"
        meta_path = clip_dir / f"clip_{clip.clip_id:03d}_metadata.json"

        # ── 1. Cut video ───────────────────────────────────────────────────────
        self._cut_video(source_video, clip, video_path)

        # ── 2. Write SRT ───────────────────────────────────────────────────────
        if clip.subtitles:
            srt_path.write_text(clip.export_srt(), encoding="utf-8")
            logger.info("Wrote SRT: %s", srt_path)

        # ── 3. Write metadata JSON ─────────────────────────────────────────────
        self._write_metadata(clip, meta_path)

        clip.output_path = str(video_path)
        return video_path

    def _cut_video(self, source: str, clip: Clip, dest: Path) -> None:
        duration = clip.end - clip.start

        if self.force_reencode:
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(clip.start),     # input seek (accurate with re-encode)
                "-i", source,
                "-t", str(duration),
                "-c:v", "libx264",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                str(dest),
            ]
        else:
            # Stream copy: fast but snaps to nearest keyframe.
            # -ss before -i = fast seek; may have ~2s inaccuracy.
            # For exact cuts, use -ss after -i (slow but frame-accurate).
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(clip.start),
                "-i", source,
                "-t", str(duration),
                "-c", "copy",
                str(dest),
            ]

        logger.info("Cutting clip %d: %.1f–%.1f s", clip.clip_id, clip.start, clip.end)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg clip cut failed for clip {clip.clip_id}:\n{result.stderr}"
            )

    @staticmethod
    def _write_metadata(clip: Clip, dest: Path) -> None:
        meta: dict = {
            "clip_id": clip.clip_id,
            "source_start": clip.start,
            "source_end": clip.end,
            "duration": clip.duration,
            "scores": clip.scored_segment.scores.model_dump(),
            "composite_score": clip.scored_segment.composite_score,
            "acoustic_energy": clip.scored_segment.acoustic_energy,
            "speech_rate_wpm": clip.scored_segment.speech_rate_wpm,
        }
        if clip.metadata:
            meta["metadata"] = clip.metadata.model_dump()

        dest.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote metadata: %s", dest)
