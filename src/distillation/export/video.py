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

    def __init__(self, config: Config, force_reencode: bool = True) -> None:
        self.cfg = config
        self.force_reencode = force_reencode
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, clip: Clip, source_video: str) -> Path:
        """
        Cut *clip* from *source_video* and write all assets.
        Burns ASS subtitles into the video by default.
        Sets clip.output_path and returns the clip video path.
        """
        clip_dir = self.output_dir / f"clip_{clip.clip_id:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)

        video_path = clip_dir / f"clip_{clip.clip_id:03d}.mp4"
        srt_path = clip_dir / f"clip_{clip.clip_id:03d}.srt"
        ass_path = clip_dir / f"clip_{clip.clip_id:03d}.ass"
        meta_path = clip_dir / f"clip_{clip.clip_id:03d}_metadata.json"

        # ── 1. Write subtitle files before cutting (needed for burn-in) ───────
        if clip.subtitles:
            srt_path.write_text(clip.export_srt(), encoding="utf-8")
            ass_path.write_text(clip.export_ass(), encoding="utf-8")
            logger.info("Wrote subtitles: %s, %s", srt_path, ass_path)

        # ── 2. Cut video (with burned-in subtitles if available) ──────────────
        self._cut_video(source_video, clip, video_path, ass_path if clip.subtitles else None)

        # ── 3. Write metadata JSON ─────────────────────────────────────────────
        self._write_metadata(clip, meta_path)

        clip.output_path = str(video_path)
        return video_path

    def _cut_video(
        self, source: str, clip: Clip, dest: Path, ass_path: Path | None = None,
    ) -> None:
        duration = clip.end - clip.start + 0.2

        if self.force_reencode:
            # Two-pass approach for accurate subtitle sync:
            #   Pass 1: Cut the video (no subtitles) — timestamps reset to 0.
            #   Pass 2: Burn ASS subtitles into the already-cut video.
            # This guarantees subs start at t=0 matching the cut video exactly.
            if ass_path and ass_path.exists():
                # Pass 1: cut to a temporary file
                tmp_cut = dest.with_suffix(".tmp.mp4")
                self._run_ffmpeg([
                    "ffmpeg", "-y",
                    "-ss", str(clip.start),
                    "-i", source,
                    "-t", str(duration),
                    "-c:v", "libx264", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    str(tmp_cut),
                ], clip.clip_id)

                # Ensure fonts are available next to the ASS file
                self._copy_fonts(ass_path)

                # Pass 2: burn subtitles into the cut video
                ass_filter_path = str(ass_path).replace("\\", "/")
                clip_fonts = ass_path.parent / "fonts"
                if clip_fonts.is_dir():
                    local_fonts = str(clip_fonts).replace("\\", "/")
                    vf = f"ass='{ass_filter_path}':fontsdir='{local_fonts}'"
                else:
                    vf = f"ass='{ass_filter_path}'"
                self._run_ffmpeg([
                    "ffmpeg", "-y",
                    "-i", str(tmp_cut),
                    "-vf", vf,
                    "-c:v", "libx264", "-crf", "23",
                    "-c:a", "copy",
                    str(dest),
                ], clip.clip_id)
                tmp_cut.unlink(missing_ok=True)
            else:
                # No subtitles — single pass
                self._run_ffmpeg([
                    "ffmpeg", "-y",
                    "-ss", str(clip.start),
                    "-i", source,
                    "-t", str(duration),
                    "-c:v", "libx264", "-crf", "23",
                    "-c:a", "aac", "-b:a", "128k",
                    str(dest),
                ], clip.clip_id)
        else:
            # Stream copy: fast but snaps to nearest keyframe.
            # Cannot burn in subtitles with stream copy (no re-encode).
            self._run_ffmpeg([
                "ffmpeg", "-y",
                "-ss", str(clip.start),
                "-i", source,
                "-t", str(duration),
                "-c", "copy",
                str(dest),
            ], clip.clip_id)
            if ass_path:
                logger.warning(
                    "Stream-copy mode: subtitles NOT burned in for clip %d. "
                    "Use --reencode for burned-in subs.",
                    clip.clip_id,
                )

    def _run_ffmpeg(self, cmd: list[str], clip_id: int) -> None:
        logger.info("ffmpeg clip %d: %s", clip_id, " ".join(cmd[:6]) + " ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg clip cut failed for clip {clip_id}:\n{result.stderr}"
            )

    @staticmethod
    def _copy_fonts(ass_path: Path) -> None:
        """Copy bundled fonts next to the ASS file for ffmpeg to find."""
        import shutil
        fonts_dir = Path(__file__).resolve().parents[3] / "fonts"
        if not fonts_dir.is_dir():
            return
        clip_fonts = ass_path.parent / "fonts"
        clip_fonts.mkdir(exist_ok=True)
        for f in fonts_dir.iterdir():
            if f.suffix.lower() in (".ttf", ".otf"):
                shutil.copy2(f, clip_fonts / f.name)

    @staticmethod
    def _write_metadata(clip: Clip, dest: Path) -> None:
        meta: dict = {
            "clip_id": clip.clip_id,
            "source_start": clip.start,
            "source_end": clip.end,
            "duration": clip.duration,
            "exported_duration": clip.duration + 0.2,
            "scores": clip.scored_segment.scores.model_dump(),
            "composite_score": clip.scored_segment.composite_score,
            "acoustic_energy": clip.scored_segment.acoustic_energy,
            "speech_rate_wpm": clip.scored_segment.speech_rate_wpm,
        }
        if clip.metadata:
            meta["metadata"] = clip.metadata.model_dump()

        dest.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote metadata: %s", dest)
