"""
Subtitle generator — produces SubtitleLine objects from word-level timestamps.

Short-form video subtitle conventions differ from standard subtitles:
  - 2-5 words per line (not full sentences) for easy reading while scrolling.
  - Max line duration: ~2 seconds.
  - Lines should align to speech, not sentence boundaries.
  - Burned-in captions on TikTok/Reels must be highly readable.

Strategy:
  1. Walk the word-level timestamps for all TranscriptSegments inside the clip.
  2. Group words into short lines by word count and time gap.
  3. Re-zero timestamps relative to clip start (not source).
  4. Export as SubtitleLine objects; the export layer can render to SRT, ASS,
     or pass directly to ffmpeg's drawtext/subtitles filter.

For the MVP, we produce SRT-compatible lines.  Advanced features (word-highlight
karaoke style, ASS positioning) are noted as TODO and can be added later without
changing the data model.
"""

from __future__ import annotations

import logging

from distillation.models import Clip, SubtitleLine, Transcript, Word

logger = logging.getLogger(__name__)

# Tune these to match platform conventions
WORDS_PER_LINE = 4      # words before forcing a line break
MAX_LINE_DURATION = 2.5  # seconds; force a new line even if word count not hit


class SubtitleGenerator:
    """
    Generates per-clip subtitle lines from Whisper word-level timestamps.

    Falls back to sentence-level subtitles if word-level data is missing.
    """

    def generate(self, clip: Clip, transcript: Transcript) -> list[SubtitleLine]:
        """
        Generate SubtitleLine objects for *clip*, with timestamps
        relative to the clip start (not source media).
        """
        # Gather all Words that fall inside the clip's time range
        words = self._get_words_in_clip(clip, transcript)

        if words:
            lines = self._words_to_lines(words, clip_start=clip.start)
        else:
            # Fallback: one subtitle line per TranscriptSegment inside clip
            logger.warning(
                "No word-level timestamps for clip %d; falling back to segment-level",
                clip.clip_id,
            )
            lines = self._segment_fallback(clip, transcript)

        return lines

    @staticmethod
    def _get_words_in_clip(clip: Clip, transcript: Transcript) -> list[Word]:
        """Extract words from all TranscriptSegments that fall inside this clip."""
        words: list[Word] = []
        seg_ids = clip.scored_segment.segment.transcript_segment_ids
        seg_id_set = set(seg_ids)

        for ts_seg in transcript.segments:
            if ts_seg.id in seg_id_set:
                words.extend(ts_seg.words)

        # Filter to only words actually within clip boundaries
        return [w for w in words if w.start >= clip.start and w.end <= clip.end]

    @staticmethod
    def _words_to_lines(words: list[Word], clip_start: float) -> list[SubtitleLine]:
        """
        Group words into subtitle lines.

        Each line is at most WORDS_PER_LINE words or MAX_LINE_DURATION seconds.
        Timestamps are relative to clip_start.
        """
        lines: list[SubtitleLine] = []
        line_words: list[Word] = []
        line_index = 1

        for word in words:
            line_words.append(word)
            line_duration = line_words[-1].end - line_words[0].start

            should_break = (
                len(line_words) >= WORDS_PER_LINE
                or line_duration >= MAX_LINE_DURATION
            )

            if should_break:
                text = " ".join(w.text.strip() for w in line_words).strip()
                if text:
                    lines.append(SubtitleLine(
                        index=line_index,
                        start=line_words[0].start - clip_start,
                        end=line_words[-1].end - clip_start,
                        text=text,
                    ))
                    line_index += 1
                line_words = []

        # Flush remaining words
        if line_words:
            text = " ".join(w.text.strip() for w in line_words).strip()
            if text:
                lines.append(SubtitleLine(
                    index=line_index,
                    start=line_words[0].start - clip_start,
                    end=line_words[-1].end - clip_start,
                    text=text,
                ))

        return lines

    @staticmethod
    def _segment_fallback(clip: Clip, transcript: Transcript) -> list[SubtitleLine]:
        """Fallback: one subtitle per TranscriptSegment."""
        lines: list[SubtitleLine] = []
        seg_ids = set(clip.scored_segment.segment.transcript_segment_ids)

        for i, ts_seg in enumerate(transcript.segments):
            if ts_seg.id not in seg_ids:
                continue
            lines.append(SubtitleLine(
                index=i + 1,
                start=max(0.0, ts_seg.start - clip.start),
                end=max(0.0, ts_seg.end - clip.start),
                text=ts_seg.text.strip(),
            ))

        return lines
