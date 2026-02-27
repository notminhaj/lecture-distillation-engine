"""
Post-selection word boundary refinement.

After clip selection, this pass aligns clip boundaries to exact Whisper
word timestamps, preventing half-cut words at the start/end of exported
clips.  It runs deterministically with no LLM involvement.
"""

from __future__ import annotations

import logging

from distillation.models import Clip, Transcript, Word

logger = logging.getLogger(__name__)

# Padding around the first/last word (seconds)
START_PAD: float = 0.15
END_PAD: float = 0.20


class WordBoundaryRefiner:
    """
    Expands each clip's start/end to include full words at both edges.

    Algorithm per clip:
      1. Collect all words from the transcript that fall within
         [clip.start, clip.end].
      2. new_start = first_word.start - START_PAD  (clamped to 0)
         new_end   = last_word.end   + END_PAD     (clamped to transcript.duration)
      3. Return a new Clip with updated start/end; all other fields unchanged.

    If no words are found inside the clip span, the clip is returned unchanged.
    """

    def refine_all(self, clips: list[Clip], transcript: Transcript) -> list[Clip]:
        """Refine boundaries of every clip and return an updated list."""
        all_words = self._collect_words(transcript)
        return [self._refine_clip(clip, all_words, transcript.duration) for clip in clips]

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _collect_words(transcript: Transcript) -> list[Word]:
        """Flatten word-level timestamps from every TranscriptSegment."""
        words: list[Word] = []
        for seg in transcript.segments:
            words.extend(seg.words)
        return words

    def _refine_clip(
        self,
        clip: Clip,
        all_words: list[Word],
        total_duration: float,
    ) -> Clip:
        inside = [
            w for w in all_words
            if w.start >= clip.start and w.end <= clip.end
        ]

        if not inside:
            logger.debug(
                "clip_id=%d: no words in [%.3f, %.3f]; boundaries unchanged",
                clip.clip_id, clip.start, clip.end,
            )
            return clip

        first_word = inside[0]
        last_word = inside[-1]

        new_start = max(0.0, first_word.start - START_PAD)
        new_end = min(total_duration, last_word.end + END_PAD)

        logger.debug(
            "clip_id=%d: [%.3f, %.3f] → [%.3f, %.3f]",
            clip.clip_id, clip.start, clip.end, new_start, new_end,
        )

        return clip.model_copy(update={"start": new_start, "end": new_end})
