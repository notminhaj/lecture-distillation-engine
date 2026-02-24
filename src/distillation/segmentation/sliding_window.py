"""
Sliding window segmenter — simple, fast, domain-agnostic.

Useful as:
  - A baseline to benchmark against semantic segmentation.
  - A fallback when the embedding model is unavailable.
  - A pre-segmentation step before LLM-level analysis.

The window always snaps to TranscriptSegment boundaries so we never cut
mid-sentence.  Given a 60s window and 15s stride, this produces overlapping
candidate segments; duplicates are resolved in the scoring/selection layer.
"""

from __future__ import annotations

import logging

from distillation.config import Config
from distillation.models import Segment, Transcript, TranscriptSegment
from distillation.segmentation.base import BaseSegmenter

logger = logging.getLogger(__name__)


class SlidingWindowSegmenter(BaseSegmenter):
    """
    Creates segments by sliding a fixed-duration window across the transcript.

    This is deliberately naive — it doesn't understand topic boundaries.
    Use SemanticSegmenter in production; this exists for benchmarking and
    as a guaranteed-available fallback.
    """

    def __init__(self, config: Config) -> None:
        self.window_size = config.sliding_window_size
        self.stride = config.sliding_window_stride
        self.min_duration = config.min_clip_duration
        self.max_duration = config.max_clip_duration

    def segment(self, transcript: Transcript) -> list[Segment]:
        segs = transcript.segments
        if not segs:
            return []

        results: list[Segment] = []
        seg_id = 0
        window_start = 0.0

        while window_start < transcript.duration:
            window_end = min(window_start + self.window_size, transcript.duration)

            # Collect TranscriptSegments that fall inside this window
            window_ts: list[TranscriptSegment] = [
                s for s in segs
                if s.start >= window_start and s.end <= window_end
            ]
            # Also include segments that straddle the window boundary
            if not window_ts:
                window_ts = [s for s in segs if s.start < window_end and s.end > window_start]

            if window_ts:
                segment = self._build_segment(seg_id, window_ts)
                if segment.duration >= self.min_duration:
                    results.append(segment)
                    seg_id += 1

            window_start += self.stride

        logger.info(
            "SlidingWindowSegmenter produced %d segments (window=%.0fs stride=%.0fs)",
            len(results), self.window_size, self.stride
        )
        return results
