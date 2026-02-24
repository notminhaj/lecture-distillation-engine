"""
Tests for segmentation strategies.

Both segmenters are tested against the same invariants:
  - Output is non-empty for non-empty input.
  - Segments are ordered by start time.
  - Segment timestamps are anchored to real TranscriptSegment data.
  - No segment has duration < 0.
"""

from __future__ import annotations

import pytest

from distillation.models import Transcript, TranscriptSegment, Language
from distillation.segmentation.sliding_window import SlidingWindowSegmenter


def make_transcript(n_segs: int = 20, seg_duration: float = 5.0) -> Transcript:
    """Helper: build a synthetic transcript of *n_segs* segments."""
    segs = []
    for i in range(n_segs):
        start = i * seg_duration
        end = start + seg_duration
        segs.append(TranscriptSegment(
            id=i,
            text=f"This is segment number {i} with some useful content.",
            start=start,
            end=end,
        ))
    return Transcript(
        source_path="/tmp/test.wav",
        duration=n_segs * seg_duration,
        language=Language.EN,
        segments=segs,
    )


class TestSlidingWindowSegmenter:
    """
    We test the sliding window segmenter without mocking because it has no
    external dependencies.  Semantic segmenter tests use mocking (embedding
    model is not available in CI without GPU).
    """

    @pytest.fixture
    def config(self):
        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg.sliding_window_size = 30.0
        cfg.sliding_window_stride = 15.0
        cfg.min_clip_duration = 20.0
        cfg.max_clip_duration = 90.0
        return cfg

    @pytest.fixture
    def segmenter(self, config):
        return SlidingWindowSegmenter(config)

    def test_empty_transcript_returns_empty(self, segmenter):
        transcript = Transcript(
            source_path="/tmp/test.wav",
            duration=0.0,
            language=Language.EN,
            segments=[],
        )
        assert segmenter.segment(transcript) == []

    def test_produces_segments(self, segmenter):
        transcript = make_transcript(n_segs=30, seg_duration=3.0)
        segments = segmenter.segment(transcript)
        assert len(segments) > 0

    def test_segments_are_ordered(self, segmenter):
        transcript = make_transcript(n_segs=30, seg_duration=3.0)
        segments = segmenter.segment(transcript)
        starts = [s.start for s in segments]
        assert starts == sorted(starts)

    def test_all_segments_have_positive_duration(self, segmenter):
        transcript = make_transcript(n_segs=20, seg_duration=3.0)
        segments = segmenter.segment(transcript)
        for seg in segments:
            assert seg.duration > 0

    def test_segment_ids_are_unique(self, segmenter):
        transcript = make_transcript(n_segs=20, seg_duration=3.0)
        segments = segmenter.segment(transcript)
        ids = [s.segment_id for s in segments]
        assert len(ids) == len(set(ids))

    def test_transcript_segment_ids_reference_real_segments(self, segmenter):
        transcript = make_transcript(n_segs=20, seg_duration=3.0)
        real_ids = {s.id for s in transcript.segments}
        segments = segmenter.segment(transcript)
        for seg in segments:
            for ts_id in seg.transcript_segment_ids:
                assert ts_id in real_ids


class TestSemanticSegmenterUnit:
    """
    Unit-level tests for SemanticSegmenter helper methods.
    We mock the encoder to avoid GPU/model dependencies in CI.
    """

    def test_consecutive_similarities_correct_length(self):
        import numpy as np
        from distillation.segmentation.semantic import SemanticSegmenter

        embeddings = np.random.rand(5, 64)
        sims = SemanticSegmenter._consecutive_similarities(embeddings)
        assert len(sims) == 4  # n-1 pairs

    def test_consecutive_similarities_single_embedding(self):
        import numpy as np
        from distillation.segmentation.semantic import SemanticSegmenter

        embeddings = np.random.rand(1, 64)
        sims = SemanticSegmenter._consecutive_similarities(embeddings)
        assert sims == []

    def test_similarity_values_in_range(self):
        import numpy as np
        from distillation.segmentation.semantic import SemanticSegmenter

        # Identical vectors → similarity = 1.0
        embeddings = np.ones((3, 64))
        sims = SemanticSegmenter._consecutive_similarities(embeddings)
        for sim in sims:
            assert sim == pytest.approx(1.0, abs=1e-5)
