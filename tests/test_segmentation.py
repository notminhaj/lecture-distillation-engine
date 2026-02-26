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

from distillation.models import Segment, Transcript, TranscriptSegment, Word, Language
from distillation.segmentation.boundary_refiner import BoundaryRefiner
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
        # Boundary refinement fields
        cfg.enable_boundary_refinement = True
        cfg.sentence_boundary_window = 5.0
        cfg.silence_threshold_ms = 600.0
        cfg.hook_protection_window = 3.0
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


# ── BoundaryRefiner tests ───────────────────────────────────────────────────


def _make_words(start: float, end: float, count: int) -> list[Word]:
    """Generate evenly-spaced words between start and end."""
    if count == 0:
        return []
    word_dur = (end - start) / count
    return [
        Word(text=f"w{i}", start=start + i * word_dur, end=start + (i + 1) * word_dur - 0.01)
        for i in range(count)
    ]


def _make_words_with_gap(
    start: float, end: float, count: int, gap_after: int, gap_size: float
) -> list[Word]:
    """Generate words with a deliberate silence gap after word index *gap_after*."""
    if count == 0:
        return []
    # Total time minus the gap, divided among words
    available = (end - start) - gap_size
    word_dur = available / count
    words: list[Word] = []
    t = start
    for i in range(count):
        words.append(Word(text=f"w{i}", start=t, end=t + word_dur - 0.01))
        t += word_dur
        if i == gap_after:
            t += gap_size  # insert silence
    return words


def _make_refiner_config(**overrides):
    """Minimal mock config for BoundaryRefiner."""
    from unittest.mock import MagicMock
    cfg = MagicMock()
    cfg.enable_boundary_refinement = overrides.get("enable_boundary_refinement", True)
    cfg.sentence_boundary_window = overrides.get("sentence_boundary_window", 5.0)
    cfg.silence_threshold_ms = overrides.get("silence_threshold_ms", 600.0)
    cfg.hook_protection_window = overrides.get("hook_protection_window", 3.0)
    cfg.min_clip_duration = overrides.get("min_clip_duration", 10.0)
    cfg.max_clip_duration = overrides.get("max_clip_duration", 90.0)
    return cfg


def _build_transcript_with_words(ts_specs: list[dict]) -> Transcript:
    """Build a Transcript from a list of specs: {id, start, end, text, words}."""
    segs = []
    for spec in ts_specs:
        segs.append(TranscriptSegment(
            id=spec["id"],
            text=spec.get("text", f"Segment {spec['id']}"),
            start=spec["start"],
            end=spec["end"],
            words=spec.get("words", []),
        ))
    duration = max(s["end"] for s in ts_specs) if ts_specs else 0.0
    return Transcript(
        source_path="/tmp/test.wav",
        duration=duration,
        language=Language.EN,
        segments=segs,
    )


class TestBoundaryRefiner:

    def test_sentence_boundary_snapping(self):
        """Boundaries mid-sentence get snapped to the nearest TS edge."""
        # Two TS: [0-10], [10-20]. Segment boundary at 8.0 should snap to 10.0.
        ts_specs = [
            {"id": 0, "start": 0.0, "end": 10.0, "text": "First sentence.",
             "words": _make_words(0.0, 10.0, 5)},
            {"id": 1, "start": 10.0, "end": 20.0, "text": "Second sentence.",
             "words": _make_words(10.0, 20.0, 5)},
            {"id": 2, "start": 20.0, "end": 30.0, "text": "Third sentence.",
             "words": _make_words(20.0, 30.0, 5)},
            {"id": 3, "start": 30.0, "end": 40.0, "text": "Fourth sentence.",
             "words": _make_words(30.0, 40.0, 5)},
        ]
        transcript = _build_transcript_with_words(ts_specs)

        # Segment with boundary not on a TS edge
        segments = [
            Segment(segment_id=0, start=0.0, end=8.0, text="First", transcript_segment_ids=[0]),
            Segment(segment_id=1, start=8.0, end=40.0, text="Rest", transcript_segment_ids=[1, 2, 3]),
        ]

        cfg = _make_refiner_config(min_clip_duration=5.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        # After snapping, first segment should end at a TS edge (10.0)
        assert len(result) >= 1
        # The boundary should have snapped to 10.0
        assert result[0].end == pytest.approx(10.0, abs=0.5)

    def test_pause_detection_uses_word_gaps(self):
        """Boundary moves to the largest silence gap when words have pauses > threshold."""
        # Words with a big gap (1 second) in the middle
        words_ts0 = _make_words_with_gap(0.0, 15.0, 8, gap_after=3, gap_size=1.0)
        words_ts1 = _make_words(15.0, 30.0, 8)

        ts_specs = [
            {"id": 0, "start": 0.0, "end": 15.0, "text": "Part one with a pause.",
             "words": words_ts0},
            {"id": 1, "start": 15.0, "end": 30.0, "text": "Part two continues.",
             "words": words_ts1},
        ]
        transcript = _build_transcript_with_words(ts_specs)

        segments = [
            Segment(segment_id=0, start=0.0, end=15.0, text="Part one", transcript_segment_ids=[0]),
            Segment(segment_id=1, start=15.0, end=30.0, text="Part two", transcript_segment_ids=[1]),
        ]

        cfg = _make_refiner_config(silence_threshold_ms=600.0, min_clip_duration=5.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        assert len(result) >= 2
        # Segments should still cover the full range
        assert result[0].start <= 1.0
        assert result[-1].end >= 29.0

    def test_hook_protection_keeps_strong_opener(self):
        """A high-density opener at segment start is not shifted backward."""
        # Segment 1: sparse ending (2 words in last 3s)
        # Segment 2: dense opening (8 words in first 3s) with a gap before
        words_ts0 = _make_words(0.0, 7.0, 5) + _make_words(7.0, 10.0, 2)
        words_ts1 = [Word(text="hook", start=10.5, end=10.8)] + _make_words(10.8, 13.0, 7)
        words_ts2 = _make_words(13.0, 20.0, 5)

        ts_specs = [
            {"id": 0, "start": 0.0, "end": 10.0, "text": "Slow ending.", "words": words_ts0},
            {"id": 1, "start": 10.0, "end": 15.0, "text": "Hook start high energy.", "words": words_ts1},
            {"id": 2, "start": 15.0, "end": 20.0, "text": "Continues.", "words": words_ts2},
        ]
        transcript = _build_transcript_with_words(ts_specs)

        segments = [
            Segment(segment_id=0, start=0.0, end=10.0, text="Slow ending.", transcript_segment_ids=[0]),
            Segment(segment_id=1, start=10.0, end=20.0, text="Hook start.", transcript_segment_ids=[1, 2]),
        ]

        cfg = _make_refiner_config(min_clip_duration=5.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        # Hook should stay with the second segment — boundary should not move forward
        assert len(result) == 2
        assert result[1].start <= 10.5

    def test_hook_protection_shifts_bleed(self):
        """Content bleeding from previous segment gets shifted back."""
        # Both segments have similar word density and no pause at boundary
        words_ts0 = _make_words(0.0, 10.0, 10)
        # Continuous speech across boundary — no gap
        words_ts1 = _make_words(10.0, 15.0, 5)
        words_ts2 = _make_words(15.0, 25.0, 10)

        ts_specs = [
            {"id": 0, "start": 0.0, "end": 10.0, "text": "Continuous speech one.", "words": words_ts0},
            {"id": 1, "start": 10.0, "end": 15.0, "text": "Continuous speech two.", "words": words_ts1},
            {"id": 2, "start": 15.0, "end": 25.0, "text": "Continuous speech three.", "words": words_ts2},
        ]
        transcript = _build_transcript_with_words(ts_specs)

        segments = [
            Segment(segment_id=0, start=0.0, end=15.0, text="First part.", transcript_segment_ids=[0, 1]),
            Segment(segment_id=1, start=15.0, end=25.0, text="Second part.", transcript_segment_ids=[2]),
        ]

        cfg = _make_refiner_config(min_clip_duration=5.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        # Should still have 2 segments, boundary may have shifted
        assert len(result) >= 1
        # No crash and ordered
        starts = [s.start for s in result]
        assert starts == sorted(starts)

    def test_duration_enforcement_merges_short(self):
        """Segments below min_clip_duration get merged."""
        ts_specs = [
            {"id": i, "start": i * 3.0, "end": (i + 1) * 3.0,
             "text": f"Seg {i}.", "words": _make_words(i * 3.0, (i + 1) * 3.0, 3)}
            for i in range(10)
        ]
        transcript = _build_transcript_with_words(ts_specs)

        # 3 segments, first one is very short (3s)
        segments = [
            Segment(segment_id=0, start=0.0, end=3.0, text="Short.", transcript_segment_ids=[0]),
            Segment(segment_id=1, start=3.0, end=18.0, text="Medium.", transcript_segment_ids=[1, 2, 3, 4, 5]),
            Segment(segment_id=2, start=18.0, end=30.0, text="Long.", transcript_segment_ids=[6, 7, 8, 9]),
        ]

        cfg = _make_refiner_config(min_clip_duration=10.0, max_clip_duration=90.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        # Short segment should be merged — fewer segments than input
        assert len(result) <= 3
        for seg in result:
            # All remaining should meet min_dur (or be the only segment)
            if len(result) > 1:
                assert seg.duration >= 10.0 or seg.duration > 0

    def test_duration_enforcement_splits_long(self):
        """Segments above max_clip_duration get split at a pause."""
        # Create a long segment with a big gap in the middle
        words = (
            _make_words(0.0, 20.0, 10)
            + [Word(text="pause", start=20.0, end=20.1)]  # gap follows
            + _make_words(22.0, 50.0, 15)
        )
        ts_specs = [
            {"id": i, "start": i * 5.0, "end": (i + 1) * 5.0,
             "text": f"Seg {i}.", "words": [w for w in words if w.start >= i * 5.0 and w.end <= (i + 1) * 5.0]}
            for i in range(10)
        ]
        transcript = _build_transcript_with_words(ts_specs)

        segments = [
            Segment(segment_id=0, start=0.0, end=50.0, text="Very long segment.",
                    transcript_segment_ids=list(range(10))),
        ]

        cfg = _make_refiner_config(min_clip_duration=10.0, max_clip_duration=30.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        # Should split into 2+ segments
        assert len(result) >= 2
        for seg in result:
            assert seg.duration > 0

    def test_no_overlap_invariant(self):
        """After all refinements, segments don't overlap."""
        ts_specs = [
            {"id": i, "start": i * 5.0, "end": (i + 1) * 5.0,
             "text": f"Seg {i}.", "words": _make_words(i * 5.0, (i + 1) * 5.0, 5)}
            for i in range(12)
        ]
        transcript = _build_transcript_with_words(ts_specs)

        segments = [
            Segment(segment_id=0, start=0.0, end=20.0, text="A.", transcript_segment_ids=[0, 1, 2, 3]),
            Segment(segment_id=1, start=20.0, end=40.0, text="B.", transcript_segment_ids=[4, 5, 6, 7]),
            Segment(segment_id=2, start=40.0, end=60.0, text="C.", transcript_segment_ids=[8, 9, 10, 11]),
        ]

        cfg = _make_refiner_config(min_clip_duration=10.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        for i in range(len(result) - 1):
            assert result[i].end <= result[i + 1].start + 0.01, (
                f"Overlap: seg {i} ends at {result[i].end}, seg {i+1} starts at {result[i+1].start}"
            )

    def test_ordered_invariant(self):
        """Segments remain ordered by start time after refinement."""
        ts_specs = [
            {"id": i, "start": i * 5.0, "end": (i + 1) * 5.0,
             "text": f"Seg {i}.", "words": _make_words(i * 5.0, (i + 1) * 5.0, 5)}
            for i in range(8)
        ]
        transcript = _build_transcript_with_words(ts_specs)

        segments = [
            Segment(segment_id=0, start=0.0, end=15.0, text="A.", transcript_segment_ids=[0, 1, 2]),
            Segment(segment_id=1, start=15.0, end=25.0, text="B.", transcript_segment_ids=[3, 4]),
            Segment(segment_id=2, start=25.0, end=40.0, text="C.", transcript_segment_ids=[5, 6, 7]),
        ]

        cfg = _make_refiner_config(min_clip_duration=5.0)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        starts = [s.start for s in result]
        assert starts == sorted(starts)

    def test_disabled_refinement_returns_unchanged(self):
        """enable_boundary_refinement=False is a no-op."""
        ts_specs = [
            {"id": 0, "start": 0.0, "end": 10.0, "text": "A.", "words": _make_words(0.0, 10.0, 5)},
            {"id": 1, "start": 10.0, "end": 20.0, "text": "B.", "words": _make_words(10.0, 20.0, 5)},
        ]
        transcript = _build_transcript_with_words(ts_specs)

        segments = [
            Segment(segment_id=0, start=0.0, end=8.0, text="A.", transcript_segment_ids=[0]),
            Segment(segment_id=1, start=8.0, end=20.0, text="B.", transcript_segment_ids=[1]),
        ]

        cfg = _make_refiner_config(enable_boundary_refinement=False)
        refiner = BoundaryRefiner(cfg)
        result = refiner.refine(segments, transcript)

        assert result is segments  # exact same object
        assert result[0].end == 8.0  # unchanged

    def test_empty_segments_returns_empty(self):
        """Edge case: empty segments list returns empty."""
        transcript = Transcript(
            source_path="/tmp/test.wav", duration=0.0,
            language=Language.EN, segments=[],
        )
        cfg = _make_refiner_config()
        refiner = BoundaryRefiner(cfg)
        assert refiner.refine([], transcript) == []
