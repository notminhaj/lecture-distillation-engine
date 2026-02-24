"""Tests for the clip selector."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from distillation.models import Domain, EngagementAxes, Segment, ScoredSegment
from distillation.selection.selector import ClipSelector


def make_scored_segment(
    segment_id: int,
    start: float,
    end: float,
    score_override: float = 0.5,
) -> ScoredSegment:
    axes = EngagementAxes(
        semantic_density=score_override,
        emotional_resonance=score_override,
        standalone_coherence=score_override,
        narrative_completeness=score_override,
        domain_integrity=score_override,
        hook_strength=score_override,
    )
    seg = Segment(
        segment_id=segment_id,
        start=start,
        end=end,
        text=f"Segment {segment_id} text.",
        transcript_segment_ids=[segment_id],
    )
    return ScoredSegment(segment=seg, scores=axes)


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.min_clip_duration = 30.0
    cfg.max_clip_duration = 90.0
    cfg.target_clip_count = 3
    return cfg


@pytest.fixture
def selector(config):
    return ClipSelector(config)


class TestClipSelector:
    def test_returns_at_most_target_count(self, selector):
        segments = [
            make_scored_segment(i, i * 60.0, i * 60.0 + 45.0)
            for i in range(10)
        ]
        clips = selector.select(segments, domain=Domain.KHUTBA)
        assert len(clips) <= 3

    def test_clips_are_non_overlapping(self, selector):
        segments = [
            make_scored_segment(i, i * 60.0, i * 60.0 + 50.0)
            for i in range(10)
        ]
        clips = selector.select(segments, domain=Domain.GENERIC)
        # Check no two clips overlap
        for i, a in enumerate(clips):
            for b in clips[i + 1:]:
                assert not (a.start < b.end and a.end > b.start), (
                    f"Clips {a.clip_id} and {b.clip_id} overlap"
                )

    def test_clips_ordered_by_timeline(self, selector):
        segments = [
            make_scored_segment(i, i * 60.0, i * 60.0 + 45.0)
            for i in range(6)
        ]
        clips = selector.select(segments)
        starts = [c.start for c in clips]
        assert starts == sorted(starts)

    def test_filters_by_min_duration(self, selector):
        """Segments shorter than min_clip_duration should be excluded."""
        short = make_scored_segment(0, 0.0, 10.0, score_override=0.99)  # too short
        long = make_scored_segment(1, 60.0, 110.0, score_override=0.50)
        # long is within [30, 90]? 50s → yes
        clips = selector.select([short, long])
        clip_segment_ids = [c.scored_segment.segment.segment_id for c in clips]
        assert 0 not in clip_segment_ids  # short excluded

    def test_empty_input_returns_empty(self, selector):
        assert selector.select([]) == []

    def test_highest_scoring_segment_selected(self, selector):
        """When segments don't overlap, the top scorer should always be chosen."""
        segments = [
            make_scored_segment(0, 0.0, 45.0, score_override=0.9),
            make_scored_segment(1, 60.0, 105.0, score_override=0.3),
            make_scored_segment(2, 120.0, 165.0, score_override=0.1),
        ]
        clips = selector.select(segments, domain=Domain.GENERIC)
        # First clip by score (seg 0 = 0.9) should appear
        selected_ids = {c.scored_segment.segment.segment_id for c in clips}
        assert 0 in selected_ids
