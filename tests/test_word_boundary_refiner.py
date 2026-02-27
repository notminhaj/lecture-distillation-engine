"""Tests for the post-selection WordBoundaryRefiner."""

from __future__ import annotations

import pytest

from distillation.models import (
    Clip,
    EngagementAxes,
    Language,
    Segment,
    ScoredSegment,
    Transcript,
    TranscriptSegment,
    Word,
)
from distillation.word_boundary_refiner import END_PAD, START_PAD, WordBoundaryRefiner


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_word(text: str, start: float, end: float) -> Word:
    return Word(text=text, start=start, end=end)


def make_clip(clip_id: int, start: float, end: float) -> Clip:
    axes = EngagementAxes(
        semantic_density=0.8,
        emotional_resonance=0.7,
        standalone_coherence=0.9,
        narrative_completeness=0.85,
        domain_integrity=0.95,
        hook_strength=0.75,
    )
    seg = Segment(
        segment_id=clip_id,
        start=start,
        end=end,
        text="test segment",
        transcript_segment_ids=[clip_id],
    )
    scored = ScoredSegment(segment=seg, scores=axes)
    return Clip(
        clip_id=clip_id,
        source_path="/tmp/test.mp4",
        start=start,
        end=end,
        scored_segment=scored,
    )


def make_transcript(words: list[Word], duration: float) -> Transcript:
    if words:
        seg = TranscriptSegment(
            id=0,
            text=" ".join(w.text for w in words),
            start=words[0].start,
            end=words[-1].end,
            words=words,
        )
        segments = [seg]
    else:
        segments = []
    return Transcript(
        source_path="/tmp/test.wav",
        duration=duration,
        language=Language.EN,
        segments=segments,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestWordBoundaryRefiner:
    def setup_method(self):
        self.refiner = WordBoundaryRefiner()

    def test_expands_start_and_end(self):
        """Boundaries expand by START_PAD before first word and END_PAD after last."""
        words = [
            make_word("Hello", 5.0, 5.5),
            make_word("world", 5.6, 6.0),
        ]
        transcript = make_transcript(words, duration=60.0)
        clip = make_clip(0, start=5.0, end=6.0)

        refined = self.refiner.refine_all([clip], transcript)

        assert len(refined) == 1
        assert refined[0].start == pytest.approx(5.0 - START_PAD)
        assert refined[0].end == pytest.approx(6.0 + END_PAD)

    def test_clamps_start_to_zero(self):
        """new_start must not go below 0 even when START_PAD would push it negative."""
        words = [make_word("First", 0.05, 0.4)]
        transcript = make_transcript(words, duration=30.0)
        clip = make_clip(0, start=0.05, end=0.4)

        refined = self.refiner.refine_all([clip], transcript)

        # 0.05 - 0.15 = -0.10 → clamped to 0.0
        assert refined[0].start == pytest.approx(0.0)

    def test_clamps_end_to_duration(self):
        """new_end must not exceed transcript.duration."""
        words = [make_word("Last", 59.7, 59.95)]
        transcript = make_transcript(words, duration=60.0)
        clip = make_clip(0, start=59.7, end=59.95)

        refined = self.refiner.refine_all([clip], transcript)

        # 59.95 + 0.20 = 60.15 → clamped to 60.0
        assert refined[0].end == pytest.approx(60.0)

    def test_no_words_in_span_returns_clip_unchanged(self):
        """If no words fall within the clip span, the original clip is returned."""
        words = [make_word("Outside", 10.0, 10.5)]
        transcript = make_transcript(words, duration=60.0)
        clip = make_clip(0, start=20.0, end=25.0)

        refined = self.refiner.refine_all([clip], transcript)

        assert refined[0].start == pytest.approx(20.0)
        assert refined[0].end == pytest.approx(25.0)

    def test_refines_multiple_clips_independently(self):
        """Each clip is refined against the same word pool independently."""
        words = [
            make_word("Alpha", 1.0, 1.5),
            make_word("Beta", 10.0, 10.5),
        ]
        transcript = make_transcript(words, duration=60.0)
        clip_a = make_clip(0, start=1.0, end=1.5)
        clip_b = make_clip(1, start=10.0, end=10.5)

        refined = self.refiner.refine_all([clip_a, clip_b], transcript)

        assert len(refined) == 2
        assert refined[0].start == pytest.approx(1.0 - START_PAD)
        assert refined[0].end == pytest.approx(1.5 + END_PAD)
        assert refined[1].start == pytest.approx(10.0 - START_PAD)
        assert refined[1].end == pytest.approx(10.5 + END_PAD)

    def test_empty_clips_list_returns_empty(self):
        """refine_all([]) should return []."""
        transcript = Transcript(
            source_path="/tmp/test.wav",
            duration=60.0,
            language=Language.EN,
            segments=[],
        )
        assert self.refiner.refine_all([], transcript) == []

    def test_other_clip_fields_preserved(self):
        """Only start/end are updated; all other Clip fields remain identical."""
        words = [make_word("Word", 5.0, 5.5)]
        transcript = make_transcript(words, duration=60.0)
        clip = make_clip(0, start=5.0, end=5.5)

        refined = self.refiner.refine_all([clip], transcript)

        r = refined[0]
        assert r.clip_id == clip.clip_id
        assert r.source_path == clip.source_path
        assert r.scored_segment == clip.scored_segment
        assert r.subtitles == clip.subtitles
        assert r.metadata == clip.metadata

    def test_words_at_exact_boundary_are_included(self):
        """Words whose start equals clip.start and end equals clip.end are included."""
        words = [
            make_word("Open", 3.0, 3.4),   # start == clip.start
            make_word("Close", 6.6, 7.0),  # end == clip.end
        ]
        transcript = make_transcript(words, duration=30.0)
        clip = make_clip(0, start=3.0, end=7.0)

        refined = self.refiner.refine_all([clip], transcript)

        assert refined[0].start == pytest.approx(3.0 - START_PAD)
        assert refined[0].end == pytest.approx(7.0 + END_PAD)
