"""Tests for core data models — validation, properties, and serialisation."""

import pytest
from pydantic import ValidationError

from distillation.models import (
    Clip,
    EngagementAxes,
    Segment,
    ScoredSegment,
    SubtitleLine,
    Transcript,
    TranscriptSegment,
    Language,
    Word,
)


class TestWord:
    def test_duration(self):
        w = Word(text="hello", start=1.0, end=1.5)
        assert w.duration == pytest.approx(0.5)


class TestTranscript:
    def test_full_text_auto_built(self, sample_transcript_segments):
        t = Transcript(
            source_path="/tmp/test.wav",
            duration=9.5,
            language=Language.EN,
            segments=sample_transcript_segments,
        )
        assert "seek knowledge" in t.full_text

    def test_full_text_not_rebuilt_if_provided(self, sample_transcript_segments):
        t = Transcript(
            source_path="/tmp/test.wav",
            duration=9.5,
            language=Language.EN,
            segments=sample_transcript_segments,
            full_text="CUSTOM",
        )
        assert t.full_text == "CUSTOM"


class TestSegment:
    def test_duration(self, sample_segment):
        assert sample_segment.duration == pytest.approx(9.5, abs=0.5)

    def test_words_per_second_positive(self, sample_segment):
        assert sample_segment.words_per_second > 0


class TestEngagementAxes:
    def test_score_bounds_enforced(self):
        with pytest.raises(ValidationError):
            EngagementAxes(
                semantic_density=1.5,   # out of bounds
                emotional_resonance=0.5,
                standalone_coherence=0.5,
                narrative_completeness=0.5,
                domain_integrity=0.5,
                hook_strength=0.5,
            )

    def test_valid_axes(self, sample_axes):
        assert 0.0 <= sample_axes.semantic_density <= 1.0


class TestScoredSegment:
    def test_composite_score_in_range(self, sample_scored_segment):
        score = sample_scored_segment.composite_score
        assert 0.0 <= score <= 1.0

    def test_composite_weights_sum_to_1(self, sample_scored_segment):
        """Verify the weights in composite_score sum to 1.0."""
        # If all axes = 1.0, composite should be 1.0
        from distillation.models import EngagementAxes, ScoredSegment, Segment
        perfect_axes = EngagementAxes(
            semantic_density=1.0,
            emotional_resonance=1.0,
            standalone_coherence=1.0,
            narrative_completeness=1.0,
            domain_integrity=1.0,
            hook_strength=1.0,
        )
        ss = ScoredSegment(segment=sample_scored_segment.segment, scores=perfect_axes)
        assert ss.composite_score == pytest.approx(1.0, abs=0.01)


class TestSubtitleLine:
    def test_srt_timestamp_format(self):
        line = SubtitleLine(index=1, start=0.0, end=2.5, text="Hello world")
        ts = line.to_srt_timestamp(65.123)
        assert ts == "00:01:05,123"

    def test_srt_block_structure(self):
        line = SubtitleLine(index=1, start=0.5, end=2.0, text="Test")
        block = line.to_srt_block()
        assert "1\n" in block
        assert "Test" in block
        assert "-->" in block


class TestClip:
    def test_duration(self, sample_scored_segment):
        clip = Clip(
            clip_id=0,
            source_path="/tmp/source.mp4",
            start=10.0,
            end=50.0,
            scored_segment=sample_scored_segment,
        )
        assert clip.duration == pytest.approx(40.0)

    def test_export_srt(self, sample_scored_segment):
        from distillation.models import SubtitleLine
        clip = Clip(
            clip_id=0,
            source_path="/tmp/source.mp4",
            start=0.0,
            end=10.0,
            scored_segment=sample_scored_segment,
            subtitles=[
                SubtitleLine(index=1, start=0.5, end=2.0, text="Test subtitle"),
            ],
        )
        srt = clip.export_srt()
        assert "Test subtitle" in srt
        assert "-->" in srt
