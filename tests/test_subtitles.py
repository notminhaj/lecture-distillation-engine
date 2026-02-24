"""Tests for subtitle generation."""

from __future__ import annotations

import pytest

from distillation.generation.subtitles import SubtitleGenerator, WORDS_PER_LINE
from distillation.models import Clip, Language, SubtitleLine, Transcript, TranscriptSegment, Word


@pytest.fixture
def generator():
    return SubtitleGenerator()


@pytest.fixture
def clip_with_words(sample_scored_segment, sample_transcript):
    return Clip(
        clip_id=0,
        source_path="/tmp/source.mp4",
        start=0.0,
        end=sample_transcript.duration,
        scored_segment=sample_scored_segment,
    )


class TestSubtitleGenerator:
    def test_generates_subtitle_lines(self, generator, clip_with_words, sample_transcript):
        lines = generator.generate(clip_with_words, sample_transcript)
        assert isinstance(lines, list)

    def test_timestamps_relative_to_clip_start(
        self, generator, clip_with_words, sample_transcript
    ):
        lines = generator.generate(clip_with_words, sample_transcript)
        for line in lines:
            assert line.start >= 0.0
            assert line.end >= line.start

    def test_line_indices_are_sequential(self, generator, clip_with_words, sample_transcript):
        lines = generator.generate(clip_with_words, sample_transcript)
        for i, line in enumerate(lines):
            assert line.index == i + 1

    def test_lines_have_non_empty_text(self, generator, clip_with_words, sample_transcript):
        lines = generator.generate(clip_with_words, sample_transcript)
        for line in lines:
            assert line.text.strip() != ""

    def test_words_per_line_respected(self, generator, clip_with_words, sample_transcript):
        lines = generator.generate(clip_with_words, sample_transcript)
        for line in lines:
            word_count = len(line.text.split())
            # Lines should not significantly exceed WORDS_PER_LINE
            # (small overflow allowed due to final line flush)
            assert word_count <= WORDS_PER_LINE + 2

    def test_empty_words_falls_back_to_segment_level(self, generator):
        """Test that missing word timestamps trigger segment-level fallback."""
        from distillation.models import EngagementAxes, Segment, ScoredSegment
        from unittest.mock import MagicMock

        seg = Segment(
            segment_id=0,
            start=0.0,
            end=5.0,
            text="Hello world.",
            transcript_segment_ids=[0],
        )
        scored = ScoredSegment(
            segment=seg,
            scores=EngagementAxes(
                semantic_density=0.5,
                emotional_resonance=0.5,
                standalone_coherence=0.5,
                narrative_completeness=0.5,
                domain_integrity=0.5,
                hook_strength=0.5,
            ),
        )
        clip = Clip(
            clip_id=0,
            source_path="/tmp/x.mp4",
            start=0.0,
            end=5.0,
            scored_segment=scored,
        )
        transcript = Transcript(
            source_path="/tmp/test.wav",
            duration=5.0,
            language=Language.EN,
            segments=[
                TranscriptSegment(
                    id=0,
                    text="Hello world.",
                    start=0.0,
                    end=5.0,
                    words=[],  # No word-level timestamps
                )
            ],
        )
        lines = generator.generate(clip, transcript)
        assert len(lines) >= 1
        assert "Hello" in lines[0].text
