"""
Shared fixtures for all tests.

Design principle: all fixtures that touch the LLM are mocked.  Tests should
be runnable without API keys or GPU — they validate logic, not model quality.
"""

from __future__ import annotations

import pytest

from distillation.models import (
    Domain,
    EngagementAxes,
    Language,
    Segment,
    ScoredSegment,
    Transcript,
    TranscriptSegment,
    Word,
)


@pytest.fixture
def sample_words() -> list[Word]:
    return [
        Word(text="The", start=0.0, end=0.3, probability=0.99),
        Word(text="Prophet", start=0.3, end=0.7, probability=0.99),
        Word(text="said", start=0.7, end=1.0, probability=0.98),
        Word(text="seek", start=1.1, end=1.4, probability=0.97),
        Word(text="knowledge", start=1.4, end=2.0, probability=0.99),
        Word(text="even", start=2.1, end=2.4, probability=0.99),
        Word(text="unto", start=2.4, end=2.7, probability=0.97),
        Word(text="China", start=2.7, end=3.2, probability=0.99),
    ]


@pytest.fixture
def sample_transcript_segments(sample_words) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(
            id=0,
            text="The Prophet said seek knowledge even unto China.",
            start=0.0,
            end=3.2,
            words=sample_words,
            avg_logprob=-0.2,
            no_speech_prob=0.01,
        ),
        TranscriptSegment(
            id=1,
            text="This is a reminder of our duty to seek beneficial knowledge.",
            start=3.5,
            end=8.0,
            words=[],
            avg_logprob=-0.3,
            no_speech_prob=0.01,
        ),
        TranscriptSegment(
            id=2,
            text="And Allah knows best.",
            start=8.1,
            end=9.5,
            words=[],
        ),
    ]


@pytest.fixture
def sample_transcript(sample_transcript_segments) -> Transcript:
    return Transcript(
        source_path="/tmp/test.wav",
        duration=9.5,
        language=Language.EN,
        segments=sample_transcript_segments,
    )


@pytest.fixture
def sample_segment(sample_transcript_segments) -> Segment:
    segs = sample_transcript_segments
    return Segment(
        segment_id=0,
        start=segs[0].start,
        end=segs[-1].end,
        text=" ".join(s.text for s in segs),
        transcript_segment_ids=[s.id for s in segs],
    )


@pytest.fixture
def sample_axes() -> EngagementAxes:
    return EngagementAxes(
        semantic_density=0.8,
        emotional_resonance=0.7,
        standalone_coherence=0.9,
        narrative_completeness=0.85,
        domain_integrity=0.95,
        hook_strength=0.75,
        llm_rationale="Strong opening with hadith citation; self-contained.",
    )


@pytest.fixture
def sample_scored_segment(sample_segment, sample_axes) -> ScoredSegment:
    return ScoredSegment(
        segment=sample_segment,
        scores=sample_axes,
        acoustic_energy=0.6,
        speech_rate_wpm=140.0,
    )
