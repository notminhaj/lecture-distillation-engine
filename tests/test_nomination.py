"""Tests for LLM moment detector — all LLM calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from distillation.models import Domain, NominatedMoment, Language, Transcript, TranscriptSegment
from distillation.nomination.moment_detector import MomentDetector


def make_long_transcript(num_segments: int = 20, seg_duration: float = 5.0) -> Transcript:
    """Create a transcript with many segments for testing."""
    segments = []
    for i in range(num_segments):
        start = i * seg_duration
        end = start + seg_duration
        segments.append(TranscriptSegment(
            id=i,
            text=f"This is segment {i} of the lecture about important topics.",
            start=start,
            end=end,
            words=[],
        ))
    return Transcript(
        source_path="/tmp/test.wav",
        duration=num_segments * seg_duration,
        language=Language.EN,
        segments=segments,
    )


def make_nomination_response(nominations: list[dict]) -> str:
    return json.dumps(nominations)


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.openai_api_key = "test-key"
    cfg.openai_model = "gpt-4o-mini"
    cfg.min_clip_duration = 15.0
    cfg.max_clip_duration = 45.0
    return cfg


@pytest.fixture
def detector(config):
    with patch("distillation.nomination.moment_detector.OpenAI"):
        return MomentDetector(config)


class TestMomentDetectorParsing:
    """Test _parse_nominations in isolation."""

    def test_valid_json_array(self):
        raw = json.dumps([
            {"approximate_start": 10.0, "approximate_end": 30.0,
             "rationale": "Great hook", "opening_line": "Did you know..."},
        ])
        noms = MomentDetector._parse_nominations(raw)
        assert len(noms) == 1
        assert noms[0].approximate_start == 10.0

    def test_markdown_fenced_json(self):
        data = [{"approximate_start": 5.0, "approximate_end": 25.0,
                 "rationale": "Test", "opening_line": "Hello"}]
        raw = f"```json\n{json.dumps(data)}\n```"
        noms = MomentDetector._parse_nominations(raw)
        assert len(noms) == 1

    def test_invalid_json_returns_empty(self):
        noms = MomentDetector._parse_nominations("NOT JSON {{{")
        assert noms == []

    def test_missing_required_fields_skipped(self):
        raw = json.dumps([
            {"rationale": "No timestamps"},
            {"approximate_start": 10.0, "approximate_end": 30.0,
             "rationale": "Valid", "opening_line": "Hello"},
        ])
        noms = MomentDetector._parse_nominations(raw)
        assert len(noms) == 1


class TestTimestampSnapping:
    """Test that nominations snap correctly to transcript segment boundaries."""

    def test_snap_to_segments(self, detector):
        transcript = make_long_transcript(num_segments=20, seg_duration=5.0)
        nominations = [
            NominatedMoment(
                approximate_start=12.0, approximate_end=32.0,
                rationale="Test", opening_line="Hello",
            ),
        ]
        segments = detector._snap_to_segments(nominations, transcript)
        assert len(segments) == 1
        # Should snap to segment boundaries (10.0 and 35.0)
        assert segments[0].start == 10.0
        assert segments[0].end == 35.0

    def test_duration_constraint_filters(self, detector):
        transcript = make_long_transcript(num_segments=20, seg_duration=5.0)
        nominations = [
            # This would snap to only ~5s — too short
            NominatedMoment(
                approximate_start=10.0, approximate_end=14.0,
                rationale="Too short", opening_line="Hi",
            ),
        ]
        segments = detector._snap_to_segments(nominations, transcript)
        assert len(segments) == 0

    def test_empty_transcript(self, detector):
        transcript = Transcript(
            source_path="/tmp/test.wav",
            duration=0.0,
            language=Language.EN,
            segments=[],
        )
        nominations = [
            NominatedMoment(
                approximate_start=10.0, approximate_end=30.0,
                rationale="Test", opening_line="Hello",
            ),
        ]
        segments = detector._snap_to_segments(nominations, transcript)
        assert len(segments) == 0


class TestSentenceExtension:
    """Test that clips extend to complete the final sentence."""

    def test_extends_when_last_segment_has_no_period(self, detector):
        """If the last matched segment doesn't end with punctuation,
        the next segment should be included to complete the thought."""
        segments = [
            TranscriptSegment(id=0, text="Opening statement.", start=0.0, end=5.0, words=[]),
            TranscriptSegment(id=1, text="He will receive the same reward", start=5.0, end=10.0, words=[]),
            TranscriptSegment(id=2, text="the fasting person receives without decreasing.", start=10.0, end=15.0, words=[]),
            TranscriptSegment(id=3, text="Next topic starts here.", start=15.0, end=20.0, words=[]),
            TranscriptSegment(id=4, text="And more content follows.", start=20.0, end=25.0, words=[]),
        ]
        transcript = Transcript(
            source_path="/tmp/test.wav", duration=25.0, language=Language.EN, segments=segments,
        )
        nominations = [
            NominatedMoment(
                approximate_start=0.0, approximate_end=10.0,
                rationale="Test", opening_line="Opening",
            ),
        ]
        result = detector._snap_to_segments(nominations, transcript)
        assert len(result) == 1
        # Should extend past seg 1 ("...same reward") to include seg 2 ("...without decreasing.")
        assert result[0].end == 15.0
        assert "without decreasing" in result[0].text

    def test_no_extension_when_last_segment_ends_with_period(self, detector):
        """If the last segment already ends with a period, no extension needed."""
        segments = [
            TranscriptSegment(id=0, text="Complete thought here.", start=0.0, end=5.0, words=[]),
            TranscriptSegment(id=1, text="Another complete thought.", start=5.0, end=10.0, words=[]),
            TranscriptSegment(id=2, text="Third sentence here.", start=10.0, end=15.0, words=[]),
            TranscriptSegment(id=3, text="Fourth sentence here.", start=15.0, end=20.0, words=[]),
        ]
        transcript = Transcript(
            source_path="/tmp/test.wav", duration=20.0, language=Language.EN, segments=segments,
        )
        nominations = [
            NominatedMoment(
                approximate_start=0.0, approximate_end=10.0,
                rationale="Test", opening_line="Complete",
            ),
        ]
        result = detector._snap_to_segments(nominations, transcript)
        assert len(result) == 1
        assert result[0].end == 15.0  # includes seg at boundary but doesn't extend further

    def test_extension_respects_max_duration(self, detector):
        """Extension should not push the clip beyond max_clip_duration."""
        segments = [
            TranscriptSegment(id=i, text=f"Segment {i} no punctuation", start=i * 5.0, end=(i + 1) * 5.0, words=[])
            for i in range(20)
        ]
        # Add punctuation only to the very last one
        segments[-1] = TranscriptSegment(id=19, text="Final segment.", start=95.0, end=100.0, words=[])
        transcript = Transcript(
            source_path="/tmp/test.wav", duration=100.0, language=Language.EN, segments=segments,
        )
        nominations = [
            NominatedMoment(
                approximate_start=0.0, approximate_end=30.0,
                rationale="Test", opening_line="Segment 0",
            ),
        ]
        result = detector._snap_to_segments(nominations, transcript)
        # Should extend but not exceed max_clip_duration (45s)
        assert len(result) == 1
        assert result[0].duration <= detector.cfg.max_clip_duration

    def test_boundary_inclusive_end(self, detector):
        """A segment starting exactly at approximate_end should be included."""
        segments = [
            TranscriptSegment(id=0, text="First part.", start=0.0, end=10.0, words=[]),
            TranscriptSegment(id=1, text="Second part.", start=10.0, end=20.0, words=[]),
            TranscriptSegment(id=2, text="Third part.", start=20.0, end=30.0, words=[]),
            TranscriptSegment(id=3, text="Fourth part.", start=30.0, end=40.0, words=[]),
        ]
        transcript = Transcript(
            source_path="/tmp/test.wav", duration=40.0, language=Language.EN, segments=segments,
        )
        nominations = [
            NominatedMoment(
                approximate_start=0.0, approximate_end=20.0,
                rationale="Test", opening_line="First",
            ),
        ]
        result = detector._snap_to_segments(nominations, transcript)
        assert len(result) == 1
        # Segment starting at exactly 20.0 should be included (inclusive boundary)
        assert result[0].end == 30.0


class TestMomentDetectorIntegration:
    """Integration test with mocked OpenAI client."""

    def test_detect_returns_segments(self, detector):
        transcript = make_long_transcript(num_segments=20, seg_duration=5.0)
        nomination_data = [
            {"approximate_start": 10.0, "approximate_end": 35.0,
             "rationale": "Great moment", "opening_line": "This is segment 2"},
            {"approximate_start": 50.0, "approximate_end": 75.0,
             "rationale": "Another moment", "opening_line": "This is segment 10"},
        ]
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(nomination_data)))
        ]
        detector.client.chat.completions.create.return_value = mock_response

        segments = detector.detect(transcript, domain=Domain.KHUTBA, target_count=4)
        assert len(segments) == 2
        assert all(s.duration >= 15.0 for s in segments)

    def test_detect_empty_transcript(self, detector):
        transcript = Transcript(
            source_path="/tmp/test.wav",
            duration=0.0,
            language=Language.EN,
            segments=[],
        )
        segments = detector.detect(transcript, domain=Domain.GENERIC, target_count=4)
        assert segments == []
