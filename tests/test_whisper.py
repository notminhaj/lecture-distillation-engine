"""Tests for Whisper transcriber — model loading and inference are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distillation.models import Language
from distillation.transcription.whisper import (
    HALLUCINATION_LOGPROB_THRESHOLD,
    WhisperTranscriber,
)


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.whisper_model = "large-v3"
    cfg.whisper_device = "cpu"
    cfg.whisper_compute_type = "int8"
    return cfg


@pytest.fixture
def transcriber(config):
    return WhisperTranscriber(config)


def _make_mock_word(text: str, start: float, end: float, prob: float = 0.99):
    w = MagicMock()
    w.word = text
    w.start = start
    w.end = end
    w.probability = prob
    return w


def _make_mock_segment(
    text: str,
    start: float,
    end: float,
    words: list | None = None,
    avg_logprob: float = -0.2,
    no_speech_prob: float = 0.01,
):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.words = words or []
    seg.avg_logprob = avg_logprob
    seg.no_speech_prob = no_speech_prob
    return seg


class TestWhisperTranscriber:
    def test_transcribe_file_not_found(self, transcriber):
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe("/nonexistent/path.wav")

    def test_transcribe_returns_transcript(self, transcriber, tmp_path):
        """Full mock transcription flow produces a valid Transcript."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake audio")

        mock_words = [
            _make_mock_word("The", 0.0, 0.3),
            _make_mock_word("Prophet", 0.3, 0.7),
        ]
        mock_segments = [
            _make_mock_segment("The Prophet said", 0.0, 1.5, words=mock_words),
            _make_mock_segment("seek knowledge", 2.0, 3.5, words=[]),
        ]
        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
        transcriber._model = mock_model

        result = transcriber.transcribe(audio_file)

        assert len(result.segments) == 2
        assert result.language == Language.EN
        assert result.duration == 3.5
        assert "Prophet" in result.full_text

    def test_transcribe_word_parsing(self, transcriber, tmp_path):
        """Word-level timestamps are correctly parsed into Word objects."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake")

        mock_words = [
            _make_mock_word("Hello", 0.0, 0.5, prob=0.95),
            _make_mock_word("world", 0.5, 1.0, prob=0.98),
        ]
        mock_segments = [
            _make_mock_segment("Hello world", 0.0, 1.0, words=mock_words),
        ]
        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
        transcriber._model = mock_model

        result = transcriber.transcribe(audio_file)
        words = result.segments[0].words
        assert len(words) == 2
        assert words[0].text == "Hello"
        assert words[0].start == 0.0
        assert words[1].probability == 0.98

    def test_hallucination_detection(self, transcriber, tmp_path):
        """Segments with very low logprob should still be included but logged."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake")

        mock_segments = [
            _make_mock_segment(
                "Normal segment", 0.0, 2.0, avg_logprob=-0.3
            ),
            _make_mock_segment(
                "Hallucinated garbage", 2.0, 4.0,
                avg_logprob=HALLUCINATION_LOGPROB_THRESHOLD - 0.5,
            ),
        ]
        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter(mock_segments), mock_info)
        transcriber._model = mock_model

        result = transcriber.transcribe(audio_file)
        # Both segments are included (hallucinated ones are flagged, not dropped)
        assert len(result.segments) == 2

    def test_empty_transcription(self, transcriber, tmp_path):
        """Empty audio should produce a transcript with duration 0."""
        audio_file = tmp_path / "test.wav"
        audio_file.write_text("fake")

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), mock_info)
        transcriber._model = mock_model

        result = transcriber.transcribe(audio_file)
        assert len(result.segments) == 0
        assert result.duration == 0.0


class TestLanguageMapping:
    def test_english(self):
        assert WhisperTranscriber._map_language("en") == Language.EN

    def test_arabic(self):
        assert WhisperTranscriber._map_language("ar") == Language.AR

    def test_unknown_defaults_to_mixed(self):
        assert WhisperTranscriber._map_language("fr") == Language.MIXED
        assert WhisperTranscriber._map_language("unknown") == Language.MIXED


class TestParseWords:
    def test_converts_raw_words(self):
        raw = [
            _make_mock_word("hello", 0.0, 0.5, 0.99),
            _make_mock_word("world", 0.5, 1.0, 0.95),
        ]
        words = WhisperTranscriber._parse_words(raw)
        assert len(words) == 2
        assert words[0].text == "hello"
        assert words[1].end == 1.0

    def test_empty_words(self):
        assert WhisperTranscriber._parse_words([]) == []
