"""Tests for the acoustic scorer — librosa calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from distillation.models import Segment
from distillation.scoring.acoustic import AcousticScorer


def make_segment(segment_id: int, start: float, end: float, text: str = "") -> Segment:
    word_count = max(1, int((end - start) * 2))  # ~2 words/sec
    if not text:
        text = " ".join(["word"] * word_count)
    return Segment(
        segment_id=segment_id,
        start=start,
        end=end,
        text=text,
        transcript_segment_ids=[segment_id],
    )


@pytest.fixture
def scorer():
    return AcousticScorer()


class TestAcousticScorer:
    @patch("distillation.scoring.acoustic.librosa")
    def test_extract_features_returns_all_segments(self, mock_librosa, scorer):
        """Should return one entry per segment."""
        sr = 16_000
        duration = 10.0
        mock_librosa.load.return_value = (np.zeros(int(sr * duration)), sr)
        mock_librosa.feature.rms.return_value = np.array([[0.1] * 100])

        segments = [
            make_segment(0, 0.0, 5.0),
            make_segment(1, 5.0, 10.0),
        ]
        result = scorer.extract_features(segments, "/fake/audio.wav")

        assert 0 in result
        assert 1 in result
        assert "energy" in result[0]
        assert "wpm" in result[0]

    @patch("distillation.scoring.acoustic.librosa")
    def test_energy_is_normalised(self, mock_librosa, scorer):
        """Energy values should be in [0, 1]."""
        sr = 16_000
        # Create audio with varying amplitude
        audio = np.concatenate([
            np.random.randn(sr * 5) * 0.1,   # quiet segment
            np.random.randn(sr * 5) * 0.9,   # loud segment
        ])
        mock_librosa.load.return_value = (audio, sr)
        mock_librosa.feature.rms.return_value = np.array([[0.5] * 100])

        segments = [
            make_segment(0, 0.0, 5.0),
            make_segment(1, 5.0, 10.0),
        ]
        result = scorer.extract_features(segments, "/fake/audio.wav")

        for seg_id in result:
            assert 0.0 <= result[seg_id]["energy"] <= 1.0

    @patch("distillation.scoring.acoustic.librosa")
    def test_wpm_calculation(self, mock_librosa, scorer):
        """WPM should be (word_count / duration_seconds) * 60."""
        sr = 16_000
        mock_librosa.load.return_value = (np.zeros(sr * 30), sr)
        mock_librosa.feature.rms.return_value = np.array([[0.1] * 100])

        # 20 words in 10 seconds = 120 WPM
        text = " ".join(["word"] * 20)
        segments = [make_segment(0, 0.0, 10.0, text=text)]
        result = scorer.extract_features(segments, "/fake/audio.wav")

        assert result[0]["wpm"] == pytest.approx(120.0, abs=0.5)

    @patch("distillation.scoring.acoustic.librosa")
    def test_zero_duration_segment(self, mock_librosa, scorer):
        """A zero-duration segment should get 0 WPM without division error."""
        sr = 16_000
        mock_librosa.load.return_value = (np.zeros(sr * 5), sr)
        mock_librosa.feature.rms.return_value = np.array([[0.1] * 100])

        segments = [make_segment(0, 5.0, 5.0, text="word")]
        result = scorer.extract_features(segments, "/fake/audio.wav")

        assert result[0]["energy"] == 0.0
        assert result[0]["wpm"] == 0.0

    @patch("distillation.scoring.acoustic.librosa")
    def test_audio_load_failure_returns_empty(self, mock_librosa, scorer):
        """If librosa can't load the audio, return empty dict gracefully."""
        mock_librosa.load.side_effect = Exception("File not found")

        segments = [make_segment(0, 0.0, 5.0)]
        result = scorer.extract_features(segments, "/nonexistent.wav")

        assert result == {}

    @patch("distillation.scoring.acoustic.librosa")
    def test_empty_segments_returns_empty(self, mock_librosa, scorer):
        """No segments should produce an empty result."""
        sr = 16_000
        mock_librosa.load.return_value = (np.zeros(sr * 5), sr)
        mock_librosa.feature.rms.return_value = np.array([[0.1] * 100])

        result = scorer.extract_features([], "/fake/audio.wav")
        assert result == {}
