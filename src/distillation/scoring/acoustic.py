"""
Acoustic feature extractor for engagement signals.

Why acoustic features alongside LLM scoring?
  - Energy correlates with speaker emphasis — a speaker raising their voice
    at the start of a segment is a natural hook signal.
  - Speech rate (WPM) distinguishes slow/deliberate speech (emotional weight)
    from rapid delivery (excitement, urgency).
  - These features are cheap (librosa, no API call) and complementary to
    semantic scores — a segment can score high semantically but low
    acoustically (monotone delivery) and vice versa.

These features are stored on ScoredSegment and can be incorporated into
ClipSelector's weighting, but are NOT passed to the LLM (it can't hear audio).
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

from distillation.models import Segment

logger = logging.getLogger(__name__)


class AcousticScorer:
    """Extracts acoustic engagement features for each segment."""

    def extract_features(
        self, segments: list[Segment], audio_path: str
    ) -> dict[int, dict]:
        """
        Returns a dict mapping segment_id → feature dict with keys:
          - energy: float [0, 1] — normalised RMS energy
          - wpm: float — words per minute

        Loading the full audio once and slicing is much faster than loading
        per-segment (avoids repeated disk seeks and format parsing).
        """
        try:
            y, sr = librosa.load(audio_path, sr=16_000, mono=True)
        except Exception as e:
            logger.warning("Could not load audio for acoustic scoring: %s", e)
            return {}

        # Compute RMS energy over the full track once
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        max_rms = float(np.max(rms)) if np.max(rms) > 0 else 1.0

        results: dict[int, dict] = {}
        for seg in segments:
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) == 0:
                results[seg.segment_id] = {"energy": 0.0, "wpm": 0.0}
                continue

            # Normalised RMS energy
            seg_rms = float(np.sqrt(np.mean(segment_audio ** 2)))
            normalised_energy = min(seg_rms / max_rms, 1.0)

            # Words per minute from transcript word count
            word_count = len(seg.text.split())
            wpm = (word_count / seg.duration) * 60.0 if seg.duration > 0 else 0.0

            results[seg.segment_id] = {
                "energy": round(normalised_energy, 4),
                "wpm": round(wpm, 1),
            }

        return results
