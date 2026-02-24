"""
Whisper-based transcription using faster-whisper.

Why faster-whisper over openai-whisper?
  - CTranslate2 backend: 2-4× faster with the same model quality.
  - Lower VRAM: int8 quantisation fits large-v3 on 6 GB GPU.
  - Word-level timestamps built-in (no extra library).
  - Supports CPU inference well via int8.

Key design decisions:
  - We always request word_timestamps=True so downstream subtitle generation
    can produce per-word karaoke-style captions if needed.
  - We detect likely hallucination via avg_logprob threshold and flag those
    segments rather than silently including them.
  - Language detection is automatic; we store it on the Transcript object.
"""

from __future__ import annotations

import logging
from pathlib import Path

from distillation.config import Config
from distillation.models import Language, Transcript, TranscriptSegment, Word
from distillation.transcription.base import BaseTranscriber

logger = logging.getLogger(__name__)

# Segments with avg_logprob below this are likely hallucinations or silence
HALLUCINATION_LOGPROB_THRESHOLD = -1.0


class WhisperTranscriber(BaseTranscriber):
    """Transcribes audio using faster-whisper with word-level timestamps."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._model = None  # lazy load — model is large, skip on import

    def _get_model(self):
        """Load model on first use (avoids paying startup cost for tests)."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading Whisper model '%s' on %s (%s)",
                self.cfg.whisper_model,
                self.cfg.whisper_device,
                self.cfg.whisper_compute_type,
            )
            self._model = WhisperModel(
                self.cfg.whisper_model,
                device=self.cfg.whisper_device,
                compute_type=self.cfg.whisper_compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str | Path) -> Transcript:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._get_model()

        logger.info("Transcribing %s", audio_path)
        segments_iter, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,            # skip long silences (common in sermons)
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
            beam_size=5,
        )

        transcript_segments: list[TranscriptSegment] = []
        for i, seg in enumerate(segments_iter):
            words = self._parse_words(seg.words or [])
            ts = TranscriptSegment(
                id=i,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                words=words,
                avg_logprob=seg.avg_logprob,
                no_speech_prob=seg.no_speech_prob,
            )
            if seg.avg_logprob < HALLUCINATION_LOGPROB_THRESHOLD:
                logger.warning(
                    "Segment %d likely hallucinated (logprob=%.2f): %r",
                    i, seg.avg_logprob, seg.text[:60]
                )
            transcript_segments.append(ts)

        lang = self._map_language(info.language)

        # Compute total duration from last segment end
        duration = transcript_segments[-1].end if transcript_segments else 0.0

        return Transcript(
            source_path=str(audio_path),
            duration=duration,
            language=lang,
            segments=transcript_segments,
        )

    @staticmethod
    def _parse_words(raw_words: list) -> list[Word]:
        """Convert faster-whisper Word objects to our Word model."""
        words = []
        for w in raw_words:
            words.append(Word(
                text=w.word,
                start=w.start,
                end=w.end,
                probability=w.probability,
            ))
        return words

    @staticmethod
    def _map_language(lang_code: str) -> Language:
        mapping = {"en": Language.EN, "ar": Language.AR}
        return mapping.get(lang_code, Language.MIXED)
