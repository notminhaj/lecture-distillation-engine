"""Abstract base class for transcribers."""

from abc import ABC, abstractmethod
from pathlib import Path

from distillation.models import Transcript


class BaseTranscriber(ABC):
    """
    Contract for all transcription backends.

    Swapping Whisper for a cloud ASR (Assembly AI, Deepgram) is a one-file
    change as long as the implementation honours this interface.
    """

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> Transcript:
        """
        Transcribe the audio file at *audio_path*.

        Must return a Transcript with word-level timestamps on every
        TranscriptSegment.  Implementations that cannot provide word-level
        timestamps should set an empty words list and accept that subtitle
        generation quality will be reduced.
        """
        ...
