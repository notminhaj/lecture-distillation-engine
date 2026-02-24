"""Abstract base class for scorers."""

from abc import ABC, abstractmethod
from pathlib import Path

from distillation.models import Domain, Segment, ScoredSegment, Transcript


class BaseScorer(ABC):
    @abstractmethod
    def score(
        self,
        segments: list[Segment],
        transcript: Transcript,
        *,
        audio_path: str | Path | None = None,
        domain: Domain = Domain.GENERIC,
    ) -> list[ScoredSegment]:
        """Score all segments and return ScoredSegments in the same order."""
        ...
