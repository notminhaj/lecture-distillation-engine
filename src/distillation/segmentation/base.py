"""Abstract base class for segmenters."""

from abc import ABC, abstractmethod

from distillation.models import Segment, Transcript


class BaseSegmenter(ABC):
    """
    All segmenters receive a Transcript and return a list of Segments.

    Segments must be:
      - Non-overlapping.
      - Ordered by start time.
      - Bounded within the transcript duration.
      - Anchored to real TranscriptSegment IDs (no fabricated timestamps).
    """

    @abstractmethod
    def segment(self, transcript: Transcript) -> list[Segment]:
        """Partition *transcript* into semantically coherent Segments."""
        ...

    @staticmethod
    def _build_segment(
        segment_id: int,
        ts_segments: list,  # list[TranscriptSegment]
    ) -> Segment:
        """
        Utility: build a Segment from a contiguous list of TranscriptSegments.
        Ensures the text and timestamps are always consistent.
        """
        from distillation.models import Segment
        return Segment(
            segment_id=segment_id,
            start=ts_segments[0].start,
            end=ts_segments[-1].end,
            text=" ".join(s.text.strip() for s in ts_segments),
            transcript_segment_ids=[s.id for s in ts_segments],
        )
