"""
Core domain models for the Lecture Distillation Engine.

These are the canonical data structures that flow through every stage of the
pipeline.  Nothing domain-specific lives here — all stages receive and return
these objects, making each stage independently testable.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────────

class Domain(str, Enum):
    """
    The content domain controls LLM scoring prompts and integrity rules.
    Add new domains here without touching any other module.
    """
    KHUTBA = "khutba"           # Islamic sermon — theological integrity matters
    PODCAST = "podcast"         # Conversational; may tolerate partial thoughts
    BUSINESS_TALK = "business_talk"
    LECTURE = "lecture"         # Academic; expects structured arguments
    GENERIC = "generic"


class Language(str, Enum):
    EN = "en"
    AR = "ar"
    MIXED = "mixed"             # Arabic/English code-switching (common in khutbas)


# ── Transcription models ───────────────────────────────────────────────────────

class Word(BaseModel):
    """
    Single word with word-level timestamps from Whisper.
    This is the finest-grained unit in the system.
    """
    text: str
    start: float                # seconds
    end: float                  # seconds
    probability: float = 1.0    # Whisper word-level confidence

    @property
    def duration(self) -> float:
        return self.end - self.start


class TranscriptSegment(BaseModel):
    """
    A Whisper-native segment (typically a sentence or short phrase).
    Not to be confused with a semantic Segment — this is raw Whisper output.
    """
    id: int
    text: str
    start: float
    end: float
    words: list[Word] = Field(default_factory=list)
    avg_logprob: float = 0.0    # Whisper confidence; very negative = hallucination
    no_speech_prob: float = 0.0


class Transcript(BaseModel):
    """Complete transcript of the source media."""
    source_path: str
    duration: float             # total audio duration in seconds
    language: Language
    segments: list[TranscriptSegment]
    full_text: str = ""         # denormalised for fast LLM access

    @model_validator(mode="after")
    def build_full_text(self) -> "Transcript":
        if not self.full_text:
            self.full_text = " ".join(s.text.strip() for s in self.segments)
        return self


# ── Segmentation models ────────────────────────────────────────────────────────

class Segment(BaseModel):
    """
    A semantic unit — a topically coherent block of the lecture identified
    by the segmentation layer.  May span many TranscriptSegments.

    The key invariant: every Segment maps exactly to a contiguous span of
    TranscriptSegments from the parent Transcript, so timestamps are always
    anchored to real Whisper words.
    """
    segment_id: int
    start: float                # seconds into source audio
    end: float
    text: str                   # concatenated text of all contained TranscriptSegments
    transcript_segment_ids: list[int]   # which TranscriptSegment.id's are inside

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def words_per_second(self) -> float:
        word_count = len(self.text.split())
        return word_count / self.duration if self.duration > 0 else 0.0


# ── Scoring models ─────────────────────────────────────────────────────────────

class EngagementAxes(BaseModel):
    """
    Multi-axis engagement score produced by the LLM scorer.
    Each axis is [0.0, 1.0].  These are intentionally separate so downstream
    selection can weight them differently (e.g., khutbas might up-weight integrity).

    Design note: we resist collapsing to a single float here so that:
      a) Evaluation can expose which axis is failing.
      b) Domain-specific weighting is transparent, not baked into the score.
    """
    semantic_density: float = Field(ge=0.0, le=1.0)
    """Information per second.  High = every sentence adds something new."""

    emotional_resonance: float = Field(ge=0.0, le=1.0)
    """Does the speaker's language evoke feeling? Rhetorical questions, urgency."""

    standalone_coherence: float = Field(ge=0.0, le=1.0)
    """Can a cold viewer understand this clip without prior context?"""

    narrative_completeness: float = Field(ge=0.0, le=1.0)
    """Does it open a thought AND close it (or close with intentional tension)?"""

    domain_integrity: float = Field(ge=0.0, le=1.0)
    """
    Domain-specific: for khutba, no fatwa/ruling is left incomplete; hadith
    attribution is present.  For podcast, the claim is grounded. Etc.
    """

    hook_strength: float = Field(ge=0.0, le=1.0)
    """
    Would the first 3 seconds stop a scroll?  Evaluates the opening statement.
    """

    llm_rationale: str = ""
    """Free-text explanation from the LLM, useful for debugging and auditing."""


class ScoredSegment(BaseModel):
    """A Segment annotated with multi-axis scores and acoustic features."""
    segment: Segment
    scores: EngagementAxes
    acoustic_energy: float = 0.0    # normalised RMS of audio in this segment
    speech_rate_wpm: float = 0.0    # words per minute

    @property
    def composite_score(self) -> float:
        """
        Default composite — equal weights.  Override in ClipSelector with
        domain-specific weighting vectors.
        """
        s = self.scores
        return (
            s.semantic_density * 0.20
            + s.emotional_resonance * 0.20
            + s.standalone_coherence * 0.20
            + s.narrative_completeness * 0.15
            + s.domain_integrity * 0.15
            + s.hook_strength * 0.10
        )


# ── Selection models ───────────────────────────────────────────────────────────

class SubtitleLine(BaseModel):
    """A single subtitle cue (SRT/ASS compatible)."""
    index: int
    start: float    # seconds relative to clip start (not source)
    end: float
    text: str

    def to_srt_timestamp(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def to_srt_block(self) -> str:
        return (
            f"{self.index}\n"
            f"{self.to_srt_timestamp(self.start)} --> {self.to_srt_timestamp(self.end)}\n"
            f"{self.text}\n"
        )


class BRollSuggestion(BaseModel):
    """A suggested visual concept to overlay on the clip."""
    timestamp_in_clip: float    # seconds from clip start
    duration: float
    description: str            # e.g. "time-lapse of a mosque at dusk"
    tags: list[str] = Field(default_factory=list)   # searchable keywords for stock footage


class ClipMetadata(BaseModel):
    """LLM-generated metadata for platform publishing."""
    hook_caption: str           # First line / TikTok caption (≤150 chars)
    description: str            # Longer description for YouTube Shorts
    hashtags: list[str]
    thumbnail_suggestion: str   # Text description of a thumbnail concept
    b_roll_suggestions: list[BRollSuggestion] = Field(default_factory=list)


class Clip(BaseModel):
    """
    A selected clip, ready for export.  This is the final deliverable object.
    Everything needed to cut the video and publish is on this model.
    """
    clip_id: int
    source_path: str
    start: float                # seconds in source media
    end: float
    scored_segment: ScoredSegment
    subtitles: list[SubtitleLine] = Field(default_factory=list)
    metadata: Optional[ClipMetadata] = None
    output_path: Optional[str] = None  # set by export layer

    @property
    def duration(self) -> float:
        return self.end - self.start

    def export_srt(self) -> str:
        """Render all subtitle lines as an SRT string."""
        return "\n".join(line.to_srt_block() for line in self.subtitles)


# ── Pipeline result ────────────────────────────────────────────────────────────

class PipelineResult(BaseModel):
    """Top-level result returned by Pipeline.run()."""
    source_path: str
    domain: Domain
    transcript: Transcript
    segments: list[Segment]
    scored_segments: list[ScoredSegment]
    clips: list[Clip]

    @property
    def top_clip(self) -> Optional[Clip]:
        return self.clips[0] if self.clips else None
