"""
LLM-driven moment detection.

Instead of segmenting by topic boundaries and hoping good clips emerge,
this module asks the LLM to *find* the most compelling moments directly.

Flow:
  1. Send the full transcript (chunked if long) to the LLM.
  2. LLM nominates top N clip-worthy moments with approximate timestamps.
  3. Snap approximate timestamps to exact word-level boundaries from Whisper.
  4. Return Segment objects that flow into the existing Scoring → Selection pipeline.

This replaces (or supplements) semantic segmentation for candidate generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from distillation.config import Config
from distillation.models import (
    Domain,
    NominatedMoment,
    Segment,
    Transcript,
    TranscriptSegment,
)

logger = logging.getLogger(__name__)

# Maximum transcript words per LLM chunk (avoids context limit issues).
CHUNK_WORD_LIMIT = 6000

# Domain-specific guidance for moment nomination.
DOMAIN_NOMINATION_CONTEXT: dict[Domain, str] = {
    Domain.KHUTBA: """
You are finding the most powerful moments from an Islamic khutba (sermon).
Prioritise moments that:
- Open with a hadith, Quranic ayah, or a rhetorical question about faith
- Deliver a self-contained spiritual lesson or admonition
- Contain emotional peaks (urgency, awe, hope, warning)
- Include complete hadith attributions (not clipped mid-narration)
Avoid moments that start with "so...", "and as I was saying...", or transitions.
""",
    Domain.PODCAST: """
You are finding the most engaging moments from a conversational podcast.
Prioritise moments that:
- Open with a provocative question or counter-intuitive claim
- Deliver a surprising insight or a well-told anecdote
- Feature strong back-and-forth energy between speakers
""",
    Domain.BUSINESS_TALK: """
You are finding the most impactful moments from a business/leadership talk.
Prioritise moments with actionable advice, data-backed insights, or memorable frameworks.
""",
    Domain.LECTURE: """
You are finding the most compelling moments from an academic lecture.
Prioritise moments with clear explanations, surprising facts, or strong argumentative arcs.
""",
    Domain.GENERIC: "",
}


class MomentDetector:
    """Asks the LLM to identify clip-worthy moments from a transcript."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.client = OpenAI(api_key=config.openai_api_key)
        # Diagnostic: store all nominations from the last detect() call
        self.last_nominations: list[NominatedMoment] = []

    def detect(
        self,
        transcript: Transcript,
        domain: Domain = Domain.GENERIC,
        target_count: int = 10,
    ) -> list[Segment]:
        """
        Detect clip-worthy moments and return them as Segments.

        Args:
            transcript:   Full transcript with word-level timestamps.
            domain:       Content domain for prompt tuning.
            target_count: How many moments to nominate (2x target_clip_count
                          to give the scorer/selector room to pick the best).

        Returns:
            List of Segments snapped to word-level timestamps, ready for scoring.
        """
        # Build timestamped transcript text for the LLM
        stamped_lines = self._build_timestamped_text(transcript)

        # Chunk if needed
        chunks = self._chunk_text(stamped_lines, CHUNK_WORD_LIMIT)

        all_nominations: list[NominatedMoment] = []
        for chunk in chunks:
            nominations = self._nominate_chunk(
                chunk, domain, target_count=target_count
            )
            all_nominations.extend(nominations)

        logger.info("LLM nominated %d moments", len(all_nominations))

        # Snap to word-level timestamps and build Segments
        segments = self._snap_to_segments(all_nominations, transcript)
        logger.info("Snapped to %d valid segments", len(segments))

        # Store for diagnostics
        self.last_nominations = all_nominations

        return segments

    @staticmethod
    def _build_timestamped_text(transcript: Transcript) -> str:
        """Build a timestamped transcript for the LLM to read."""
        lines = []
        for seg in transcript.segments:
            ts = f"[{seg.start:.1f}s–{seg.end:.1f}s]"
            lines.append(f"{ts} {seg.text.strip()}")
        return "\n".join(lines)

    @staticmethod
    def _chunk_text(text: str, word_limit: int) -> list[str]:
        """Split text into chunks that fit within the word limit."""
        words = text.split()
        if len(words) <= word_limit:
            return [text]

        chunks = []
        lines = text.split("\n")
        current_chunk: list[str] = []
        current_words = 0

        for line in lines:
            line_words = len(line.split())
            if current_words + line_words > word_limit and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_words = 0
            current_chunk.append(line)
            current_words += line_words

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _nominate_chunk(
        self,
        chunk_text: str,
        domain: Domain,
        target_count: int,
    ) -> list[NominatedMoment]:
        """Ask the LLM to nominate moments from a transcript chunk."""
        system_prompt = self._build_system_prompt(domain, target_count)

        response = self.client.chat.completions.create(
            model=self.cfg.openai_model,
            max_tokens=4096,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"TRANSCRIPT:\n{chunk_text}"},
            ],
        )

        raw = response.choices[0].message.content
        return self._parse_nominations(raw)

    @staticmethod
    def _build_system_prompt(domain: Domain, target_count: int) -> str:
        domain_ctx = DOMAIN_NOMINATION_CONTEXT.get(domain, "")
        return f"""You are an expert content strategist for short-form video.
Your job: read a timestamped lecture transcript and identify the {target_count} most
clip-worthy moments (15–45 seconds each) that would perform best as standalone
short-form clips (TikTok, YouTube Shorts, Instagram Reels).

{domain_ctx}

SELECTION CRITERIA (in priority order):
- Self-contained: viewer needs NO prior context from the lecture
- Hook-strong: the first 3 seconds grab attention (question, claim, emotion)
- Emotionally resonant: speaker's language evokes feeling
- Narratively complete: opens and closes a thought within 15–45 seconds
- Duration: aim for 15–45 seconds per moment

SENTENCE BOUNDARY RULES (MANDATORY — these are hard rules, not preferences):
- Moments MUST begin at a natural sentence boundary where the speaker starts a
  new thought. The opening_line MUST start with a capitalized word.
- REJECT any moment whose opening_line begins with a lowercase word.
- REJECT any moment whose opening_line begins with a continuation word:
  "and", "but", "so", "or", "yet", "like", "for", "that", "however",
  "therefore", "then", "now", "also", "still", "even", "well", "you know".
- REJECT any moment that opens mid-sentence, mid-argument, or mid-comparison.
- If no moment in a section satisfies these rules, skip that section entirely.

REASONING PROCESS (MANDATORY — think before selecting):
For each candidate moment, FIRST write a "rationale" explaining:
1. WHY this moment is compelling (hook, emotion, completeness)
2. Whether it opens AND closes a complete thought
3. Whether a cold viewer (no prior context) would understand it
Only THEN assign timestamps. If your rationale reveals a problem, do NOT include the moment.

EXAMPLES:
GOOD moment:
  {{"rationale": "Opens with a direct rhetorical question that hooks immediately. The speaker poses the question, delivers a hadith with full attribution, and concludes with the lesson — complete narrative arc in ~30 seconds. No prior context needed.",
  "approximate_start": 120.5, "approximate_end": 152.0,
  "opening_line": "What is the one deed that the Prophet valued above all others?"}}

BAD moment (do NOT nominate moments like this):
  {{"rationale": "Emotionally strong but opens mid-argument with 'And that is why' — presupposes the preceding reasoning. Viewer would be confused.",
  "approximate_start": 45.0, "approximate_end": 72.0,
  "opening_line": "And that is why we must reflect on this"}}

OUTPUT FORMAT:
- Return ONLY valid JSON — no markdown, no prose outside the JSON.
- Return a JSON object with a single key "moments" whose value is an array.
- Each array element must have fields in this order: rationale, approximate_start,
  approximate_end, opening_line.
- Example: {{"moments": [{{"rationale": "...", "approximate_start": 10.5,
  "approximate_end": 45.2, "opening_line": "The Prophet said..."}}]}}
- Moments MUST NOT overlap.
"""

    @staticmethod
    def _parse_nominations(raw: str) -> list[NominatedMoment]:
        """Parse the LLM's JSON response into NominatedMoment objects."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse nomination response: %s\nRaw: %s", e, raw[:500])
            return []

        # Unwrap {"moments": [...]} wrapper (used when response_format=json_object)
        if isinstance(data, dict):
            data = data.get("moments", [])
            if not isinstance(data, list):
                data = [data]

        nominations: list[NominatedMoment] = []
        for item in data:
            try:
                nominations.append(NominatedMoment(
                    approximate_start=float(item["approximate_start"]),
                    approximate_end=float(item["approximate_end"]),
                    rationale=str(item.get("rationale", "")),
                    opening_line=str(item.get("opening_line", "")),
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("Skipping malformed nomination: %s (%s)", item, e)

        return nominations

    def _snap_to_segments(
        self,
        nominations: list[NominatedMoment],
        transcript: Transcript,
    ) -> list[Segment]:
        """
        Snap approximate LLM timestamps to exact TranscriptSegment boundaries.

        For each nomination, find the TranscriptSegments that best cover the
        nominated time range, then extend to complete the final sentence if it
        was cut mid-thought.
        """
        ts_segs = transcript.segments
        if not ts_segs:
            return []

        # Build an index for fast next-segment lookup
        ts_by_id: dict[int, int] = {s.id: i for i, s in enumerate(ts_segs)}

        segments: list[Segment] = []
        for seg_id, nom in enumerate(nominations):
            # Find TranscriptSegments that overlap with the nominated range
            matching = self._find_overlapping_ts_segments(
                ts_segs, nom.approximate_start, nom.approximate_end
            )

            if not matching:
                logger.warning(
                    "No transcript segments found for nomination %.1f–%.1f, skipping",
                    nom.approximate_start, nom.approximate_end,
                )
                continue

            # Extend to complete the final sentence if it ends mid-thought
            matching = self._extend_to_sentence_boundary(
                matching, ts_segs, ts_by_id, self.cfg.max_clip_duration
            )

            # Build segment from the matching TranscriptSegments
            segment = Segment(
                segment_id=seg_id,
                start=matching[0].start,
                end=matching[-1].end,
                text=" ".join(s.text.strip() for s in matching),
                transcript_segment_ids=[s.id for s in matching],
            )

            # Enforce duration constraints
            if (self.cfg.min_clip_duration <= segment.duration <= self.cfg.max_clip_duration):
                nom.snapped_segment = segment
                segments.append(segment)
            else:
                logger.debug(
                    "Nomination %.1f–%.1f snapped to %.1fs, outside [%.0f, %.0f], skipping",
                    nom.approximate_start, nom.approximate_end,
                    segment.duration,
                    self.cfg.min_clip_duration, self.cfg.max_clip_duration,
                )

        return segments

    @staticmethod
    def _find_overlapping_ts_segments(
        ts_segs: list[TranscriptSegment],
        start: float,
        end: float,
    ) -> list[TranscriptSegment]:
        """Find TranscriptSegments that overlap with [start, end].

        Uses inclusive end boundary (<=) so a segment starting exactly at the
        nominated end time is included — the LLM sees "[Xs-Ys]" markers and
        often places approximate_end right on a segment boundary.
        """
        matching = []
        for seg in ts_segs:
            if seg.start <= end and seg.end >= start:
                matching.append(seg)
        return matching

    @staticmethod
    def _extend_to_sentence_boundary(
        matching: list[TranscriptSegment],
        all_segs: list[TranscriptSegment],
        ts_by_id: dict[int, int],
        max_duration: float,
    ) -> list[TranscriptSegment]:
        """Extend matching segments to include the next segment(s) if the last
        one ends mid-sentence (no sentence-ending punctuation).

        This prevents clips from cutting off thoughts like:
          "He will receive the same reward" (clip ends)
          "the fasting person receives without decreasing." (lost)
        """
        sentence_enders = {'.', '!', '?'}
        start_time = matching[0].start

        while True:
            last_text = matching[-1].text.strip()
            # If the last segment ends with sentence-ending punctuation, stop
            if last_text and last_text[-1] in sentence_enders:
                break

            # Find the next TranscriptSegment
            last_idx = ts_by_id.get(matching[-1].id)
            if last_idx is None or last_idx + 1 >= len(all_segs):
                break

            next_seg = all_segs[last_idx + 1]

            # Don't extend beyond max clip duration
            if next_seg.end - start_time > max_duration:
                break

            matching.append(next_seg)

            # Safety: at most 3 extensions to avoid runaway expansion
            if len(matching) > len(ts_by_id):
                break

        return matching
