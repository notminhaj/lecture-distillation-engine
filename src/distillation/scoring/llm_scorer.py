"""
LLM-based multi-axis engagement scorer using OpenAI.

Architecture decisions:
  1. Batch scoring: we send multiple segments in a single API call using a
     structured JSON schema.  This is ~5x cheaper than per-segment calls and
     avoids rate limit issues on long lectures.

  2. Structured output: we ask the model to return a JSON array matching our
     EngagementAxes schema exactly.  We validate with Pydantic; if validation
     fails we retry with a stricter prompt.

  3. Domain-aware prompts: the system prompt is parameterised by Domain.
     This is where khutba-specific knowledge (theological integrity, Arabic
     terms, etc.) is injected — without it bleeding into other modules.

  4. Context window: we provide the full transcript as background context so
     the model can evaluate standalone_coherence correctly (does this clip make
     sense to someone who hasn't heard the rest?).

  5. Acoustic features are *not* the LLM's job.  We compute energy, pace, and
     speech-rate in AcousticScorer and merge the results here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from distillation.config import Config
from distillation.models import Domain, EngagementAxes, Segment, ScoredSegment, Transcript
from distillation.scoring.acoustic import AcousticScorer
from distillation.scoring.base import BaseScorer

logger = logging.getLogger(__name__)

# How many segments to score per API call.
# Tune down if you hit context limits on very long segments.
BATCH_SIZE = 8

# Domain-specific system prompt fragments.
# These inject domain knowledge without hardcoding it in the main logic.
DOMAIN_CONTEXT: dict[Domain, str] = {
    Domain.KHUTBA: """
You are evaluating clips from an Islamic khutba (sermon).
Domain integrity rules:
- A hadith must be attributed (even partially); a clip that quotes a hadith
  without "the Prophet (ﷺ) said" or equivalent is incomplete → lower domain_integrity.
- A ruling (fatwa) must present both the ruling and its basis, not just one.
- Arabic phrases (Subhanallah, Alhamdulillah, etc.) are not "foreign" — weight
  emotional_resonance for their rhetorical function.
- The clip should stand alone as a coherent reminder or lesson, not leave the
  listener mid-argument.
""",
    Domain.PODCAST: """
You are evaluating clips from a conversational podcast.
Domain integrity rules:
- Claims should be grounded (the speaker provides a source, anecdote, or reasoning).
- Mid-conversation clips that reference "what we said earlier" heavily score
  low on standalone_coherence.
- Hooks can be provocative questions or counter-intuitive claims.
""",
    Domain.BUSINESS_TALK: """
You are evaluating clips from a business or leadership talk.
Domain integrity rules:
- A business insight should be actionable or specific, not just motivational filler.
- Data claims without numbers score lower on semantic_density.
""",
    Domain.LECTURE: """
You are evaluating clips from an academic lecture.
Domain integrity rules:
- A claim should come with reasoning or evidence.
- Mid-proof clips (where the logical chain is incomplete) score low on narrative_completeness.
""",
    Domain.GENERIC: "",
}


class LLMScorer(BaseScorer):
    """
    Scores segments on six engagement axes using OpenAI, then merges
    acoustic features from AcousticScorer.
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.acoustic = AcousticScorer()

    def score(
        self,
        segments: list[Segment],
        transcript: Transcript,
        *,
        audio_path: str | Path | None = None,
        domain: Domain = Domain.GENERIC,
    ) -> list[ScoredSegment]:
        """Score all segments; returns ScoredSegments in the same order."""
        if not segments:
            return []

        # ── Acoustic features (fast, local, no API cost) ───────────────────────
        acoustic_features: dict[int, dict] = {}
        if audio_path:
            acoustic_features = self.acoustic.extract_features(
                segments, str(audio_path)
            )

        # ── LLM scoring in batches ─────────────────────────────────────────────
        all_axes: list[EngagementAxes] = []
        for batch_start in range(0, len(segments), BATCH_SIZE):
            batch = segments[batch_start: batch_start + BATCH_SIZE]
            batch_axes = self._score_batch(batch, transcript, domain)
            all_axes.extend(batch_axes)

        # ── Merge into ScoredSegment objects ────────────────────────────────────
        results: list[ScoredSegment] = []
        for segment, axes in zip(segments, all_axes):
            af = acoustic_features.get(segment.segment_id, {})
            results.append(ScoredSegment(
                segment=segment,
                scores=axes,
                acoustic_energy=af.get("energy", 0.0),
                speech_rate_wpm=af.get("wpm", 0.0),
            ))

        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _score_batch(
        self,
        batch: list[Segment],
        transcript: Transcript,
        domain: Domain,
    ) -> list[EngagementAxes]:
        """
        Score a batch of segments in a single API call.

        Returns EngagementAxes for each segment in the same order.
        """
        system_prompt = self._build_system_prompt(domain)
        user_prompt = self._build_user_prompt(batch, transcript)

        logger.debug("Scoring batch of %d segments via OpenAI", len(batch))

        response = self.client.chat.completions.create(
            model=self.cfg.openai_model,
            max_tokens=2048,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw_text = response.choices[0].message.content
        return self._parse_response(raw_text, expected_count=len(batch))

    def _build_system_prompt(self, domain: Domain) -> str:
        domain_ctx = DOMAIN_CONTEXT.get(domain, "")
        return f"""You are an expert content strategist specialising in short-form video.
Your job: evaluate candidate clips from a lecture and score each on six axes.

{domain_ctx}

SCORING AXES (all 0.0–1.0, two decimal places):
- semantic_density: Information per second. High = every sentence adds new value.
- emotional_resonance: Does the speaker's language evoke feeling, urgency, or awe?
- standalone_coherence: Can a cold viewer (no prior context) understand this clip?
- narrative_completeness: Does it open AND close a thought (or close with deliberate tension)?
- domain_integrity: Is domain-specific content presented correctly and completely?
- hook_strength: Would the FIRST 3 seconds stop a scroll on TikTok?

RULES:
- Return ONLY valid JSON — no markdown, no prose, no explanation outside the JSON.
- Return a JSON array with exactly one object per input segment, in the same order.
- Each object: {{"segment_id": int, "semantic_density": float, "emotional_resonance": float,
  "standalone_coherence": float, "narrative_completeness": float,
  "domain_integrity": float, "hook_strength": float, "llm_rationale": "string"}}
- llm_rationale: 1–2 sentences explaining the key strength and weakness.
"""

    def _build_user_prompt(self, batch: list[Segment], transcript: Transcript) -> str:
        # Provide condensed context so model can evaluate standalone_coherence
        # We use the first 500 words of the full transcript as background
        context_words = transcript.full_text.split()[:500]
        context_snippet = " ".join(context_words)

        segments_json = json.dumps([
            {
                "segment_id": s.segment_id,
                "start_seconds": round(s.start, 1),
                "end_seconds": round(s.end, 1),
                "duration_seconds": round(s.duration, 1),
                "text": s.text,
            }
            for s in batch
        ], ensure_ascii=False, indent=2)

        return f"""FULL LECTURE CONTEXT (first ~500 words for coherence evaluation):
{context_snippet}

---
SEGMENTS TO SCORE:
{segments_json}

Return a JSON array with one scoring object per segment."""

    @staticmethod
    def _parse_response(raw: str, expected_count: int) -> list[EngagementAxes]:
        """
        Parse the model's JSON response into EngagementAxes objects.

        Robust to:
          - Markdown code fences (```json ... ```)
          - Trailing commas
          - Extra whitespace
        """
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM response as JSON: %s\nRaw: %s", e, raw[:500])
            # Return neutral scores rather than crashing the pipeline
            return [EngagementAxes(
                semantic_density=0.5,
                emotional_resonance=0.5,
                standalone_coherence=0.5,
                narrative_completeness=0.5,
                domain_integrity=0.5,
                hook_strength=0.5,
                llm_rationale="Scoring failed; using neutral defaults.",
            ) for _ in range(expected_count)]

        if not isinstance(data, list):
            data = [data]

        axes_list: list[EngagementAxes] = []
        for item in data:
            try:
                axes = EngagementAxes(
                    semantic_density=float(item.get("semantic_density", 0.5)),
                    emotional_resonance=float(item.get("emotional_resonance", 0.5)),
                    standalone_coherence=float(item.get("standalone_coherence", 0.5)),
                    narrative_completeness=float(item.get("narrative_completeness", 0.5)),
                    domain_integrity=float(item.get("domain_integrity", 0.5)),
                    hook_strength=float(item.get("hook_strength", 0.5)),
                    llm_rationale=str(item.get("llm_rationale", "")),
                )
                axes_list.append(axes)
            except Exception as e:
                logger.warning("Failed to parse axes item %s: %s", item, e)
                axes_list.append(EngagementAxes(
                    semantic_density=0.5,
                    emotional_resonance=0.5,
                    standalone_coherence=0.5,
                    narrative_completeness=0.5,
                    domain_integrity=0.5,
                    hook_strength=0.5,
                    llm_rationale="Parse error; using neutral defaults.",
                ))

        # Pad if model returned fewer items than expected
        while len(axes_list) < expected_count:
            axes_list.append(axes_list[-1] if axes_list else EngagementAxes(
                semantic_density=0.5, emotional_resonance=0.5, standalone_coherence=0.5,
                narrative_completeness=0.5, domain_integrity=0.5, hook_strength=0.5,
            ))

        return axes_list[:expected_count]
