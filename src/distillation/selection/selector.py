"""
Constraint-based clip selector.

Problem: given N scored segments, select K non-overlapping clips of duration
[min, max] that together maximise aggregate engagement score.

This is a variant of the weighted interval scheduling problem.  We use a
greedy approach (sort by composite score, take greedily with overlap check)
which is optimal when segment quality is well-calibrated and the overlapping
sliding-window structure is used.  For non-overlapping semantic segments (the
normal case) it degenerates to simple top-K selection.

Domain-specific weighting:
  We re-weight the composite score per domain before sorting.  For example,
  khutba clips up-weight domain_integrity (incomplete rulings are harmful)
  and standalone_coherence (context from the khutba body is unavailable to
  a TikTok viewer).

  Future: these weight vectors should be learnable from human feedback.
"""

from __future__ import annotations

import logging
import re

from distillation.config import Config
from distillation.models import Clip, Domain, EngagementAxes, Segment, ScoredSegment

logger = logging.getLogger(__name__)

# Weight for acoustic_energy in the composite score (speaker emphasis signal).
ACOUSTIC_ENERGY_WEIGHT = 0.05

# Per-domain scoring weight overrides.
# Keys must match EngagementAxes field names.
DOMAIN_WEIGHTS: dict[Domain, dict[str, float]] = {
    Domain.KHUTBA: {
        "semantic_density": 0.10,
        "emotional_resonance": 0.15,    # reduced: emotion ≠ completeness
        "standalone_coherence": 0.25,   # increased: cold viewer comprehension
        "narrative_completeness": 0.22, # increased: open+close a thought
        "domain_integrity": 0.15,       # unchanged
        "hook_strength": 0.13,          # minor reduction
    },
    Domain.PODCAST: {
        "semantic_density": 0.20,
        "emotional_resonance": 0.15,
        "standalone_coherence": 0.25,
        "narrative_completeness": 0.15,
        "domain_integrity": 0.05,
        "hook_strength": 0.20,         # up-weighted: hook is king on TikTok
    },
    Domain.GENERIC: {
        "semantic_density": 0.20,
        "emotional_resonance": 0.20,
        "standalone_coherence": 0.20,
        "narrative_completeness": 0.15,
        "domain_integrity": 0.15,
        "hook_strength": 0.10,
    },
}


class ClipSelector:
    """
    Selects the top-K clips from a list of ScoredSegments subject to:
      1. Duration constraints [min_clip_duration, max_clip_duration].
      2. Non-overlap (no two clips share audio time).
      3. Target count (target_clip_count).
    """

    def __init__(self, config: Config) -> None:
        self.cfg = config

    def select(
        self,
        scored: list[ScoredSegment],
        domain: Domain = Domain.GENERIC,
    ) -> list[Clip]:
        """
        Select top clips from scored segments.

        Returns Clips ordered by source timeline (not score rank), which
        makes the output package coherent for playlist-style consumption.
        """
        # ── 1. Filter by duration constraints and completeness gate ──────────
        weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS[Domain.GENERIC])
        min_nc = self.cfg.min_narrative_completeness
        candidates = []
        duration_ok = []
        for ss in scored:
            in_duration = (
                self.cfg.min_clip_duration <= ss.segment.duration <= self.cfg.max_clip_duration
            )
            if not in_duration:
                logger.debug(
                    "Duration gate excluded segment %d (%.1fs, %.0f–%.0fs window): %r",
                    ss.segment.segment_id,
                    ss.segment.duration,
                    self.cfg.min_clip_duration,
                    self.cfg.max_clip_duration,
                    ss.segment.text[:60],
                )
                continue
            duration_ok.append(ss)
            if ss.scores.narrative_completeness >= min_nc:
                candidates.append(ss)

        # Log how many were filtered by completeness gate
        completeness_filtered = len(duration_ok) - len(candidates)
        if completeness_filtered > 0:
            logger.info(
                "Completeness gate (%.2f) filtered out %d segments",
                min_nc, completeness_filtered,
            )

        if len(candidates) < self.cfg.target_clip_count:
            logger.warning(
                "Only %d segments meet duration constraints [%.0f, %.0f]s; "
                "target was %d.  Consider relaxing constraints.",
                len(candidates),
                self.cfg.min_clip_duration,
                self.cfg.max_clip_duration,
                self.cfg.target_clip_count,
            )

        # ── 1b. Generate adjacent-segment merge candidates ────────────────────
        merge_candidates = self._generate_merge_candidates(candidates)
        if merge_candidates:
            logger.info("Generated %d merge candidates from adjacent segments", len(merge_candidates))
            candidates = candidates + merge_candidates

        # ── 2. Rank by domain-weighted composite score ─────────────────────────
        ranked = sorted(candidates, key=lambda ss: self._weighted_score(ss, weights), reverse=True)

        # ── 3. Greedy non-overlap selection ────────────────────────────────────
        selected: list[ScoredSegment] = []
        for ss in ranked:
            if len(selected) >= self.cfg.target_clip_count:
                break
            if not self._overlaps_any(ss, selected):
                selected.append(ss)

        # ── 4. Sort by timeline order ──────────────────────────────────────────
        selected.sort(key=lambda ss: ss.segment.start)

        # ── 5. Build Clip objects ──────────────────────────────────────────────
        clips: list[Clip] = []
        for i, ss in enumerate(selected):
            clips.append(Clip(
                clip_id=i,
                source_path=ss.segment.text,   # will be overwritten by pipeline
                start=ss.segment.start,
                end=ss.segment.end,
                scored_segment=ss,
            ))

        logger.info("Selected %d clips", len(clips))
        return clips

    @classmethod
    def _weighted_score(cls, ss: ScoredSegment, weights: dict[str, float]) -> float:
        s = ss.scores

        # Apply energy_onset_ratio as a multiplier on hook_strength:
        # ratio > 1 means speaker starts strong → boost hook score
        # ratio < 1 means weak opening (silence/throat-clear) → penalise
        # Clamp multiplier to [0.7, 1.3] to avoid extreme swings
        onset_multiplier = max(0.7, min(1.3, ss.energy_onset_ratio))
        effective_hook = s.hook_strength * onset_multiplier

        # Structural completeness: penalise segments that start mid-sentence
        # or end without terminal punctuation
        structural_mult = cls._structural_completeness_penalty(ss)
        effective_narrative = s.narrative_completeness * structural_mult

        # Referential opener: cap standalone_coherence and narrative_completeness
        # for segments whose first words presuppose unheard context.
        sc_cap, nc_cap = cls._referential_opener_penalty(ss.segment.text)
        effective_standalone = min(s.standalone_coherence, sc_cap)
        effective_narrative = min(effective_narrative, nc_cap)

        total_weight = sum(weights.values()) + ACOUSTIC_ENERGY_WEIGHT
        return (
            s.semantic_density * weights.get("semantic_density", 0)
            + s.emotional_resonance * weights.get("emotional_resonance", 0)
            + effective_standalone * weights.get("standalone_coherence", 0)
            + effective_narrative * weights.get("narrative_completeness", 0)
            + s.domain_integrity * weights.get("domain_integrity", 0)
            + effective_hook * weights.get("hook_strength", 0)
            + ss.acoustic_energy * ACOUSTIC_ENERGY_WEIGHT
        ) / total_weight

    # Words that definitively cannot open an independent thought when lowercase.
    # Their presence at position 0 (lowercase) means the segment is mid-sentence.
    _CONTINUATION_STARTERS = frozenset({
        "and", "but", "or", "so", "yet", "for", "nor",
        "however", "therefore", "thus", "hence", "because", "since",
        "immensely", "completely", "actually", "basically", "essentially",
        "furthermore", "moreover", "additionally", "consequently",
        "like",  # "like all the prophets" = mid-comparison
    })

    @classmethod
    def _structural_completeness_penalty(cls, ss: ScoredSegment) -> float:
        """
        Returns a multiplier in [0.7, 1.0] that penalises structurally
        incomplete segments:
          - Starts mid-sentence (first char is lowercase, no sentence boundary)
          - Ends without terminal punctuation (., !, ?)

        Applied as a multiplier on the narrative_completeness component.

        Continuation-word starters (and, but, immensely, etc.) receive a
        stronger penalty (-0.25) since they definitively indicate a mid-sentence
        clip regardless of capitalisation.
        """
        text = ss.segment.text.strip()
        if not text:
            return 1.0

        penalty = 1.0

        # Penalise mid-sentence starts: first non-whitespace char is lowercase
        # (but allow quotes, Arabic, digits — only penalise clear lowercase Latin)
        first_char = text[0]
        if first_char.isalpha() and first_char.islower():
            first_word = text.split()[0].lower().strip(".,!?")
            if first_word in cls._CONTINUATION_STARTERS:
                # Strong penalty: continuation word guarantees mid-sentence position
                penalty -= 0.25
            else:
                penalty -= 0.15

        # Penalise missing terminal punctuation
        if not re.search(r'[.!?"\u06D4]$', text):  # \u06D4 = Arabic full stop
            penalty -= 0.15

        return max(0.7, penalty)

    @staticmethod
    def _referential_opener_penalty(text: str) -> tuple[float, float]:
        """
        Returns cap values (sc_cap, nc_cap) for standalone_coherence and
        narrative_completeness when the segment opens with a referential phrase
        that presupposes unheard context.

        If a referential pattern is detected in the first 20 words, both caps
        are set to 0.5 — the LLM score cannot exceed this regardless of what it
        returned.  Returns (1.0, 1.0) (no cap) when no pattern is found.
        """
        opening = " ".join(text.split()[:20]).lower()
        referential_patterns = [
            # Explicit audience-knowledge references
            r"\bas you( all)? know\b",
            r"\bwe all know\b",
            # Simple present referential ("as i mentioned", "as we said")
            r"\bas (i|we) (mentioned|said|discussed|noted|explained)\b",
            r"\blike (i|we) said\b",
            # Past-perfect referential ("we had mentioned", "he had said")
            # — previously missing, confirmed bug in clip_000
            r"\b(we|i|he|she|they)\s+had\s+(mentioned|said|discussed|noted|talked|covered)\b",
            # Passive past-perfect ("as was mentioned", "as had been discussed")
            r"\bas (was |had been )?(mentioned|said|discussed|noted|explained)\b",
            # Continuation markers at start of clip
            r"\band so as\b",
            r"\bcontinuing (from|with)\b",
            r"\bgoing back to\b",
            # Callback to prior story or event
            r"\bthe story (we|i) (mentioned|told|discussed)\b",
            r"\bremember (when|what|how)\b",
            # Continuation conjunction openers (mid-sentence guaranteed)
            # Catches: "and so", "but he cannot", "or the other", etc.
            r"^(and|but|or|so|yet|nor|however|therefore|thus|hence|immensely)\b",
            # Mid-comparison openers: "like all the prophets", "like every scholar"
            # — presuppose a subject already established
            r"^like (all|every|most|the|these|those|other) (the )?\b",
        ]
        for pattern in referential_patterns:
            if re.search(pattern, opening):
                logger.debug(
                    "Referential opener detected in segment — capping sc/nc at 0.5. "
                    "Opening: %r", opening[:80],
                )
                return 0.5, 0.5
        return 1.0, 1.0

    def _generate_merge_candidates(
        self, scored: list[ScoredSegment],
    ) -> list[ScoredSegment]:
        """
        Generate merged [N, N+1] candidates where:
          - Combined duration <= max_clip_duration
          - First segment has narrative_completeness < 0.5 (likely incomplete)

        Scores are interpolated from both segments with a completeness bonus,
        since merging is expected to improve narrative arc.
        """
        # Sort by timeline to find true adjacencies
        by_time = sorted(scored, key=lambda ss: ss.segment.start)
        merged: list[ScoredSegment] = []

        for i in range(len(by_time) - 1):
            a, b = by_time[i], by_time[i + 1]
            combined_duration = b.segment.end - a.segment.start

            if combined_duration > self.cfg.max_clip_duration:
                continue
            if combined_duration < self.cfg.min_clip_duration:
                continue
            if a.scores.narrative_completeness >= 0.5 and a.scores.standalone_coherence >= 0.5:
                continue

            # Build merged segment
            merged_segment = Segment(
                segment_id=-(i + 1),  # negative IDs for synthetic segments
                start=a.segment.start,
                end=b.segment.end,
                text=a.segment.text + " " + b.segment.text,
                transcript_segment_ids=(
                    a.segment.transcript_segment_ids + b.segment.transcript_segment_ids
                ),
            )

            # Interpolate scores: weighted average biased toward the better segment,
            # plus a per-axis completeness bonus for whichever axis triggered the merge
            sa, sb = a.scores, b.scores
            nc_was_weak = a.scores.narrative_completeness < 0.5
            sc_was_weak = a.scores.standalone_coherence < 0.5
            merged_axes = EngagementAxes(
                semantic_density=(sa.semantic_density + sb.semantic_density) / 2,
                emotional_resonance=max(sa.emotional_resonance, sb.emotional_resonance),
                standalone_coherence=min(
                    1.0,
                    (sa.standalone_coherence + sb.standalone_coherence) / 2
                    + (0.10 if sc_was_weak else 0.0),
                ),
                narrative_completeness=min(
                    1.0,
                    (sa.narrative_completeness + sb.narrative_completeness) / 2
                    + (0.10 if nc_was_weak else 0.0),
                ),
                domain_integrity=min(sa.domain_integrity, sb.domain_integrity),
                hook_strength=sa.hook_strength,  # hook comes from the first segment
                llm_rationale=f"Merged segments {a.segment.segment_id}+{b.segment.segment_id}",
            )

            merged_scored = ScoredSegment(
                segment=merged_segment,
                scores=merged_axes,
                acoustic_energy=(a.acoustic_energy + b.acoustic_energy) / 2,
                energy_onset_ratio=a.energy_onset_ratio,  # onset from first segment
                speech_rate_wpm=(a.speech_rate_wpm + b.speech_rate_wpm) / 2,
            )
            merged.append(merged_scored)

        return merged

    @staticmethod
    def _overlaps_any(candidate: ScoredSegment, selected: list[ScoredSegment]) -> bool:
        c_start = candidate.segment.start
        c_end = candidate.segment.end
        for ss in selected:
            # Segments overlap if one starts before the other ends
            if c_start < ss.segment.end and c_end > ss.segment.start:
                return True
        return False
