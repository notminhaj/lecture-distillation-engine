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

from distillation.config import Config
from distillation.models import Clip, Domain, ScoredSegment

logger = logging.getLogger(__name__)

# Per-domain scoring weight overrides.
# Keys must match EngagementAxes field names.
DOMAIN_WEIGHTS: dict[Domain, dict[str, float]] = {
    Domain.KHUTBA: {
        "semantic_density": 0.15,
        "emotional_resonance": 0.20,
        "standalone_coherence": 0.25,  # up-weighted: viewer has no context
        "narrative_completeness": 0.15,
        "domain_integrity": 0.20,      # up-weighted: theological accuracy matters
        "hook_strength": 0.05,
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
        # ── 1. Filter by duration constraints ─────────────────────────────────
        weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS[Domain.GENERIC])
        candidates = [
            ss for ss in scored
            if self.cfg.min_clip_duration <= ss.segment.duration <= self.cfg.max_clip_duration
        ]

        if len(candidates) < self.cfg.target_clip_count:
            logger.warning(
                "Only %d segments meet duration constraints [%.0f, %.0f]s; "
                "target was %d.  Consider relaxing constraints.",
                len(candidates),
                self.cfg.min_clip_duration,
                self.cfg.max_clip_duration,
                self.cfg.target_clip_count,
            )

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

    @staticmethod
    def _weighted_score(ss: ScoredSegment, weights: dict[str, float]) -> float:
        s = ss.scores
        total_weight = sum(weights.values())
        return (
            s.semantic_density * weights.get("semantic_density", 0)
            + s.emotional_resonance * weights.get("emotional_resonance", 0)
            + s.standalone_coherence * weights.get("standalone_coherence", 0)
            + s.narrative_completeness * weights.get("narrative_completeness", 0)
            + s.domain_integrity * weights.get("domain_integrity", 0)
            + s.hook_strength * weights.get("hook_strength", 0)
        ) / total_weight

    @staticmethod
    def _overlaps_any(candidate: ScoredSegment, selected: list[ScoredSegment]) -> bool:
        c_start = candidate.segment.start
        c_end = candidate.segment.end
        for ss in selected:
            # Segments overlap if one starts before the other ends
            if c_start < ss.segment.end and c_end > ss.segment.start:
                return True
        return False
