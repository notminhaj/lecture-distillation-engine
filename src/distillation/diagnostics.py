"""
Diagnostic reporter for full pipeline visibility.

Collects intermediate LLM decisions at every stage and writes a comprehensive
JSON report so you can trace exactly why each clip was nominated, scored,
selected, or rejected.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from datetime import datetime, timezone

from distillation.models import (
    Clip,
    Domain,
    NominatedMoment,
    Segment,
    ScoredSegment,
)
from distillation.selection.selector import ClipSelector, DOMAIN_WEIGHTS, ACOUSTIC_ENERGY_WEIGHT

logger = logging.getLogger(__name__)


class DiagnosticReporter:
    """Builds a full diagnostic report from pipeline intermediates."""

    @classmethod
    def build_report(
        cls,
        *,
        domain: Domain,
        nominations: list[NominatedMoment],
        segments: list[Segment],
        scored_segments: list[ScoredSegment],
        raw_scorer_responses: list[str],
        selected_clips: list[Clip],
        config_snapshot: dict,
    ) -> dict:
        weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS[Domain.GENERIC])

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "domain": domain.value,
            "config": config_snapshot,
            "summary": cls._build_summary(nominations, segments, scored_segments, selected_clips),
            "stage_1_nominations": cls._build_nomination_section(nominations, segments),
            "stage_2_scoring": cls._build_scoring_section(scored_segments, raw_scorer_responses, weights, domain),
            "stage_3_selection": cls._build_selection_section(scored_segments, selected_clips, weights, domain),
        }
        return report

    @classmethod
    def _build_summary(cls, nominations, segments, scored_segments, clips) -> dict:
        return {
            "moments_nominated": len(nominations),
            "segments_after_snap": len(segments),
            "segments_scored": len(scored_segments),
            "clips_selected": len(clips),
            "selected_clip_ids": [c.clip_id for c in clips],
        }

    @classmethod
    def _build_nomination_section(cls, nominations: list[NominatedMoment], segments: list[Segment]) -> list[dict]:
        items = []
        for i, nom in enumerate(nominations):
            entry = {
                "nomination_index": i,
                "approximate_start": nom.approximate_start,
                "approximate_end": nom.approximate_end,
                "approximate_duration": round(nom.approximate_duration, 1),
                "opening_line": nom.opening_line,
                "rationale": nom.rationale,
                "snapped": nom.snapped_segment is not None,
            }
            if nom.snapped_segment:
                seg = nom.snapped_segment
                entry["snapped_segment"] = {
                    "segment_id": seg.segment_id,
                    "start": seg.start,
                    "end": seg.end,
                    "duration": round(seg.duration, 1),
                    "first_20_words": " ".join(seg.text.split()[:20]),
                    "last_20_words": " ".join(seg.text.split()[-20:]),
                }
            items.append(entry)
        return items

    @classmethod
    def _build_scoring_section(
        cls,
        scored: list[ScoredSegment],
        raw_responses: list[str],
        weights: dict[str, float],
        domain: Domain,
    ) -> dict:
        return {
            "raw_llm_responses": [
                {"batch_index": i, "response_preview": r[:2000] if r else "<empty>"}
                for i, r in enumerate(raw_responses)
            ],
            "per_segment": [
                cls._score_detail(ss, weights, domain)
                for ss in scored
            ],
        }

    @classmethod
    def _score_detail(cls, ss: ScoredSegment, weights: dict[str, float], domain: Domain) -> dict:
        s = ss.scores
        text = ss.segment.text

        # Reproduce selector penalty analysis
        sc_cap, nc_cap = ClipSelector._referential_opener_penalty(text)
        struct_mult = ClipSelector._structural_completeness_penalty(ss)

        onset_mult = max(0.7, min(1.3, ss.energy_onset_ratio))
        effective_hook = s.hook_strength * onset_mult
        effective_narrative = s.narrative_completeness * struct_mult
        effective_standalone = min(s.standalone_coherence, sc_cap)
        effective_narrative = min(effective_narrative, nc_cap)

        total_weight = sum(weights.values()) + ACOUSTIC_ENERGY_WEIGHT
        composite = (
            s.semantic_density * weights.get("semantic_density", 0)
            + s.emotional_resonance * weights.get("emotional_resonance", 0)
            + effective_standalone * weights.get("standalone_coherence", 0)
            + effective_narrative * weights.get("narrative_completeness", 0)
            + s.domain_integrity * weights.get("domain_integrity", 0)
            + effective_hook * weights.get("hook_strength", 0)
            + ss.acoustic_energy * ACOUSTIC_ENERGY_WEIGHT
        ) / total_weight

        # Detect issues
        issues = []
        first_word = text.split()[0] if text.strip() else ""
        if first_word and first_word[0].islower():
            issues.append(f"STARTS_LOWERCASE: '{first_word}' — mid-sentence opener")
        if sc_cap < 1.0:
            issues.append(f"REFERENTIAL_OPENER_DETECTED: sc/nc capped at {sc_cap}")
        if struct_mult < 1.0:
            issues.append(f"STRUCTURAL_PENALTY: multiplier={struct_mult:.2f}")
        if not re.search(r'[.!?"\u06D4]$', text.strip()):
            issues.append("NO_TERMINAL_PUNCTUATION: clip ends mid-sentence")
        if s.standalone_coherence >= 0.7 and sc_cap < 1.0:
            issues.append(
                f"LLM_OVERSCORED_COHERENCE: LLM gave {s.standalone_coherence}, "
                f"but referential opener detected → capped to {effective_standalone}"
            )
        if s.narrative_completeness >= 0.7 and (struct_mult < 1.0 or nc_cap < 1.0):
            issues.append(
                f"LLM_OVERSCORED_NARRATIVE: LLM gave {s.narrative_completeness}, "
                f"effective={effective_narrative:.2f} after penalties"
            )

        return {
            "segment_id": ss.segment.segment_id,
            "duration": round(ss.segment.duration, 1),
            "first_20_words": " ".join(text.split()[:20]),
            "last_20_words": " ".join(text.split()[-20:]),
            "llm_scores": {
                "semantic_density": s.semantic_density,
                "emotional_resonance": s.emotional_resonance,
                "standalone_coherence": s.standalone_coherence,
                "narrative_completeness": s.narrative_completeness,
                "domain_integrity": s.domain_integrity,
                "hook_strength": s.hook_strength,
            },
            "llm_reasoning": s.llm_rationale,
            "penalties_applied": {
                "referential_opener_cap": {"sc_cap": sc_cap, "nc_cap": nc_cap},
                "structural_multiplier": struct_mult,
                "onset_multiplier": onset_mult,
            },
            "effective_scores": {
                "standalone_coherence": round(effective_standalone, 3),
                "narrative_completeness": round(effective_narrative, 3),
                "hook_strength": round(effective_hook, 3),
            },
            "composite_score": round(composite, 4),
            "acoustic": {
                "energy": round(ss.acoustic_energy, 4),
                "onset_ratio": round(ss.energy_onset_ratio, 4),
                "speech_rate_wpm": round(ss.speech_rate_wpm, 1),
            },
            "issues": issues,
        }

    @classmethod
    def _build_selection_section(
        cls,
        scored: list[ScoredSegment],
        clips: list[Clip],
        weights: dict[str, float],
        domain: Domain,
    ) -> dict:
        # Rank all segments by composite score (same logic as selector)
        ranked = []
        for ss in scored:
            composite = ClipSelector._weighted_score(ss, weights)
            ranked.append((ss, composite))
        ranked.sort(key=lambda x: x[1], reverse=True)

        selected_seg_ids = {c.scored_segment.segment.segment_id for c in clips}

        ranking = []
        for rank, (ss, composite) in enumerate(ranked, 1):
            seg_id = ss.segment.segment_id
            status = "SELECTED" if seg_id in selected_seg_ids else "REJECTED"
            entry = {
                "rank": rank,
                "segment_id": seg_id,
                "composite_score": round(composite, 4),
                "status": status,
                "duration": round(ss.segment.duration, 1),
                "first_10_words": " ".join(ss.segment.text.split()[:10]),
            }
            if status == "REJECTED":
                # Explain why
                reasons = []
                dur = ss.segment.duration
                from distillation.config import get_config
                cfg = get_config()
                if dur < cfg.min_clip_duration or dur > cfg.max_clip_duration:
                    reasons.append(f"DURATION_OUT_OF_RANGE: {dur:.1f}s not in [{cfg.min_clip_duration}, {cfg.max_clip_duration}]")
                if ss.scores.narrative_completeness < cfg.min_narrative_completeness:
                    reasons.append(f"BELOW_NC_GATE: nc={ss.scores.narrative_completeness} < {cfg.min_narrative_completeness}")
                # Check overlap with selected
                for c in clips:
                    c_ss = c.scored_segment
                    if ss.segment.start < c_ss.segment.end and ss.segment.end > c_ss.segment.start:
                        reasons.append(f"OVERLAPS_WITH_SELECTED: clip segment_id={c_ss.segment.segment_id}")
                if not reasons:
                    reasons.append("OUTRANKED: higher-scoring non-overlapping segments were chosen first")
                entry["rejection_reasons"] = reasons
            ranking.append(entry)

        return {
            "domain_weights": weights,
            "ranking": ranking,
        }

    @classmethod
    def write_report(cls, report: dict, output_dir: Path) -> Path:
        diagnostics_dir = output_dir
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        path = diagnostics_dir / "diagnostics_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Diagnostic report written to %s", path)
        return path
