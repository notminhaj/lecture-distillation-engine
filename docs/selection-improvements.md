# Clip Selection: Improvements — Status & Remaining Work

## 1. Remaining Open Issue

All Phase 1 (zero-cost) improvements are complete. One gap remains:

### I2 (partial): LLM re-score of merged candidates

`_generate_merge_candidates()` (`selector.py`) heuristically merges adjacent [N, N+1] pairs where the first segment's `narrative_completeness < 0.5` and the combined duration fits. Scores are **interpolated** from both segments with a +0.1 completeness bonus rather than being re-evaluated by the LLM.

**Risk:** A poor first segment merged with a mediocre second can produce a misleadingly high composite score. The interpolation assumes merging always improves quality, which isn't guaranteed.

**Fix:** Send merge candidates as an additional LLM scoring batch (~1 extra API call per run, bounded by N/2 candidates).

---

## 2. What Has Been Implemented

### I1: Boundary refinement (heuristic, pre-selection) ✅

`BoundaryRefiner` (`segmentation/boundary_refiner.py`) runs 4 ordered passes on raw segments:
1. Snap boundaries to the nearest `TranscriptSegment` edge (±`sentence_boundary_window` seconds)
2. Shift boundaries to the largest inter-word silence gap (≥ `silence_threshold_ms`)
3. Protect strong hook openings — shift bleed backward by one TS if the hook has no preceding pause
4. Merge short / split long segments to enforce `[min_clip_duration, max_clip_duration]`

Toggle: `enable_boundary_refinement` in config (default: `True`).

> **Note:** The original I1 plan called for LLM-based *post-selection* refinement. Heuristic *pre-selection* refinement was implemented instead — zero API cost, but may miss cases requiring rhetorical judgment.

### I2: Adjacent-segment merge candidates (heuristic trigger) ✅ partial

Heuristic merge trigger and score interpolation implemented. LLM re-score of merged candidates not yet done (see §1 above).

### I3: Acoustic energy in composite score ✅

`ACOUSTIC_ENERGY_WEIGHT = 0.05` applied to `ss.acoustic_energy` inside `_weighted_score()` (`selector.py`). The weight is normalised with domain weights so the total remains comparable.

### I4: Sliding context window for LLM scoring ✅

`_build_sliding_context()` (`llm_scorer.py`) sends ~200 words from segments before the batch and ~200 words after, replacing the old fixed 500-word transcript head. The LLM now has local context to accurately judge `standalone_coherence` and `narrative_completeness` for deep-lecture segments.

### I5: Completeness gate ✅

`min_narrative_completeness = 0.4` config field. Segments below threshold are filtered before ranking, regardless of other scores. Logged as a separate metric.

### I6: Audio-onset hook detection ✅

`energy_onset_ratio` (energy in first 3s / segment average energy) computed in `AcousticScorer` and stored on `ScoredSegment`. Used in `_weighted_score()` as a `[0.7, 1.3]` multiplier on `hook_strength` — boosting segments with a strong acoustic opening, penalising silence or throat-clearing at the start.

### Bonus: Structural completeness penalty ✅

`_structural_completeness_penalty()` (`selector.py`) applies a `[0.7, 1.0]` multiplier on the `narrative_completeness` component for segments that:
- Start mid-sentence (first non-whitespace character is lowercase Latin)
- Lack terminal punctuation (`., !, ?, "`, or Arabic full stop `؟`)

This grounds structural completeness in text evidence without an LLM call.

---

## 3. Recommended Next Step

**I2 LLM re-score** (Phase 2 — bounded LLM cost):
- After the heuristic merge, collect merge candidates whose interpolated score would rank in the top-2K
- Send as a single extra LLM batch alongside the next scoring round
- Replace the interpolated scores with real LLM `EngagementAxes`
- Cost: ~1 extra API call per run (at most N/2 candidates, but only top scorers need re-evaluation)

---

## 4. Measurable Signals

### Completeness signals
- `narrative_completeness` score distribution (target: median > 0.65 for selected clips)
- `standalone_coherence` score (target: all selected clips > 0.5)
- **First/last sentence analysis**: does the clip start mid-sentence or end without terminal punctuation? (`_structural_completeness_penalty` already tracks this as a score modifier)
- **Context dependency ratio**: count of deictic references ("this", "that", "what I said", "as we discussed") in clip text — lower is better for standalone clips

### Engagement signals
- `emotional_resonance × acoustic_energy` product — combines semantic and acoustic emphasis
- `hook_strength × energy_onset_ratio` — combined text + audio hook signal (already wired into composite)
- Speech rate variance within segment — monotone (low variance) vs. dynamic delivery (high variance)
- **Rhetorical density**: count of questions, exclamations, and imperatives per minute in segment text
- Composite score gap between selected and rejected clips — should be meaningful, not noise
