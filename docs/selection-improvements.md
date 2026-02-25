# Clip Selection: Failure Modes & Improvement Plan

## 1. Failure Modes in the Current Approach

### F1: Boundary-blind selection — segments are taken as-is from the segmenter

The selector (`selector.py:85-88`) filters by duration and picks top-scored segments, but never questions whether a segment's boundaries are correct. The `SemanticSegmenter` cuts on *topic shift* (cosine similarity drop), which doesn't align with *rhetorical unit boundaries*. A topic can shift mid-sentence while an argument is still being concluded. Result: clips that start after the setup or cut before the punchline.

### F2: No boundary adjustment or padding

`Clip.start` and `Clip.end` are copied verbatim from `Segment.start`/`Segment.end` (`selector.py:120-121`). There's no mechanism to extend a clip by a few seconds to capture a trailing conclusion, or trim a slow preamble. The segment boundaries from cosine similarity are treated as gospel.

### F3: Acoustic features are computed but never used in selection

`AcousticScorer` computes `acoustic_energy` and `speech_rate_wpm`, stores them on `ScoredSegment` (`models.py:171-172`), but `_weighted_score()` (`selector.py:128-139`) only uses the 6 `EngagementAxes` fields. Energy and speech rate are dead weight in the selection decision.

### F4: LLM context window is truncated to 500 words

`llm_scorer.py:199` sends only the first 500 words of the lecture as context. For a 1-hour khutba (~8,000 words), the LLM evaluating segment #47 has no idea what preceded it. The `standalone_coherence` score becomes unreliable for late-lecture segments — the LLM can't distinguish "genuinely standalone" from "I lack context to tell."

### F5: `narrative_completeness` is scored but has no structural grounding

The LLM is asked to judge whether a segment "opens AND closes a thought." But the LLM only sees the segment text in isolation — it doesn't see what comes immediately before or after. It cannot know if the speaker was mid-argument. This axis is the most important for completeness, yet it's the least grounded.

### F6: No merge/split flexibility

The selector operates on fixed segments. If segment N scores 0.9 on engagement but 0.3 on narrative_completeness because the conclusion bleeds into segment N+1, there's no mechanism to consider the union of [N, N+1]. The selector can only pick or skip whole segments.

### F7: Hook strength is evaluated on text, not audio onset

The first 3 seconds of audio determine scroll-stopping power. But `hook_strength` is scored by the LLM from text alone — a speaker who pauses, clears throat, or has 2s of silence before a powerful statement would score well textually but poorly in practice. No audio-onset analysis exists.

---

## 2. Concrete Algorithmic Improvements

### I1: Boundary refinement pass (post-selection)

After selecting top-K segments, run a boundary refinement step:
- Look at the 1-2 `TranscriptSegment`s immediately before/after each selected `Segment`
- Use an LLM call to evaluate: "Does this segment start/end at a natural rhetorical boundary, or should it be extended?"
- Adjust `Clip.start` / `Clip.end` accordingly (capped at `max_clip_duration`)

**Method:** LLM-based. Only runs on K selected clips (not all N segments), so cost is bounded.

### I2: Adjacent-segment merging in the selector

Before ranking, generate candidate "super-segments" by merging adjacent segment pairs [N, N+1] when:
- Combined duration ≤ `max_clip_duration`
- The first segment's `narrative_completeness` < 0.5 (likely incomplete)
- Re-score the merged candidate (or interpolate scores with a completeness bonus)

**Method:** Heuristic merge trigger + LLM re-score of merged candidates.

### I3: Incorporate acoustic features into composite score

Add two signals to `_weighted_score()`:
- `acoustic_energy` as a proxy for speaker emphasis (weight ~0.05)
- `energy_onset_ratio`: ratio of energy in first 3s vs. segment average — high ratio = natural hook

**Method:** Pure heuristic (librosa). Zero additional latency or cost.

### I4: Sliding context window for LLM scoring

Replace the fixed 500-word context with a sliding window centered on each batch:
- Send ~200 words before the first segment in the batch and ~200 words after the last
- This gives the LLM local context to accurately judge `standalone_coherence` and `narrative_completeness`

**Method:** Prompt engineering (no new model calls — same batch, better context).

### I5: Completeness gate — hard filter on `narrative_completeness`

Add a minimum threshold: segments with `narrative_completeness < 0.4` are excluded before ranking, regardless of other scores. A clip that cuts mid-argument is worse than no clip.

**Method:** Heuristic threshold in the selector.

### I6: Audio-onset hook detection

Compute energy in the first 3s of each segment vs. the segment average. If the first 3s are silence or low-energy throat-clearing, penalize `hook_strength` by a multiplier (e.g., 0.7x).

**Method:** Heuristic (librosa). Complements the LLM's text-based hook evaluation.

---

## 3. Method Classification

| Improvement | Method | Why |
|---|---|---|
| I1: Boundary refinement | **LLM** | Requires rhetorical judgment ("is this a complete thought?") |
| I2: Adjacent merging | **Heuristic trigger + LLM re-score** | Merge trigger is simple; quality check needs LLM |
| I3: Acoustic in composite | **Heuristic** | librosa features already computed; just wire them in |
| I4: Sliding context | **Prompt engineering** | Same LLM calls, better input |
| I5: Completeness gate | **Heuristic** | Simple threshold filter |
| I6: Audio-onset hook | **Heuristic** | Energy ratio from librosa |

---

## 4. Tradeoffs

| Improvement | Compute/Latency | Reliability | Risk |
|---|---|---|---|
| I1: Boundary refinement | +1 LLM call per clip (~5 total) ≈ +3s | High — LLM good at rhetorical judgment | May extend clips beyond comfortable duration |
| I2: Adjacent merging | +N/2 candidate evaluations; ~1 extra LLM batch | Medium — interpolated scores may not reflect merged quality | Combinatorial explosion if merging 3+ segments |
| I3: Acoustic in composite | **Zero** — features already computed | High — energy is a reliable emphasis signal | Over-weighting could bias toward loud speakers |
| I4: Sliding context | **Zero** — same API calls | High — strictly better context | Slightly more tokens per call (~$0.001 more) |
| I5: Completeness gate | **Zero** | Medium — depends on LLM scoring calibration | May reject all candidates if threshold too aggressive |
| I6: Audio-onset hook | **Zero** — reuses loaded audio | High — silence detection is robust | False positives on intentional dramatic pauses |

---

## 5. Recommended MVP Improvement Path

### Phase 1 — Zero-cost wins (no new API calls):
1. **I4**: Sliding context window → better `narrative_completeness` and `standalone_coherence` scores
2. **I3**: Wire `acoustic_energy` into `_weighted_score()` with weight ~0.05
3. **I5**: Add `narrative_completeness >= 0.4` gate in selector
4. **I6**: Compute onset energy ratio, use as hook_strength multiplier

### Phase 2 — Bounded LLM cost:
5. **I1**: Post-selection boundary refinement (1 LLM call for K clips)

### Phase 3 — Structural change:
6. **I2**: Adjacent-segment merge candidates (requires re-scoring pipeline changes)

---

## 6. Measurable Signals

### Completeness signals:
- `narrative_completeness` score distribution (target: median > 0.65 for selected clips)
- `standalone_coherence` score (target: all selected clips > 0.5)
- **First/last sentence analysis**: does the clip's first `TranscriptSegment` start mid-sentence? Does the last one end mid-sentence? (Heuristic: check for lowercase start, no terminal punctuation)
- **Context dependency ratio**: count of deictic references ("this", "that", "what I said", "as we discussed") in the clip text — lower is better

### Engagement signals:
- `emotional_resonance` × `acoustic_energy` product — combines semantic and acoustic emphasis
- `hook_strength` adjusted by onset energy ratio
- Speech rate variance within segment — monotone (low variance) vs. dynamic delivery (high variance)
- **Rhetorical density**: count of questions, exclamations, and imperatives per minute in the segment text
- Composite weighted score of selected clips vs. rejected clips (gap should be meaningful, not noise)
