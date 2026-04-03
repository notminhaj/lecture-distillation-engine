# Khutba Clip Quality Reward Model — Implementation Plan

## Vision

Automate the extraction and ranking of high-quality short-form khutba clips (15–20s) that a viewer would want to share — clips where someone who has never heard the full khutba would learn something meaningful and want to follow the entire lecture for more Islamic wisdom.

---

## What Makes a Good Khutba Clip

A good clip satisfies all three of these qualities:

1. **Completeness** — It makes a full point without needing what came before or after. The viewer doesn't feel like they walked into the middle of a conversation.
2. **Emotional Pull** — It hits you. It's not "ok cool, good to know." There's a human weight to it — conviction, urgency, vulnerability, or awe.
3. **Opening Hook** — The first few seconds grab attention. Something about the opener makes you stop scrolling.

A bonus quality (captured as a feature, not a labeling axis): the clip references a source of wisdom — a hadith, Quranic verse, or prophetic story — briefly (1–2 sentences). Longer citations dilute the short-form impact.

### Known Failure Modes

These are the problems the reward model must learn to detect:

| Failure | What It Looks Like | Root Cause |
|---|---|---|
| **Incomplete meaning** | The purpose of what the speaker is saying is unclear. You're left wondering "...and?" | Clip boundaries cut through a rhetorical unit |
| **Overshoot** | The clip starts strong but continues past its natural endpoint. The message dilutes. | End-of-clip detection is weak |
| **Zero emotional impact** | The content is technically correct but feels like a textbook. No reason to share it. | Candidate selection over-indexes on information density, ignores affect |

---

## Architecture Overview

```
YouTube lectures
       │
       ▼
┌──────────────────┐
│  Existing Pipeline │  ← candidate generation (keep as-is for now)
│  --clips 20-30    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Feature Extractor │  ← NEW: text + acoustic + embedding features
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Reward Model      │  ← NEW: trained on pairwise human preferences
│  P(good) per clip  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Ranked Output     │  ← top clips + metadata (description, hashtags, source references)
└──────────────────┘
```

---

## Phase 0: Infrastructure & Feature Engineering

**Goal:** Build the feature extractor and labeling data store. No behavior change to existing pipeline.

### 0a. Feature Extractor

New file: `src/distillation/scoring/feature_extractor.py`

Builds a feature vector per clip from three sources:

**Text-derived features (bypass LLM where possible):**

| Feature | Type | What It Captures |
|---|---|---|
| `starts_with_conjunction` | bool | Clip opens with "and", "but", "so", etc. |
| `starts_with_continuation_word` | bool | Uses existing `_CONTINUATION_STARTERS` |
| `has_terminal_punctuation` | bool | Clip ends with `.` `!` `?` |
| `referential_opener_detected` | bool | Opens with "this is why...", "that's the point..." |
| `unresolved_pronoun_count` | int | Pronouns without antecedents near clip start |
| `word_count` | int | Total words in clip |
| `sentence_count` | int | Number of sentences |
| `avg_sentence_length` | float | Words per sentence |
| `question_count` | int | Rhetorical questions (engagement signal) |
| `exclamation_count` | int | Emphasis markers |
| `duration_seconds` | float | Clip length |
| `words_per_second` | float | Speaking pace |
| `contains_quran_reference` | bool | Detects Quranic citation patterns |
| `contains_hadith_reference` | bool | Detects hadith citation patterns |
| `citation_length_ratio` | float | What fraction of the clip is citation vs. commentary |

**Semantic embedding features (via `sentence-transformers`):**

| Feature | Type | What It Captures |
|---|---|---|
| `semantic_self_containedness` | float | 1 - cosine_similarity(clip_embedding, surrounding_context_embedding). Higher = more standalone |
| `opening_hook_distinctiveness` | float | Cosine distance between first sentence embedding and rest of clip. Higher = the opener stands apart |

Use `all-MiniLM-L6-v2` for embeddings — lightweight, fast, good enough for this.

**Acoustic features (keep existing):**

| Feature | Type | What It Captures |
|---|---|---|
| `acoustic_energy` | float | Overall energy level |
| `energy_onset_ratio` | float | Energy at clip start vs. average |
| `speech_rate_wpm` | float | Words per minute |

**LLM scores (keep as noisy signals, don't rely on them):**

The 6 existing LLM axes (`semantic_density`, `emotional_resonance`, `standalone_coherence`, `narrative_completeness`, `domain_integrity`, `hook_strength`) remain as features. They're noisy but still carry signal. The reward model will learn to weight them appropriately — probably down-weighting `standalone_coherence` which has been unreliable.

**Key design principle:** Every existing penalty/rule in `_weighted_score()` becomes a feature the model can learn to weight. `starts_with_conjunction` is not a rule that says "bad" — it's a signal the model considers alongside everything else. "And the prophet said..." starts with a conjunction but is a great clip, and the model will learn this interaction.

### 0b. Label Store

New file: `src/distillation/labeling/store.py`

Append-only JSONL at `data/pairwise_labels.jsonl`. Each entry is one pairwise comparison:

```json
{
  "comparison_id": "cmp_001",
  "clip_a": {
    "source_file": "lecture_001.mp4",
    "segment_id": 7,
    "clip_dir": "outputs/clip_007"
  },
  "clip_b": {
    "source_file": "lecture_001.mp4",
    "segment_id": 12,
    "clip_dir": "outputs/clip_012"
  },
  "preferences": {
    "completeness": "a" | "b" | "tie",
    "emotional_pull": "a" | "b" | "tie",
    "hook": "a" | "b" | "tie",
    "overall": "a" | "b" | "tie"
  },
  "labeled_at": "2026-04-03T12:00:00Z",
  "session_id": "session_001",
  "features_a": { ... },
  "features_b": { ... }
}
```

Features are stored at label time so training never needs the original video files.

### 0c. Config Additions

Modify: `src/distillation/config.py`

```python
reward_model_weight: float = 0.0      # 0 = hand-coded only, 1 = model only
reward_model_path: Path = Path("data/models/reward_model.pkl")
label_store_path: Path = Path("data/pairwise_labels.jsonl")
clips_per_lecture: int | None = None   # None = auto-scale by duration
```

Auto-scaling logic: `clips = max(15, min(30, duration_minutes // 3))`

### 0d. Diagnostics Integration

Modify: `src/distillation/diagnostics.py`

Include per-clip feature vectors in the diagnostics report. When the reward model exists, include its predicted score alongside the hand-coded score for comparison.

---

## Phase 1: Labeling App

**Goal:** Build a local web app for pairwise clip comparison.

### Architecture

```
Flask/FastAPI backend
    │
    ├── Serves clip video files from outputs/clip_XXX/
    ├── Randomly pairs clips for comparison
    ├── Stores preferences to data/pairwise_labels.jsonl
    │
    └── React frontend
         ├── Two video players side-by-side
         ├── Transcript text below each player
         ├── 4 preference buttons per pair:
         │     Completeness:    [← A]  [Tie]  [B →]
         │     Emotional Pull:  [← A]  [Tie]  [B →]
         │     Hook:            [← A]  [Tie]  [B →]
         │     Overall:         [← A]  [Tie]  [B →]
         ├── Keyboard shortcuts (arrow keys for overall, number keys for axes)
         ├── Progress counter: "12 / 30 comparisons this session"
         └── Session summary on completion
```

### Key Design Decisions

- **Random pairing:** Clips are paired randomly across the full pool. No structure needed — Bradley-Terry handles sparse comparisons well.
- **Session length:** 30 comparisons per session (user-chosen limit).
- **All 4 judgments per pair:** 3 axes + 1 overall. The overall is the primary training signal; axis preferences are diagnostic and become features.
- **Video playback:** Both clips play simultaneously or sequentially (user toggles). Replay buttons for each.
- **No editing of labels:** Once submitted, a comparison is final. This prevents second-guessing and keeps sessions fast.

### CLI Integration

```bash
# Import clips from one or more pipeline runs
distill label import --run-dir outputs/

# Launch the labeling app
distill label start --session-size 30

# View labeling stats
distill label stats
```

---

## Phase 2: Data Collection

**Goal:** Generate enough clips and label enough comparisons to train a v1 model.

### Batch Pipeline Script

New file: `scripts/batch_pipeline.py`

```bash
# Process multiple YouTube lectures in sequence
python scripts/batch_pipeline.py \
  --urls urls.txt \
  --output-dir data/batch_001/
```

Where `urls.txt` is a plain text file with one YouTube URL per line. The script runs the existing pipeline on each, auto-scaling clip count by lecture duration.

### Data Collection Target

| What | Target | Reasoning |
|---|---|---|
| **Speakers** | 4–5 distinct khateebs | Prevents overfitting to one speaker's style |
| **Lectures** | ~20 total | 4–5 per speaker |
| **Clips per lecture** | 20–30 (auto-scaled) | ~500 candidate clips total |
| **Labeling sessions** | ~10 sessions of 30 | ~300 pairwise comparisons |
| **Main speaker allocation** | ~40% of clips | Can be higher since it's the primary use case, but leave room for generalization |

### Labeling Protocol

When comparing two clips, ask yourself for each axis:

- **Completeness:** "If I sent just this clip to someone, would they get the full point? Or would they ask 'wait, what was the context?'"
- **Emotional Pull:** "Does this clip make me *feel* something? Or is it just information?"
- **Hook:** "In the first 3 seconds, would I stop scrolling?"
- **Overall:** "Which of these two clips would I actually share?"

---

## Phase 3: Training

**Goal:** Convert pairwise preferences into clip quality scores, then train a reward model.

### Step 1: Pairwise → Rankings via Bradley-Terry

Library: `choix` (Python, pip install)

```python
import choix
import numpy as np

# For each axis and for overall:
# Convert pairwise preferences to (winner_idx, loser_idx) tuples
# Feed to choix to get per-clip strength parameters

# Overall quality scores
overall_data = [(clip_a_idx, clip_b_idx) for comp in comparisons
                if comp["preferences"]["overall"] == "a"]
overall_params = choix.ilsr_pairwise(n_clips, overall_data)
# overall_params[i] = latent quality score for clip i (higher = better)

# Per-axis scores (same process)
completeness_params = choix.ilsr_pairwise(n_clips, completeness_data)
emotional_params = choix.ilsr_pairwise(n_clips, emotional_data)
hook_params = choix.ilsr_pairwise(n_clips, hook_data)
```

The per-axis Bradley-Terry scores become **additional features** for the reward model. The overall Bradley-Terry score becomes the **training target**.

### Step 2: Train Reward Model

New file: `scripts/train_reward_model.py`

**Input features per clip:**
- Text features (~15)
- Acoustic features (3)
- Embedding features (2)
- LLM scores (6)
- Bradley-Terry axis scores (3: completeness, emotional, hook)

Total: ~29 features → predicting overall quality score

**Model:** `GradientBoostingRegressor` from scikit-learn.
- With ~500 clips and ~300 comparisons, this is the right complexity level.
- Fast inference (microseconds per clip).
- Interpretable via feature importances — you can see what the model learned matters.
- Falls back to `Ridge` regression if < 100 clips with scores.

**Training script outputs:**
- Saved model: `data/models/reward_model.pkl`
- Feature importance ranking (printed + saved)
- Leave-one-out cross-validation correlation
- Scatter plot: predicted vs. actual quality scores

### Step 3: Active Learning (After v1)

Once v1 is trained, the model can identify clips where it's **most uncertain** — these are the most valuable clips to label next. The labeling app gains a mode:

```bash
distill label start --mode active --model data/models/reward_model.pkl
```

This surfaces pairs where the model's predicted quality scores are closest together (hardest to distinguish). Labeling these maximally improves the model per comparison.

---

## Phase 4: Integration

**Goal:** Wire the reward model into the pipeline so it ranks clips at selection time.

### Reward Model Wrapper

New file: `src/distillation/scoring/reward_model.py`

```python
class RewardModel:
    def predict_quality(self, segment: ScoredSegment, domain: Domain) -> float:
        """Returns predicted quality score (higher = better)."""
        features = self.feature_extractor.extract(segment)
        return self.model.predict([features])[0]
```

### Blend into Selection

Modify: `src/distillation/selection/selector.py`

```python
def _weighted_score(self, segment, domain):
    hand_coded = self._original_weighted_score(segment, domain)
    if self.config.reward_model_weight == 0.0 or not self.reward_model:
        return hand_coded
    model_score = self.reward_model.predict_quality(segment, domain)
    w = self.config.reward_model_weight
    return (1 - w) * hand_coded + w * model_score
```

### Cold-Start Progression

| Labels | `reward_model_weight` | Behavior |
|---|---|---|
| 0 | 0.0 | Hand-coded only (today's behavior, nothing changes) |
| < 100 comparisons | 0.0 | Model predictions appear in diagnostics only (advisory) |
| 100–200 comparisons | 0.3 | Blend: model influences ranking but doesn't dominate |
| 200+ comparisons | 0.7–1.0 | Model primary. Hand-coded scoring becomes fallback. |

**The blend is transitional, not permanent.** The goal is full replacement of `_weighted_score()` once the model is validated.

---

## Phase 5: Evaluation

### The Blind Test

After each training iteration:

1. Run the pipeline on a **new lecture** the model has never seen clips from
2. Take the model's top 5 ranked clips
3. Watch/listen to all 5 **without seeing their scores**
4. Score each yourself: would you share this? (yes/no)
5. Compare your pass rate to the target

| Metric | Baseline (Current) | v1 Target | Stretch Goal |
|---|---|---|---|
| **Pass rate (top 5)** | 0.5 / 5 (~10%) | 3 / 5 (60%) | 4 / 5 (80%) |

### Diagnostic Analysis

When the model gets a clip wrong (ranked high but you'd never share it, or ranked low but it's actually great), the diagnostics report should show:

- The clip's full feature vector
- Which features contributed most to the model's score (SHAP values or feature contribution breakdown)
- The closest clips in the training set that the model considered similar
- The model's per-axis scores vs. its overall score

This tells you whether the problem is:
- **Missing feature:** The model can't see the quality that matters (→ add a new feature)
- **Labeling gap:** The model hasn't seen enough clips like this (→ label more in this region)
- **Model limitation:** The features are there but the model can't learn the pattern (→ more data or different model)

### Feedback Loop

Your corrections during evaluation become new training data:

1. You disagree with the model's ranking of a clip → that becomes implicit pairwise data (you preferred clip X over the model's higher-ranked clip Y)
2. Retrain with the expanded dataset (takes seconds with gradient boosting)
3. Re-evaluate on another new lecture
4. Repeat until pass rate stabilizes

---

## Files Summary

### Create

| File | Purpose |
|---|---|
| `src/distillation/scoring/feature_extractor.py` | Extracts ~29 features per clip |
| `src/distillation/scoring/reward_model.py` | Wrapper for trained model inference |
| `src/distillation/labeling/__init__.py` | Package init |
| `src/distillation/labeling/store.py` | Pairwise label JSONL store |
| `src/distillation/labeling/app.py` | Flask/FastAPI labeling app backend |
| `src/distillation/labeling/frontend/` | React frontend for pairwise comparison |
| `scripts/train_reward_model.py` | Training script (Bradley-Terry + GBR) |
| `scripts/batch_pipeline.py` | Process multiple YouTube lectures |
| `tests/test_feature_extractor.py` | Feature extraction tests |
| `tests/test_labeling.py` | Label store tests |
| `tests/test_reward_model.py` | Model training + prediction tests |
| `data/pairwise_labels.jsonl` | Empty label store |
| `data/models/.gitkeep` | Model output directory |

### Modify

| File | Change |
|---|---|
| `src/distillation/selection/selector.py` | Blend reward model into `_weighted_score()` |
| `src/distillation/config.py` | Add `reward_model_weight`, `reward_model_path`, `label_store_path`, `clips_per_lecture` |
| `src/distillation/cli.py` | Add `label` subcommand with `import`, `start`, `stats` |
| `src/distillation/diagnostics.py` | Include feature vectors and model predictions in report |
| `pyproject.toml` | Add dependencies: `choix`, `sentence-transformers`, `flask`, `joblib` |

---

## Dependencies

| Package | Purpose |
|---|---|
| `choix` | Bradley-Terry model for pairwise → rankings |
| `sentence-transformers` | Semantic embeddings for self-containedness features |
| `flask` or `fastapi` | Labeling app backend |
| `joblib` | Model serialization |
| `scikit-learn` | GradientBoostingRegressor (likely already installed) |
| `shap` | Feature importance analysis (optional, for diagnostics) |

---

## Implementation Order

This is the order to build things, optimized for getting value fast:

1. **Feature extractor** (0a) — pure function, easy to test, no dependencies on other new code
2. **Label store** (0b) — simple JSONL append, needed before labeling
3. **Config additions** (0c) — quick, unblocks everything else
4. **Batch pipeline script** — generate clips from multiple lectures
5. **Labeling app** (Phase 1) — the main data collection tool
6. **Data collection** (Phase 2) — 10 labeling sessions
7. **Training script** (Phase 3) — Bradley-Terry + reward model
8. **Integration** (Phase 4) — blend into selector
9. **Blind test evaluation** (Phase 5) — validate and iterate
10. **Diagnostics integration** (0d, 0e) — enrich reports with model predictions

Steps 1–4 can be handed to Claude Code sequentially. Step 5 (the labeling app) is the most complex build and deserves a focused session. Steps 6 onward are your workflow, not code changes.

---

## Notes for Claude Code

When handing this plan to Claude Code for implementation:

- **Phase 0 is safe to implement immediately** — it adds code but changes no behavior (reward_model_weight defaults to 0.0).
- **The feature extractor must be a pure function** — `extract(segment) → dict[str, float]`. No side effects, fully deterministic for the text features. Embedding features require model loading but should be lazy-loaded.
- **The labeling app is a standalone service** — it reads from the pipeline's output directory structure (`outputs/clip_XXX/`) and writes to `data/pairwise_labels.jsonl`. It does not import or depend on the pipeline's runtime code beyond the data schemas.
- **The training script is offline** — it reads `data/pairwise_labels.jsonl`, trains, and writes to `data/models/`. It never runs during pipeline execution.
- **Existing tests must not break.** The selector's `_weighted_score()` with `reward_model_weight=0.0` must produce identical results to today's behavior.
