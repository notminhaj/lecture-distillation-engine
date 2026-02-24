# Lecture Distillation Engine — Architecture

## 1. System Overview

The engine is a **sequential, stage-based pipeline** that transforms a long-form
lecture into a set of platform-ready short-form clips.

```
Source Media (file or URL)
        │
        ▼
┌─────────────────┐
│   1. Ingestion  │  yt-dlp download + ffmpeg audio extraction → 16kHz mono WAV
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Transcription│  faster-whisper → word-level timestamps + language detection
└────────┬────────┘
         │  Transcript (segments + words)
         ▼
┌─────────────────┐
│ 3. Segmentation │  Embedding cosine similarity → topic boundary detection
└────────┬────────┘
         │  List[Segment]
         ▼
┌─────────────────┐
│  4. Scoring     │  LLM (Claude) + librosa → multi-axis EngagementAxes
└────────┬────────┘
         │  List[ScoredSegment]
         ▼
┌─────────────────┐
│  5. Selection   │  Greedy non-overlap + domain weights → top-K clips
└────────┬────────┘
         │  List[Clip]
         ▼
┌─────────────────┐
│  6. Generation  │  Subtitles (word timestamps) + Metadata (LLM captions)
└────────┬────────┘
         │  Clips with subtitles + metadata
         ▼
┌─────────────────┐
│   7. Export     │  ffmpeg clip cutting + SRT + metadata JSON
└─────────────────┘
         │
         ▼
outputs/clip_000/clip_000.mp4
         clip_000.srt
         clip_000_metadata.json
```

---

## 2. Data Model Flow

Every stage is a transformation between these core types:

```
Transcript
  └── TranscriptSegment[]
        └── Word[]            ← word-level timestamps from Whisper

Segment                       ← contiguous span of TranscriptSegments
  └── transcript_segment_ids  ← anchored to real Whisper output

ScoredSegment
  ├── Segment
  ├── EngagementAxes          ← 6 LLM-scored axes [0.0, 1.0]
  ├── acoustic_energy         ← librosa RMS
  └── speech_rate_wpm

Clip
  ├── ScoredSegment
  ├── SubtitleLine[]
  └── ClipMetadata            ← hook caption, hashtags, B-roll suggestions
```

Key invariant: **Segment timestamps always resolve to real TranscriptSegment
data**. There are no fabricated timestamps — every boundary is anchored to a
Whisper word boundary. This ensures subtitle alignment is always correct.

---

## 3. Hardest Technical Challenges

### 3.1 Semantic Boundary Detection

The core challenge: a khutba is not a podcast. It has:
- **Multi-paragraph arguments** that build across 5-10 minutes.
- **Arabic/English code-switching** that breaks embedding models tuned for English.
- **Rhetorical recycling**: the same Quran verse may be cited in three different
  contexts across the lecture.

**Solution**: Use multilingual embeddings (`intfloat/multilingual-e5-small`) for
code-switching content, and tune the cosine similarity threshold on a labelled
set of 10-20 khutbas. Threshold is a single hyperparameter in config — no code
changes needed to re-tune per domain.

**Failure mode**: TextTiling approaches (our current approach) tend to over-segment
on topics that circle back. The LLM scoring layer partially compensates: clips
that are mid-argument score low on `narrative_completeness`.

### 3.2 LLM Scoring Quality

The LLM scores are only as good as the prompt. Key risks:

- **Sycophancy**: Claude may score everything above 0.7 to seem helpful.
  Mitigation: use comparative prompting ("rank these 8 segments") rather than
  absolute scoring. *Not yet implemented — marked as v2.*

- **Context window for long lectures**: a 90-minute khutba is ~15,000 words.
  We can't fit the full text as context for coherence evaluation.
  Current mitigation: send first 500 words as context. Better: chunk + summarise
  the lecture first, send the summary as context. *Marked as v2.*

- **Domain calibration**: `domain_integrity` for khutba requires knowing what
  constitutes a complete ruling. This is injected via `DOMAIN_CONTEXT` in
  `llm_scorer.py` — domain experts should review and refine these prompts.

### 3.3 Clip Boundary Precision

ffmpeg stream-copy (`-c copy`) snaps to the nearest keyframe (typically every
2-5 seconds). For a 30-second clip, this could cut off the first sentence.

Options:
1. **Stream-copy** (default): fast, ~2s inaccuracy. Acceptable for review.
2. **Re-encode** (`force_reencode=True`): frame-accurate, ~10× slower.
3. **Smart seek**: `-ss` before `-i` (fast seek) + `-ss` after `-i` (trim).

For production, use re-encode. Add it as a flag in `VideoExporter`.

### 3.4 Subtitle Alignment for Arabic

Whisper's word-level timestamps for Arabic have higher variance than English.
Code-switched segments (Arabic phrase, English sentence) can produce misaligned
timestamps at the language boundary.

Mitigation: treat Arabic words as atomic subtitle units (don't split mid-word),
and set a higher `MAX_LINE_DURATION` for Arabic-heavy segments.

### 3.5 Measuring "Engagement" Without Ground Truth

We have no labels. The LLM scores are a proxy, not ground truth.

**Evaluation pipeline** (see Section 6) must include human feedback collection
from the earliest possible point. Even 50 labelled clips (human-ranked) enables:
- Pairwise ranking loss to calibrate axis weights.
- Threshold tuning for duration constraints.

---

## 4. LLM vs. Traditional NLP — Where Each Adds Value

| Task | Tool | Reason |
|------|------|--------|
| Transcription | Whisper (local model) | Deterministic, fast, word-level timestamps |
| Topic boundary detection | Sentence embeddings + cosine similarity | O(n), no API cost, well-calibrated thresholds |
| Acoustic energy, speech rate | librosa (signal processing) | Can't be done by LLM; no access to audio |
| Engagement axis scoring | Claude (LLM) | Requires language understanding, domain knowledge, judgment |
| Clip hook quality | Claude (LLM) | Cultural/rhetorical context needed |
| Subtitle line generation | Rule-based (word timestamps) | Deterministic, fast, no hallucination risk |
| Caption/hashtag generation | Claude (LLM) | Creative, platform-aware, domain-contextual |
| B-roll suggestions | Claude (LLM) | Requires understanding what's being described |
| Clip deduplication | Cosine similarity on embeddings | Fast, no API cost |

**Principle**: Use the LLM where the decision requires **judgment** or **domain
knowledge**. Use deterministic tools where the output is **computable** from
the data directly.

---

## 5. MVP Scope

The MVP is the current codebase. It deliberately excludes:

**Included in MVP:**
- [ ] yt-dlp download + ffmpeg audio extraction
- [ ] Whisper transcription with word timestamps
- [ ] Sliding window segmentation (fast, no GPU)
- [ ] Semantic segmentation (better quality, needs sentence-transformers)
- [ ] LLM multi-axis scoring (6 axes, batched)
- [ ] Acoustic feature extraction (energy, WPM)
- [ ] Domain-weighted clip selection (non-overlapping, greedy)
- [ ] SRT subtitle generation from word timestamps
- [ ] LLM caption + hashtag + B-roll generation
- [ ] ffmpeg clip cutting
- [ ] CLI (`distill run ...`)

**Explicitly deferred to v2:**
- Speaker diarization (pyannote) — adds ~15 min processing time per lecture
- Comparative LLM scoring ("rank these 8 clips") — better calibration
- Full lecture summarisation as LLM context
- Platform upload APIs (TikTok, YouTube Shorts, Instagram)
- Human feedback collection loop
- Finetuned scoring model (replace LLM scorer after collecting labels)
- Gameplay/visual overlay rendering
- Real-time processing pipeline (currently batch only)

**Why not speaker diarization in MVP?**
Khutbas typically have a single speaker. Diarization adds infrastructure
complexity (pyannote needs GPU, HuggingFace auth) without improving the core
product for the primary use case. Add when expanding to panels/podcasts.

---

## 6. Evaluation Metrics

### Offline (no user data needed)

| Metric | How to compute | Target |
|--------|---------------|--------|
| **Transcription WER** | Whisper vs. human transcript on 10 test khutbas | < 10% for clear audio |
| **Segmentation F1** | Human-labelled boundaries vs. predicted | > 0.70 |
| **Subtitle sync error** | Mean offset between word boundary and subtitle display | < 200ms |
| **LLM score variance** | Std dev of scores across batch; low variance = sycophancy risk | > 0.15 |
| **Clip duration distribution** | Are selected clips within [min, max]? | 100% pass |

### Online (requires publishing + tracking)

| Metric | Proxy for | How to collect |
|--------|-----------|----------------|
| **Watch-through rate** | Clip engagement | Platform analytics |
| **Completion rate** | Narrative completeness | YouTube/TikTok analytics |
| **Follower conversion** | Standalone coherence | Before/after publish tracking |
| **Human ranking agreement** | Overall score quality | A/B test: LLM-selected vs. random |

### Human Evaluation Protocol

For each test khutba, have a domain expert (Islamic scholar or senior content
creator) rank all candidate clips from 1–5 on:
- "Does this clip make sense without context?" (standalone_coherence proxy)
- "Is this theologically complete?" (domain_integrity proxy)
- "Would I watch this to the end?" (narrative_completeness + emotional_resonance proxy)

Use this to calibrate `DOMAIN_WEIGHTS` in `selector.py`.

---

## 7. Repository Structure

```
lecture-distillation-engine/
├── src/distillation/
│   ├── __init__.py           # Public API: Pipeline, Domain
│   ├── cli.py                # Typer CLI entry point
│   ├── config.py             # Pydantic Settings — all env vars
│   ├── models.py             # Canonical data models (Word → Clip)
│   ├── pipeline.py           # Thin orchestration layer
│   ├── ingestion/
│   │   ├── downloader.py     # yt-dlp wrapper
│   │   └── extractor.py      # ffmpeg audio normalisation
│   ├── transcription/
│   │   ├── base.py           # BaseTranscriber ABC
│   │   └── whisper.py        # faster-whisper implementation
│   ├── segmentation/
│   │   ├── base.py           # BaseSegmenter ABC
│   │   ├── sliding_window.py # Fixed-size window (baseline)
│   │   └── semantic.py       # Embedding cosine similarity
│   ├── scoring/
│   │   ├── base.py           # BaseScorer ABC
│   │   ├── llm_scorer.py     # Claude multi-axis scorer
│   │   └── acoustic.py       # librosa energy + WPM
│   ├── selection/
│   │   └── selector.py       # Constraint-based greedy selector
│   ├── generation/
│   │   ├── subtitles.py      # Word-timestamp → SubtitleLine
│   │   └── metadata.py       # LLM caption + hashtag generator
│   └── export/
│       └── video.py          # ffmpeg clip cutter + SRT/JSON writer
├── tests/
│   ├── conftest.py           # Shared fixtures (all LLM calls mocked)
│   ├── test_models.py
│   ├── test_segmentation.py
│   ├── test_scoring.py
│   ├── test_selection.py
│   └── test_subtitles.py
├── docs/
│   ├── architecture.md       # This document
│   └── domain_guidelines/
│       └── khutba.md         # Domain-specific prompt engineering notes
├── scripts/
│   └── run_pipeline.py       # Standalone runner
├── pyproject.toml
├── Makefile
└── .env.example
```

**Why this structure?**

1. `src/` layout: standard Python packaging convention, prevents import confusion.
2. Each stage is a Python package: isolated, independently testable, no circular imports.
3. Abstract base classes: every stage can be swapped without touching the pipeline.
4. `config.py` is the single source of truth: no `os.getenv()` scattered in modules.
5. Tests mirror the source structure: easy to find the test for any module.

---

## 8. Anti-Overengineering Checklist

Before adding any feature, ask:

- [ ] **Is this needed for the first real user?** If not, defer it.
- [ ] **Does this require a new external dependency?** Justify the cost (setup
  complexity, version conflicts, maintenance burden).
- [ ] **Am I abstracting for one use case?** Three similar lines of code > premature abstraction.
- [ ] **Can I measure whether this improves output quality?** If not, it's speculation.
- [ ] **Is the existing interface sufficient?** Prefer extending `DOMAIN_CONTEXT`
  in `llm_scorer.py` over adding a new class hierarchy.

**Specific things to resist:**
- Vector database for segment retrieval (you have << 1,000 segments per lecture;
  use a list).
- Microservice decomposition (this is a batch pipeline, not a web service yet).
- Custom training of a scoring model before collecting 500+ labelled examples.
- Async pipeline before profiling shows I/O is the bottleneck.
- Multiple embedding model variants before benchmarking the default.

---

## 9. Extending to New Domains

To add a new domain (e.g., `ted_talk`):

1. Add `TED_TALK = "ted_talk"` to `Domain` enum in `models.py`.
2. Add a system prompt fragment to `DOMAIN_CONTEXT` in `llm_scorer.py`.
3. Add domain-specific hashtag seeds to `DOMAIN_HASHTAG_SEEDS` in `metadata.py`.
4. Add domain weights to `DOMAIN_WEIGHTS` in `selector.py`.
5. Add a `--domain ted_talk` test case to `test_selection.py`.

No other files change. This is the extensibility payoff of the current design.
