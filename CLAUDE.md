# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make install    # pip install -e ".[dev]"
make lint       # ruff check + mypy
make test       # pytest with coverage
make run-demo   # smoke test on sample.mp4
make clean      # rm outputs/ cache/ .pytest_cache/

# Run a single test file
pytest tests/test_scoring.py -v

# Run a single test
pytest tests/test_scoring.py::test_llm_scorer_batch -v
```

The CLI entry point is `distill` (installed via pyproject.toml):
```bash
distill run --input lecture.mp4 --domain khutba --clips 5
distill run --input "https://www.youtube.com/watch?v=..." --domain khutba --segmenter semantic
```

## Architecture

**7-stage sequential pipeline** wired in `src/distillation/pipeline.py`:

```
Source (file or URL)
  → [1] Ingestion     yt-dlp download + ffmpeg → 16kHz mono WAV
  → [2] Transcription faster-whisper → word-level timestamps
  → [3] Segmentation  embedding cosine similarity → semantic boundaries
  → [4] Scoring       Claude (6 axes) + librosa (acoustic features)
  → [5] Selection     greedy non-overlapping interval scheduling
  → [6] Generation    subtitles (word timestamps → SRT) + metadata (LLM captions)
  → [7] Export        ffmpeg clip cutting + JSON metadata
  → outputs/clip_NNN/{clip_NNN.mp4, clip_NNN.srt, clip_NNN_metadata.json}
```

Each stage is a package under `src/distillation/` with an abstract base class (e.g., `BaseTranscriber`, `BaseSegmenter`, `BaseScorer`) enabling swappable implementations without touching the pipeline.

## Data Flow

Canonical models in `src/distillation/models.py` flow through all stages:

`TranscriptSegment[]` → `Transcript` → `Segment[]` → `ScoredSegment[]` → `Clip[]` → `PipelineResult`

The `EngagementAxes` model (6 float fields, all [0.0, 1.0]) is the central scoring object: `semantic_density`, `emotional_resonance`, `standalone_coherence`, `narrative_completeness`, `domain_integrity`, `hook_strength`.

## Configuration

All config in `src/distillation/config.py` via Pydantic Settings. Loaded from `.env` (see `.env.example`). Accessed via `get_config()` (lazy singleton via `@lru_cache`).

Key env vars: `ANTHROPIC_API_KEY`, `WHISPER_MODEL` (default: `large-v3`), `CLAUDE_MODEL` (default: `claude-sonnet-4-6`), `DEFAULT_DOMAIN` (default: `khutba`).

## Key Design Decisions

- **LLM only where judgment is needed**: Claude for engagement scoring and caption generation; embeddings for topic boundaries; librosa for acoustic features; rule-based for subtitles.
- **Batch LLM calls**: `llm_scorer.py` sends 8 segments per API call with ~500-word transcript context window for coherence evaluation.
- **Domain weights**: `selector.py`'s `DOMAIN_WEIGHTS` dict controls per-axis composite scores. Khutba up-weights `standalone_coherence` (0.25) and `domain_integrity` (0.20) since TikTok viewers lack context and theological accuracy is critical.
- **ffmpeg stream-copy** (default) snaps to keyframes (~2-5s precision); `--reencode` flag gives frame-accurate cuts at the cost of processing time.

## Extending with a New Domain

Add to exactly 4 locations:
1. `Domain` enum in `models.py`
2. `DOMAIN_CONTEXT` prompt dict in `scoring/llm_scorer.py`
3. `DOMAIN_HASHTAG_SEEDS` dict in `generation/metadata.py`
4. `DOMAIN_WEIGHTS` dict in `selection/selector.py`

## Testing

The test suite is **fully mocked** — no API keys or GPU required. LLM calls are patched via fixtures in `tests/conftest.py`. Tests mirror the `src/distillation/` package structure.

## Segmentation Details

`SemanticSegmenter` (`segmentation/semantic.py`) uses sentence-transformers TextTiling:
- Default model: `sentence-transformers/all-MiniLM-L6-v2`
- For khutba (Arabic/English code-switching): `intfloat/multilingual-e5-small` is recommended
- Threshold: 0.35 (English), 0.45 (mixed language)

`SlidingWindowSegmenter` is the naive baseline (60s window, 15s stride) — useful for benchmarking or as fallback.
