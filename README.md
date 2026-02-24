# Lecture Distillation Engine

Transforms long-form lectures (initially Islamic khutbas/sermons) into
platform-ready short-form clips for TikTok, Instagram Reels, and YouTube Shorts.

## What it does

Given a lecture video or YouTube URL, the engine:

1. **Transcribes** with word-level timestamps (Whisper large-v3)
2. **Segments** into semantically coherent units (embedding-based topic detection)
3. **Scores** each segment on 6 engagement axes using Claude (LLM) + acoustic features (librosa)
4. **Selects** the top-K non-overlapping clips, weighted by domain
5. **Generates** subtitles, hook captions, hashtags, and B-roll suggestions
6. **Exports** trimmed clips with SRT and metadata JSON

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for the full system design,
technical challenges, LLM vs. traditional NLP decision matrix, evaluation
metrics, and anti-overengineering checklist.

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY

# 3. Run
distill run --input /path/to/lecture.mp4 --domain khutba --clips 5

# Or on a YouTube URL
distill run --input "https://www.youtube.com/watch?v=..." --domain khutba
```

## Output structure

```
outputs/
  clip_000/
    clip_000.mp4           # trimmed clip
    clip_000.srt           # subtitles (word-level timing)
    clip_000_metadata.json # scores, captions, hashtags, B-roll ideas
  clip_001/
    ...
```

## Configuration

All settings in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required |
| `WHISPER_MODEL` | `large-v3` | `tiny` for speed, `large-v3` for accuracy |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Scoring + generation model |
| `MIN_CLIP_DURATION` | `30` | Seconds |
| `MAX_CLIP_DURATION` | `90` | Seconds (YouTube Shorts hard limit) |
| `TARGET_CLIP_COUNT` | `5` | Clips to produce per lecture |
| `DEFAULT_DOMAIN` | `khutba` | `khutba \| podcast \| business_talk \| lecture \| generic` |
| `SEGMENTATION_STRATEGY` | `semantic` | `semantic \| sliding_window` |

## Supported Domains

| Domain | Scoring adjustments |
|--------|-------------------|
| `khutba` | Up-weights `domain_integrity` (theological completeness) and `standalone_coherence` |
| `podcast` | Up-weights `hook_strength` |
| `business_talk` | Up-weights `semantic_density` |
| `lecture` | Balanced weights |
| `generic` | Default weights |

Adding a new domain requires editing 4 constants in existing files.
See `docs/architecture.md` §9.

## Development

```bash
make install   # pip install -e ".[dev]"
make lint      # ruff + mypy
make test      # pytest with coverage
```

Tests are fully mocked — no API key or GPU required to run them.

## Extending

- **New transcriber**: subclass `BaseTranscriber`, implement `transcribe()`.
- **New segmenter**: subclass `BaseSegmenter`, implement `segment()`.
- **New domain**: add 4 lines across `models.py`, `llm_scorer.py`, `metadata.py`, `selector.py`.
- **Platform uploader**: add a new module in `export/` — the `Clip` object contains everything needed.
