# Lecture Distillation Engine

> **Work in progress** — this is an early prototype. Core functionality works end-to-end, but clip quality and output polish are still being actively improved.

Transforms long-form lectures (initially Islamic khutbas/sermons) into
platform-ready short-form clips for TikTok, Instagram Reels, and YouTube Shorts.

## What it does

Given a lecture video or YouTube URL, the engine runs a 7-stage pipeline:

1. **Ingests** — downloads via yt-dlp (or reads local file), extracts 16 kHz mono WAV for Whisper
2. **Transcribes** — word-level timestamps with Whisper large-v3, hallucination filtering, language detection
3. **Segments / Nominates** — finds candidate clip boundaries via embedding-based topic detection, LLM moment detection, or both
4. **Scores** — rates each candidate on 6 engagement axes using an LLM + acoustic features (librosa)
5. **Selects** — greedy non-overlapping interval scheduling, weighted per domain
6. **Generates** — word-level subtitles (SRT + ASS), hook captions, hashtags, B-roll suggestions
7. **Exports** — trimmed MP4s with burned-in subtitles and metadata JSON

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (and OPENAI_API_KEY if using LLM nomination/scoring)

# 3. Run on a file
distill run --input /path/to/lecture.mp4 --domain khutba --clips 5

# Or on a YouTube URL
distill run --input "https://www.youtube.com/watch?v=..." --domain khutba
```

## CLI Reference

```bash
distill run --input <path|url> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --input` | — | Video file path or YouTube URL (required) |
| `-d, --domain` | `khutba` | Content domain — controls scoring weights and prompts |
| `-n, --clips` | `5` | Number of clips to generate |
| `--segmenter` | `semantic` | Segmentation strategy: `semantic` or `sliding_window` |
| `--nominator` | `llm` | Candidate strategy: `llm`, `semantic`, or `hybrid` |
| `-o, --output` | `outputs/` | Output directory |
| `--no-export` | — | Skip video export (score and preview clips only) |
| `--fast-export` | — | Stream-copy instead of re-encode (faster, ~2–5s keyframe drift, no subtitle burn-in) |

## Output Structure

```
outputs/
  clip_000/
    clip_000.mp4              # trimmed clip with burned-in subtitles
    clip_000.srt              # word-level subtitles (for platform upload)
    clip_000.ass              # styled subtitles (Komika Axis font, shorts-optimised)
    clip_000_metadata.json    # scores, hook caption, hashtags, B-roll suggestions
  clip_001/
    ...
```

### Metadata JSON schema

```json
{
  "clip_id": 0,
  "source_start": 291.0,
  "source_end": 334.3,
  "duration": 43.3,
  "exported_duration": 43.5,
  "scores": {
    "semantic_density": 0.8,
    "emotional_resonance": 0.9,
    "standalone_coherence": 0.85,
    "narrative_completeness": 0.75,
    "domain_integrity": 0.95,
    "hook_strength": 0.7
  },
  "composite_score": 0.71,
  "acoustic_energy": 0.62,
  "speech_rate_wpm": 138.0,
  "metadata": {
    "hook_caption": "One deed can change your fate forever...",
    "description": "...",
    "hashtags": ["#islamicreminder", "#khutba", ...],
    "thumbnail_suggestion": "...",
    "b_roll_suggestions": [...]
  }
}
```

## Configuration

All settings in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for Claude-based scoring |
| `OPENAI_API_KEY` | — | Required for LLM nomination + metadata generation |
| `WHISPER_MODEL` | `large-v3` | `tiny` for speed, `large-v3` for accuracy |
| `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` on GPU, `int8` on CPU |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Scoring model |
| `OPENAI_MODEL` | `gpt-4o-mini` | Nomination + metadata generation model |
| `MIN_CLIP_DURATION` | `30` | Minimum clip length in seconds |
| `MAX_CLIP_DURATION` | `90` | Maximum clip length in seconds (YouTube Shorts hard limit) |
| `TARGET_CLIP_COUNT` | `5` | Clips to produce per lecture |
| `DEFAULT_DOMAIN` | `khutba` | Default content domain |
| `SEGMENTATION_STRATEGY` | `semantic` | `semantic` or `sliding_window` |
| `OUTPUT_DIR` | `./outputs` | Output directory |
| `CACHE_DIR` | `./cache` | Cache directory for downloads and extracted audio |

## Engagement Scoring

Each candidate segment is rated on 6 axes (all 0.0–1.0) by an LLM:

| Axis | What it measures |
|------|-----------------|
| `semantic_density` | Information per second — no filler |
| `emotional_resonance` | Evokes feeling, urgency, or empathy |
| `standalone_coherence` | A cold viewer can follow without prior context |
| `narrative_completeness` | Opens and closes a thought — no cliffhangers |
| `domain_integrity` | Domain-specific correctness (e.g. hadith attribution for khutbas) |
| `hook_strength` | First 3 seconds can stop a scroll |

Scores are combined into a weighted composite that varies per domain.

## Supported Domains

| Domain | Scoring emphasis |
|--------|-----------------|
| `khutba` | `emotional_resonance` (0.25), `standalone_coherence` (0.20), `domain_integrity` (0.15) — theological accuracy and standalone clarity are critical for TikTok viewers with no sermon context |
| `podcast` | `standalone_coherence` (0.25), `hook_strength` (0.20) — hook is king |
| `business_talk` | `semantic_density` (0.25), `standalone_coherence` (0.20) |
| `lecture` | Balanced weights across all axes |
| `generic` | Default equal weights |

Adding a new domain requires editing 4 constants across existing files — see `docs/architecture.md §9`.

## Architecture

### Pipeline

```
Source (file or URL)
  → [1] Ingestion      yt-dlp download + ffmpeg → 16kHz mono WAV
  → [2] Transcription  faster-whisper → word-level timestamps + hallucination filter
  → [3] Nomination     LLM moment detection and/or embedding-based segmentation
  → [4] Scoring        LLM (6 axes, batched 8 segments/call) + librosa (energy, WPM)
  → [5] Selection      greedy non-overlapping interval scheduling, domain-weighted
  → [6] Generation     word-level subtitles (SRT/ASS) + LLM hook captions + hashtags
  → [7] Export         ffmpeg clip cutting + subtitle burn-in + JSON metadata
  → outputs/clip_NNN/{clip_NNN.mp4, .srt, .ass, _metadata.json}
```

### Data flow

```
TranscriptSegment[] → Transcript
                    → Segment[]
                    → ScoredSegment[]
                    → Clip[]
                    → PipelineResult
```

### Nomination strategies

- **`llm`** — sends timestamped transcript to an LLM, which nominates the most clip-worthy moments directly; snaps to word boundaries and extends to sentence boundaries
- **`semantic`** — embedding cosine similarity between consecutive sentences; new segment where similarity drops below threshold (0.35 English, 0.45 mixed-language)
- **`hybrid`** — runs both and deduplicates overlapping candidates

### Segmentation models

| Model | Use case |
|-------|----------|
| `sentence-transformers/all-MiniLM-L6-v2` | Default (English) |
| `intfloat/multilingual-e5-small` | Arabic/English code-switching (recommended for khutbas) |

### Stream-copy vs. re-encode

| Mode | Speed | Accuracy | Subtitle burn-in |
|------|-------|----------|-----------------|
| Re-encode (default) | ~30s/clip | Frame-accurate | Yes (two-pass) |
| Stream-copy (`--fast-export`) | ~5s/clip | ±2–5s keyframe drift | No |

## Coming Soon

- **Better clip selection** — improved scoring and ranking to surface the most impactful moments
- **Hook title overlay** — auto-generated title card burned into the first 3 seconds of each clip
- **Background vocals for khutbas** — ambient nasheeds mixed under the speech for the khutba domain

## Development

```bash
make install   # pip install -e ".[dev]"
make lint      # ruff + mypy
make test      # pytest with coverage
make clean     # rm -rf outputs/ cache/ .pytest_cache/

# Run a single test file
pytest tests/test_scoring.py -v
```

Tests are **fully mocked** — no API key or GPU required to run them (120 tests).

## Extending

- **New transcriber**: subclass `BaseTranscriber`, implement `transcribe()`.
- **New segmenter**: subclass `BaseSegmenter`, implement `segment()`.
- **New scorer**: subclass `BaseScorer`, implement `score()`.
- **New domain**: add 4 entries across `models.py`, `llm_scorer.py`, `metadata.py`, `selector.py`. See `docs/architecture.md`.
- **Platform uploader**: add a module in `export/` — the `Clip` object has everything needed (video path, SRT, metadata).
