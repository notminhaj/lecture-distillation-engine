# Lecture Distillation Engine

> **Work in progress** — this is an early prototype. Core functionality works end-to-end, but clip quality and output polish are still being actively improved.

Transforms long-form lectures (initially Islamic khutbas/sermons) into
platform-ready short-form clips for TikTok, Instagram Reels, and YouTube Shorts.

## What it does

Given a lecture video or YouTube URL, the engine runs a 7-stage pipeline:

1. **Ingests** — downloads via yt-dlp (or reads local file), extracts 16 kHz mono WAV for Whisper
2. **Transcribes** — word-level timestamps with Whisper large-v3, hallucination filtering, language detection
3. **Segments / Nominates** — finds candidate clip boundaries via embedding-based topic detection, LLM moment detection, or both; boundaries are refined via 4-pass post-processing (sentence snapping, pause detection, hook protection, duration enforcement)
4. **Scores** — rates each candidate on 6 engagement axes using an LLM + acoustic features (librosa); LLM receives a sliding ±200-word context window around each batch for coherence judgment
5. **Selects** — greedy non-overlapping interval scheduling, domain-weighted; applies completeness gate, acoustic-onset hook boost, and structural penalty; merges adjacent incomplete segments
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
| `GEMINI_API_KEY` | — | Optional; for Gemini-based scoring |
| `WHISPER_MODEL` | `large-v3` | `tiny` for speed, `large-v3` for accuracy |
| `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` on GPU, `int8` on CPU |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Scoring model |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini scoring model |
| `OPENAI_MODEL` | `gpt-4o-mini` | Nomination + metadata generation model |
| `MIN_CLIP_DURATION` | `30` | Minimum clip length in seconds |
| `MAX_CLIP_DURATION` | `90` | Maximum clip length in seconds (YouTube Shorts hard limit) |
| `TARGET_CLIP_COUNT` | `5` | Clips to produce per lecture |
| `MIN_NARRATIVE_COMPLETENESS` | `0.5` | Segments below this score are excluded before ranking |
| `DEFAULT_DOMAIN` | `khutba` | Default content domain |
| `SEGMENTATION_STRATEGY` | `semantic` | `semantic` or `sliding_window` |
| `ENABLE_BOUNDARY_REFINEMENT` | `true` | Toggle 4-pass boundary refinement after segmentation |
| `SENTENCE_BOUNDARY_WINDOW` | `5.0` | ±seconds to search for sentence boundary when snapping |
| `SILENCE_THRESHOLD_MS` | `600` | Minimum silence gap (ms) preferred as a clip boundary |
| `HOOK_PROTECTION_WINDOW` | `3.0` | Seconds at segment start checked for hook energy |
| `COOKIES_FROM_BROWSER` | — | Browser to pull cookies from (`chrome`, `firefox`, `edge`) |
| `COOKIES_FILE` | — | Path to a Netscape-format `cookies.txt` for yt-dlp |
| `OUTPUT_DIR` | `./outputs` | Output directory |
| `CACHE_DIR` | `./cache` | Cache directory for downloads and extracted audio |

### YouTube authentication (Windows)

Chrome v127+ uses app-bound cookie encryption that yt-dlp cannot read directly.
The downloader handles this automatically:

1. If `COOKIES_FILE` is set and the file exists, it is used as-is.
2. On bot-detection errors, the downloader auto-extracts cookies via a headless
   browser (Edge first, then Chrome) using Selenium and writes
   `cache/cookies.txt`.
3. Alternatively, set `COOKIES_FROM_BROWSER=edge` (or `chrome`) to let yt-dlp
   extract cookies itself — works on Linux/macOS but may fail on Windows with
   Chrome v127+.

To pre-generate a `cookies.txt` manually:

```bash
python scripts/export_cookies.py          # uses headless Edge/Chrome
```

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
A sliding ±200-word context window surrounds each scoring batch so the LLM can
accurately judge coherence for segments deep in the lecture.

Two signal augmentations are applied before ranking:

- **Acoustic-onset boost**: `energy_onset_ratio` (first-3s RMS ÷ segment average)
  multiplies `hook_strength` by `[0.7, 1.3]` — boosting strong acoustic opens,
  penalising silence or throat-clearing at the start.
- **Structural penalty**: segments that start mid-sentence or lack terminal
  punctuation receive up to a 0.30 reduction on the `narrative_completeness`
  component, grounding structural completeness in text evidence.

## Supported Domains

| Domain | Scoring emphasis |
|--------|-----------------|
| `khutba` | `standalone_coherence` (0.25), `narrative_completeness` (0.22), `emotional_resonance` (0.15), `domain_integrity` (0.15) — cold-viewer comprehension and narrative closure are critical; incomplete rulings are harmful |
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
                       └─ BoundaryRefiner → sentence snap → pause snap → hook protect → duration enforce
  → [4] Scoring        LLM (6 axes, batched 8 segments/call, sliding context) + librosa (energy, WPM)
  → [5] Selection      completeness gate → merge candidates → greedy non-overlapping, domain-weighted
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

### Boundary refinement

After nomination, `BoundaryRefiner` runs 4 ordered passes on raw segments:

1. **Sentence snap** — moves each boundary to the nearest `TranscriptSegment` edge within ±`SENTENCE_BOUNDARY_WINDOW`
2. **Pause snap** — shifts the boundary between adjacent segments to the largest inter-word silence gap ≥ `SILENCE_THRESHOLD_MS`
3. **Hook protection** — if a segment's opening words flow continuously from the previous segment, shifts the boundary back to preserve the hook
4. **Duration enforcement** — merges short segments and splits long ones to stay within `[MIN_CLIP_DURATION, MAX_CLIP_DURATION]`

Toggle via `ENABLE_BOUNDARY_REFINEMENT` (default: `true`).

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

- **Hook title overlay** — auto-generated title card burned into the first 3 seconds of each clip
- **Background vocals for khutbas** — ambient nasheeds mixed under the speech for the khutba domain
- **LLM re-score of merged candidates** — send heuristically merged adjacent segments back for a real LLM scoring pass instead of interpolated scores

## Development

```bash
make install   # pip install -e ".[dev]"
make lint      # ruff + mypy
make test      # pytest with coverage
make clean     # rm -rf outputs/ cache/ .pytest_cache/

# Run a single test file
pytest tests/test_scoring.py -v
```

Tests are **fully mocked** — no API key or GPU required to run them.

## Extending

- **New transcriber**: subclass `BaseTranscriber`, implement `transcribe()`.
- **New segmenter**: subclass `BaseSegmenter`, implement `segment()`.
- **New scorer**: subclass `BaseScorer`, implement `score()`.
- **New domain**: add 4 entries across `models.py`, `llm_scorer.py`, `metadata.py`, `selector.py`. See `docs/architecture.md`.
- **Platform uploader**: add a module in `export/` — the `Clip` object has everything needed (video path, SRT, metadata).
