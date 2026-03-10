"""
Nomination Dry-Run Evaluation Script
=====================================
Evaluates MomentDetector nomination quality without running the full pipeline.
Calls only the LLM nomination step (no FFmpeg, no subtitles, no export).

Usage:
    python scripts/eval_nominations.py \\
        --transcript path/to/transcript.json \\
        --domain khutba \\
        --output reports/eval-nominations-2026-03-11.md

How to generate a transcript JSON from the pipeline cache:
    Run the pipeline once with at least the ingestion + transcription stages.
    The transcript is saved to: outputs/<clip_id>/transcript.json (if pipeline
    is configured to persist intermediate outputs). Alternatively, use the
    fixture at: scripts/fixtures/sample_transcript.json for testing.

Quality flags:
    PASS - Opening line starts capitalized, not a continuation word
    WARN - Opening line OK but duration < 20s (risk of being too short)
    FAIL - Opening line starts lowercase or with a continuation word

Requirements:
    OPENAI_API_KEY must be set in environment or .env file.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src/ to path so we can import distillation modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.config import Config
from distillation.models import Domain, Language, NominatedMoment, Transcript, TranscriptSegment
from distillation.nomination.moment_detector import MomentDetector


# Continuation words that indicate an opening mid-sentence or mid-thought
_CONTINUATION_WORDS = frozenset({
    "and", "but", "so", "or", "yet", "like", "for", "that", "however",
    "therefore", "then", "now", "also", "still", "even", "well", "you",
})


def _quality_flag(nom: NominatedMoment) -> tuple[str, str]:
    """Return (flag, reason) for a nominated moment."""
    opening = nom.opening_line.strip()
    if not opening:
        return "FAIL", "opening_line is empty"

    first_word = opening.split()[0].rstrip(".,!?;:")

    if first_word[0].islower():
        return "FAIL", f"opening_line starts lowercase: '{first_word}'"

    if first_word.lower() in _CONTINUATION_WORDS:
        return "FAIL", f"opening_line starts with continuation word: '{first_word}'"

    duration = nom.approximate_end - nom.approximate_start
    if duration < 20.0:
        return "WARN", f"short duration: {duration:.1f}s (< 20s threshold)"

    return "PASS", "clean sentence start, acceptable duration"


def _load_transcript(path: Path) -> Transcript:
    """Load a Transcript from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return Transcript.model_validate(data)


def _run_nominations(
    transcript: Transcript,
    domain: Domain,
    target_count: int,
    config: Config,
) -> list[NominatedMoment]:
    """Run only the LLM nomination step, returning NominatedMoment objects."""
    detector = MomentDetector(config)
    stamped_text = detector._build_timestamped_text(transcript)
    chunks = detector._chunk_text(stamped_text, 6000)

    all_nominations: list[NominatedMoment] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Nominating chunk {i}/{len(chunks)}...", flush=True)
        nominations = detector._nominate_chunk(chunk, domain, target_count=target_count)
        all_nominations.extend(nominations)

    return all_nominations


def _build_report(
    nominations: list[NominatedMoment],
    domain: Domain,
    transcript_path: Path,
    target_count: int,
) -> str:
    """Build a markdown evaluation report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Nomination Evaluation Report",
        f"",
        f"- **Date:** {now}",
        f"- **Transcript:** `{transcript_path}`",
        f"- **Domain:** `{domain.value}`",
        f"- **Target nominations:** {target_count}",
        f"- **Total nominated:** {len(nominations)}",
        f"",
    ]

    # Summary counts
    flags = [_quality_flag(n)[0] for n in nominations]
    pass_count = flags.count("PASS")
    warn_count = flags.count("WARN")
    fail_count = flags.count("FAIL")

    lines += [
        f"## Summary",
        f"",
        f"| Flag | Count |",
        f"|------|-------|",
        f"| PASS | {pass_count} |",
        f"| WARN | {warn_count} |",
        f"| FAIL | {fail_count} |",
        f"",
    ]

    if not nominations:
        lines.append("_No nominations returned from LLM._")
        return "\n".join(lines)

    lines.append("## Nominations")
    lines.append("")

    for i, nom in enumerate(nominations, 1):
        flag, reason = _quality_flag(nom)
        duration = nom.approximate_end - nom.approximate_start
        flag_emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[flag]

        lines += [
            f"### {i}. {flag_emoji} {flag} — {nom.opening_line[:60]}{'...' if len(nom.opening_line) > 60 else ''}",
            f"",
            f"- **Time:** {nom.approximate_start:.1f}s – {nom.approximate_end:.1f}s ({duration:.1f}s)",
            f"- **Opening line:** _{nom.opening_line}_",
            f"- **Quality flag:** `{flag}` — {reason}",
            f"- **Rationale:** {nom.rationale}",
            f"",
        ]

    lines += [
        "---",
        f"_Generated by scripts/eval_nominations.py_",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run nomination evaluation — no FFmpeg, no subtitles required."
    )
    parser.add_argument(
        "--transcript", required=True, type=Path,
        help="Path to Whisper transcript JSON (Transcript model format)"
    )
    parser.add_argument(
        "--domain", default="khutba",
        choices=[d.value for d in Domain],
        help="Content domain for prompt tuning (default: khutba)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for markdown report (default: prints to stdout)"
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="Target number of nominations to request from LLM (default: 10)"
    )
    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"ERROR: transcript file not found: {args.transcript}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading transcript: {args.transcript}", flush=True)
    transcript = _load_transcript(args.transcript)
    print(f"  {len(transcript.segments)} segments, {transcript.duration:.1f}s total", flush=True)

    domain = Domain(args.domain)
    config = Config()

    if not config.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    print(f"Running LLM nominations (domain={domain.value}, target={args.count})...", flush=True)
    nominations = _run_nominations(transcript, domain, args.count, config)
    print(f"  Got {len(nominations)} nominations.", flush=True)

    report = _build_report(nominations, domain, args.transcript, args.count)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}", flush=True)
    else:
        print("\n" + report)

    # Exit summary
    flags = [_quality_flag(n)[0] for n in nominations]
    fail_count = flags.count("FAIL")
    pass_count = flags.count("PASS")
    print(f"\nResult: {pass_count} PASS, {flags.count('WARN')} WARN, {fail_count} FAIL", flush=True)
    if fail_count > 0:
        print(f"WARNING: {fail_count}/{len(nominations)} nominations failed sentence-boundary check.", flush=True)


if __name__ == "__main__":
    main()
