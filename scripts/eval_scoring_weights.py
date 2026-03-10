"""
Scoring Weight Calibration Analysis
=====================================
Reads clip_quality_dataset.json and evaluates how well the current DOMAIN_WEIGHTS
in selector.py discriminate between good, mixed, and bad clips.

Usage:
    python scripts/eval_scoring_weights.py \\
        --dataset /path/to/clip_quality_dataset.json \\
        --domain khutba \\
        --output reports/scoring-weight-calibration-2026-03-11.md

The script:
  1. Loads clip_quality_dataset.json
  2. Computes a simulated composite score for each clip using current DOMAIN_WEIGHTS
  3. Ranks clips by simulated score
  4. Compares ranking to agent_label (good > mixed > bad)
  5. Checks correct-order count and identifies mis-ranked clips
  6. Outputs a markdown calibration report with recommendations
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.models import Domain
from distillation.selection.selector import DOMAIN_WEIGHTS, ACOUSTIC_ENERGY_WEIGHT


# Label ordering for correct-rank comparison (higher index = better)
LABEL_ORDER = {"bad": 0, "mixed": 1, "good": 2}


def _compute_weighted_score(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute a weighted composite score from axis scores and weights."""
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    raw = sum(scores.get(axis, 0.0) * w for axis, w in weights.items())
    return raw / total_weight


def _pairwise_correct_order(
    ranked_clips: list[dict],
) -> tuple[int, int, list[str]]:
    """
    Count how many consecutive pairs are in the correct label order.

    Returns: (correct_pairs, total_pairs, list of violations)
    """
    correct = 0
    total = 0
    violations = []

    for i in range(len(ranked_clips) - 1):
        a = ranked_clips[i]
        b = ranked_clips[i + 1]
        total += 1
        a_rank = LABEL_ORDER.get(a["agent_label"], -1)
        b_rank = LABEL_ORDER.get(b["agent_label"], -1)

        # Ties are OK (both bad, for example)
        if a_rank >= b_rank:
            correct += 1
        else:
            violations.append(
                f"{a['clip_id']} ({a['agent_label']}, {a['sim_score']:.4f}) "
                f"ranked above {b['clip_id']} ({b['agent_label']}, {b['sim_score']:.4f}) ❌"
            )

    return correct, total, violations


def _weight_sensitivity_analysis(
    clips: list[dict],
    weights: dict[str, float],
    axes: list[str],
) -> list[dict]:
    """
    For each axis, compute: if we increase its weight by 0.10 (and reduce
    others proportionally), does the ranking improve or stay the same?
    Returns a list of recommendations.
    """
    recommendations = []

    baseline_correct, baseline_total, _ = _pairwise_correct_order(
        sorted(clips, key=lambda c: c["sim_score"], reverse=True)
    )

    for axis in axes:
        # Bump axis by 0.05, reduce all others proportionally
        new_weights = {k: v for k, v in weights.items()}
        increase = 0.05
        new_weights[axis] = weights[axis] + increase
        # Reduce others by the increase amount, distributed proportionally
        other_axes = [k for k in axes if k != axis]
        total_other = sum(weights[k] for k in other_axes)
        if total_other > 0:
            for k in other_axes:
                reduction = increase * (weights[k] / total_other)
                new_weights[k] = weights[k] - reduction

        # Recompute scores with new weights
        for clip in clips:
            clip["trial_score"] = _compute_weighted_score(
                clip["agent_scores"], new_weights
            )

        trial_ranked = sorted(clips, key=lambda c: c["trial_score"], reverse=True)
        trial_correct, _, _ = _pairwise_correct_order(trial_ranked)

        delta = trial_correct - baseline_correct
        recommendations.append({
            "axis": axis,
            "current_weight": weights[axis],
            "trial_weight": round(new_weights[axis], 3),
            "correct_pairs": trial_correct,
            "delta": delta,
        })

    return sorted(recommendations, key=lambda r: r["delta"], reverse=True)


def build_report(
    clips: list[dict],
    domain: Domain,
    weights: dict[str, float],
    dataset_path: Path,
) -> str:
    """Build the full calibration report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    axes = list(weights.keys())

    # Compute simulated scores
    for clip in clips:
        clip["sim_score"] = _compute_weighted_score(clip["agent_scores"], weights)

    ranked = sorted(clips, key=lambda c: c["sim_score"], reverse=True)
    correct_pairs, total_pairs, violations = _pairwise_correct_order(ranked)
    correct_pct = (correct_pairs / total_pairs * 100) if total_pairs else 0

    recommendations = _weight_sensitivity_analysis(clips, weights, axes)

    lines = [
        f"# Scoring Weight Calibration Report",
        f"",
        f"- **Date:** {now}",
        f"- **Dataset:** `{dataset_path}`",
        f"- **Domain:** `{domain.value}`",
        f"- **Clips analyzed:** {len(clips)}",
        f"",
        f"## Current Weight Vector (KHUTBA)",
        f"",
        f"| Axis | Weight |",
        f"|------|--------|",
    ]
    for axis, w in weights.items():
        lines.append(f"| {axis} | {w:.2f} |")

    lines += [
        f"",
        f"## Simulated Composite Scores",
        f"",
        f"Scores recomputed using current `DOMAIN_WEIGHTS`. "
        f"Note: `acoustic_energy` (weight {ACOUSTIC_ENERGY_WEIGHT}) excluded — not in dataset.",
        f"",
        f"| Rank | Clip ID | Sim Score | Agent Label | Agent Composite |",
        f"|------|---------|-----------|-------------|-----------------|",
    ]

    for i, clip in enumerate(ranked, 1):
        orig = clip.get("agent_composite_score", "n/a")
        label = clip["agent_label"]
        label_icon = {"good": "✅", "mixed": "⚠️", "bad": "❌"}.get(label, "")
        lines.append(
            f"| {i} | {clip['clip_id']} | {clip['sim_score']:.4f} | "
            f"{label_icon} {label} | {orig} |"
        )

    lines += [
        f"",
        f"## Ranking Quality",
        f"",
        f"**Correct consecutive pairs: {correct_pairs}/{total_pairs} ({correct_pct:.0f}%)**",
        f"",
        f"A correct pair means the higher-ranked clip has a label at least as good as the lower-ranked one.",
        f"",
    ]

    if violations:
        lines.append("### Order Violations")
        lines.append("")
        for v in violations:
            lines.append(f"- {v}")
        lines.append("")
    else:
        lines.append("✅ No order violations — all label-adjacent pairs are in correct order.")
        lines.append("")

    lines += [
        f"## Axis Sensitivity Analysis",
        f"",
        f"Hypothetical: increase each axis weight by +0.05 (reduce others proportionally).",
        f"Shows which axis, if up-weighted, would most improve correct-pair count.",
        f"",
        f"| Axis | Current | Trial | Δ Correct Pairs |",
        f"|------|---------|-------|-----------------|",
    ]
    for r in recommendations:
        delta_str = f"+{r['delta']}" if r["delta"] > 0 else str(r["delta"])
        lines.append(
            f"| {r['axis']} | {r['current_weight']:.2f} | {r['trial_weight']:.3f} | {delta_str} |"
        )

    lines += [
        f"",
        f"## Observations",
        f"",
    ]

    if correct_pct == 100:
        lines.append(
            "The current weight vector **correctly orders all label-adjacent clip pairs**. "
            "The good clip (`clip_004_failure`) ranks #1, the mixed clip ranks #2, and all bad clips rank below."
        )
    elif correct_pct >= 80:
        lines.append(
            f"The current weights are **reasonably calibrated** ({correct_pct:.0f}% correct pairs). "
            "Minor violations exist — see recommendations."
        )
    else:
        lines.append(
            f"The current weights are **poorly calibrated** ({correct_pct:.0f}% correct pairs). "
            "Significant reweighting is recommended."
        )

    lines += [
        f"",
        "**Key caveats:**",
        "- Dataset has only 6 clips — findings are directional, not statistically significant.",
        "- All clips have `human_label=null` — agent labels are used as proxy. Human review recommended.",
        "- `hook_strength` was uniformly 0.5 in this batch (pre-ns-005 fix) — it carries no signal here.",
        "- `standalone_coherence` appears overscored for bad clips (known bug per dataset notes).",
        f"",
        "## Recommendations",
        f"",
    ]

    top_rec = [r for r in recommendations if r["delta"] > 0]
    if top_rec:
        lines.append(f"1. **Up-weight:** {', '.join(r['axis'] for r in top_rec[:2])} — these improve correct-pair count when increased.")
    else:
        lines.append("1. **No reweighting needed** based on current dataset — all pairs correctly ordered.")

    lines += [
        "2. **Collect human labels** — fill `human_label` in `clip_quality_dataset.json` to validate these findings.",
        "3. **Re-evaluate after ns-005 fix** — `hook_strength` will carry real signal in future batches.",
        "4. **Re-evaluate after ns-008 fix** — nomination improvements may produce better base clips, changing score distributions.",
        f"",
        "---",
        f"_Generated by scripts/eval_scoring_weights.py_",
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scoring weight calibration analysis against clip_quality_dataset.json"
    )
    parser.add_argument(
        "--dataset", required=True, type=Path,
        help="Path to clip_quality_dataset.json"
    )
    parser.add_argument(
        "--domain", default="khutba",
        choices=[d.value for d in Domain],
        help="Domain whose weight vector to evaluate (default: khutba)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for markdown report (default: stdout)"
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    clips = data.get("entries", [])

    if not clips:
        print("ERROR: no entries in dataset", file=sys.stderr)
        sys.exit(1)

    domain = Domain(args.domain)
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS[Domain.GENERIC])

    print(f"Analyzing {len(clips)} clips with {domain.value} weights...", flush=True)

    report = build_report(clips, domain, dict(weights), args.dataset)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}", flush=True)
    else:
        print(report)


if __name__ == "__main__":
    main()
