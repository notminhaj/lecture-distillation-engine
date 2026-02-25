"""
CLI entry point.

Usage:
    distill run --input lecture.mp4 --domain khutba
    distill run --input https://youtube.com/watch?v=... --clips 3
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from distillation.models import Domain

app = typer.Typer(name="distill", help="Distill long-form lectures into short-form clips.")
console = Console()


@app.command()
def run(
    input: str = typer.Option(..., "--input", "-i", help="Path to video/audio file or YouTube URL"),
    domain: Domain = typer.Option(Domain.KHUTBA, "--domain", "-d", help="Content domain"),
    clips: int = typer.Option(5, "--clips", "-n", help="Number of clips to generate"),
    no_export: bool = typer.Option(False, "--no-export", help="Skip video export (metadata only)"),
    fast_export: bool = typer.Option(False, "--fast-export", help="Use stream-copy (fast but ~2-5s keyframe drift)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    segmenter: str = typer.Option("semantic", "--segmenter", help="'semantic' or 'sliding_window'"),
    nominator: str = typer.Option("llm", "--nominator", help="'llm' (default), 'semantic' (old behavior), or 'hybrid'"),
) -> None:
    """Run the full distillation pipeline on a lecture."""
    from distillation.config import get_config
    from distillation.pipeline import Pipeline

    config = get_config()
    config.target_clip_count = clips
    config.segmentation_strategy = segmenter
    config.nomination_strategy = nominator
    if output_dir:
        config.output_dir = output_dir

    pipeline = Pipeline(config=config, force_reencode=not fast_export)

    console.print(f"[bold]Running distillation pipeline[/bold]")
    console.print(f"  Input:  {input}")
    console.print(f"  Domain: {domain.value}")
    console.print(f"  Clips:  {clips}")

    result = pipeline.run(input, domain=domain, export_video=not no_export)

    # Display results table
    table = Table(title="Selected Clips", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Duration")
    table.add_column("Score")
    table.add_column("Hook Caption")

    for clip in result.clips:
        table.add_row(
            str(clip.clip_id),
            f"{clip.start:.1f}s",
            f"{clip.end:.1f}s",
            f"{clip.duration:.1f}s",
            f"{clip.scored_segment.composite_score:.2f}",
            (clip.metadata.hook_caption[:60] + "…") if clip.metadata else "—",
        )

    console.print(table)
    console.print(f"\n[green]Done.[/green] Clips written to: {config.output_dir}")


if __name__ == "__main__":
    app()
