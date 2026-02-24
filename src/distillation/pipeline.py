"""
Pipeline orchestration layer.

This is intentionally thin — it wires together the independent stages and
passes data through.  No business logic lives here; that belongs in each
stage module.

Design principle: each stage is a pure function (or stateless class) that
takes an input model and returns an output model.  This makes every stage
independently testable and swappable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from distillation.config import Config, get_config
from distillation.models import Domain, PipelineResult, Transcript

logger = logging.getLogger(__name__)
console = Console()


class Pipeline:
    """
    Orchestrates: Ingestion → Transcription → Segmentation → Scoring
                  → Selection → Generation → Export

    Each stage is injected at construction time, making it trivial to swap
    implementations (e.g., replace WhisperTranscriber with an API-based one)
    without touching this class.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        transcriber=None,
        segmenter=None,
        scorer=None,
        selector=None,
        subtitle_generator=None,
        metadata_generator=None,
        exporter=None,
    ) -> None:
        self.cfg = config or get_config()
        self._transcriber = transcriber
        self._segmenter = segmenter
        self._scorer = scorer
        self._selector = selector
        self._subtitle_gen = subtitle_generator
        self._metadata_gen = metadata_generator
        self._exporter = exporter

    # ── Lazy defaults ──────────────────────────────────────────────────────────

    @property
    def transcriber(self):
        if self._transcriber is None:
            from distillation.transcription.whisper import WhisperTranscriber
            self._transcriber = WhisperTranscriber(self.cfg)
        return self._transcriber

    @property
    def segmenter(self):
        if self._segmenter is None:
            if self.cfg.segmentation_strategy == "semantic":
                from distillation.segmentation.semantic import SemanticSegmenter
                self._segmenter = SemanticSegmenter(self.cfg)
            else:
                from distillation.segmentation.sliding_window import SlidingWindowSegmenter
                self._segmenter = SlidingWindowSegmenter(self.cfg)
        return self._segmenter

    @property
    def scorer(self):
        if self._scorer is None:
            from distillation.scoring.llm_scorer import LLMScorer
            self._scorer = LLMScorer(self.cfg)
        return self._scorer

    @property
    def selector(self):
        if self._selector is None:
            from distillation.selection.selector import ClipSelector
            self._selector = ClipSelector(self.cfg)
        return self._selector

    @property
    def subtitle_gen(self):
        if self._subtitle_gen is None:
            from distillation.generation.subtitles import SubtitleGenerator
            self._subtitle_gen = SubtitleGenerator()
        return self._subtitle_gen

    @property
    def metadata_gen(self):
        if self._metadata_gen is None:
            from distillation.generation.metadata import MetadataGenerator
            self._metadata_gen = MetadataGenerator(self.cfg)
        return self._metadata_gen

    @property
    def exporter(self):
        if self._exporter is None:
            from distillation.export.video import VideoExporter
            self._exporter = VideoExporter(self.cfg)
        return self._exporter

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        source: str | Path,
        domain: Domain | None = None,
        audio_path: str | Path | None = None,
        export_video: bool = True,
    ) -> PipelineResult:
        """
        Run the full pipeline on a source file or URL.

        Args:
            source:       Path to video/audio file, or a YouTube URL.
            domain:       Content domain; defaults to config.default_domain.
            audio_path:   Pre-extracted audio path to skip ingestion (useful
                          for re-runs where audio already exists).
            export_video: Whether to cut video clips at the end.

        Returns:
            PipelineResult with all intermediate data and final clips.
        """
        domain = domain or self.cfg.default_domain
        source = str(source)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:

            # ── Stage 1: Ingestion ─────────────────────────────────────────────
            task = progress.add_task("Ingesting media...", total=None)
            if audio_path is None:
                audio_path = self._ingest(source)
            progress.update(task, completed=True)

            # ── Stage 2: Transcription ─────────────────────────────────────────
            progress.update(task, description="Transcribing (Whisper)...")
            transcript: Transcript = self.transcriber.transcribe(audio_path)
            logger.info(
                "Transcribed %d segments, %.0fs audio",
                len(transcript.segments),
                transcript.duration,
            )

            # ── Stage 3: Segmentation ──────────────────────────────────────────
            progress.update(task, description="Segmenting into semantic units...")
            segments = self.segmenter.segment(transcript)
            logger.info("Produced %d segments", len(segments))

            # ── Stage 4: Scoring ───────────────────────────────────────────────
            progress.update(task, description="Scoring segments (LLM + acoustic)...")
            scored = self.scorer.score(segments, transcript, audio_path=str(audio_path), domain=domain)

            # ── Stage 5: Selection ─────────────────────────────────────────────
            progress.update(task, description="Selecting top clips...")
            selected = self.selector.select(scored, domain=domain)

            # ── Stage 6: Subtitle generation ──────────────────────────────────
            progress.update(task, description="Generating subtitles...")
            for clip in selected:
                clip.subtitles = self.subtitle_gen.generate(clip, transcript)

            # ── Stage 7: Metadata generation ──────────────────────────────────
            progress.update(task, description="Generating captions and metadata...")
            for clip in selected:
                clip.metadata = self.metadata_gen.generate(clip, domain=domain)

            # ── Stage 8: Export ────────────────────────────────────────────────
            if export_video:
                progress.update(task, description="Cutting clips...")
                for clip in selected:
                    self.exporter.export(clip, source_video=source)

        return PipelineResult(
            source_path=source,
            domain=domain,
            transcript=transcript,
            segments=segments,
            scored_segments=scored,
            clips=selected,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _ingest(self, source: str) -> Path:
        """Download (if URL) and extract audio."""
        if source.startswith("http://") or source.startswith("https://"):
            from distillation.ingestion.downloader import Downloader
            dl = Downloader(self.cfg)
            video_path = dl.download(source)
        else:
            video_path = Path(source)

        from distillation.ingestion.extractor import AudioExtractor
        extractor = AudioExtractor(self.cfg)
        return extractor.extract(video_path)
