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
        moment_detector=None,
        scorer=None,
        selector=None,
        subtitle_generator=None,
        metadata_generator=None,
        exporter=None,
        force_reencode: bool = True,
    ) -> None:
        self.cfg = config or get_config()
        self._transcriber = transcriber
        self._segmenter = segmenter
        self._moment_detector = moment_detector
        self._scorer = scorer
        self._selector = selector
        self._subtitle_gen = subtitle_generator
        self._metadata_gen = metadata_generator
        self._exporter = exporter
        self._force_reencode = force_reencode

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
    def moment_detector(self):
        if self._moment_detector is None:
            from distillation.nomination.moment_detector import MomentDetector
            self._moment_detector = MomentDetector(self.cfg)
        return self._moment_detector

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
    def word_boundary_refiner(self):
        from distillation.word_boundary_refiner import WordBoundaryRefiner
        return WordBoundaryRefiner()

    @property
    def exporter(self):
        if self._exporter is None:
            from distillation.export.video import VideoExporter
            self._exporter = VideoExporter(self.cfg, force_reencode=self._force_reencode)
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
            video_path = Path(source)
            if audio_path is None:
                video_path, audio_path = self._ingest(source)
            progress.update(task, completed=True)

            # ── Stage 2: Transcription ─────────────────────────────────────────
            progress.update(task, description="Transcribing (Whisper)...")
            transcript: Transcript = self.transcriber.transcribe(audio_path)
            logger.info(
                "Transcribed %d segments, %.0fs audio",
                len(transcript.segments),
                transcript.duration,
            )

            # ── Stage 3: Candidate generation (nomination / segmentation) ─────
            strategy = self.cfg.nomination_strategy
            segments = []

            if strategy in ("llm", "hybrid"):
                progress.update(task, description="Detecting clip-worthy moments (LLM)...")
                nominated = self.moment_detector.detect(
                    transcript,
                    domain=domain,
                    target_count=self.cfg.target_clip_count * 2,
                )
                segments.extend(nominated)
                logger.info("LLM nominated %d candidate segments", len(nominated))

            if strategy in ("semantic", "hybrid"):
                progress.update(task, description="Segmenting into semantic units...")
                semantic_segments = self.segmenter.segment(transcript)
                if strategy == "hybrid":
                    # Deduplicate: only add semantic segments that don't heavily
                    # overlap with already-nominated segments
                    for ss in semantic_segments:
                        if not self._overlaps_existing(ss, segments):
                            segments.append(ss)
                else:
                    segments = semantic_segments
                logger.info("Semantic segmenter produced %d segments", len(semantic_segments))

            # Re-number segment IDs to be sequential
            for i, seg in enumerate(segments):
                seg.segment_id = i

            logger.info("Total candidate segments: %d", len(segments))

            # ── Stage 4: Scoring ───────────────────────────────────────────────
            progress.update(task, description="Scoring segments (LLM + acoustic)...")
            scored = self.scorer.score(segments, transcript, audio_path=str(audio_path), domain=domain)

            # ── Stage 5: Selection ─────────────────────────────────────────────
            progress.update(task, description="Selecting top clips...")
            selected = self.selector.select(scored, domain=domain)

            # ── Stage 5b: Word boundary refinement ───────────────────────
            if self.cfg.enable_word_boundary_refinement:
                progress.update(task, description="Refining clip word boundaries...")
                selected = self.word_boundary_refiner.refine_all(selected, transcript)

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
                    try:
                        self.exporter.export(clip, source_video=str(video_path))
                    except Exception as exc:
                        logger.error(
                            "Export failed for clip %d: %s — skipping clip, others will proceed.",
                            clip.clip_id, exc,
                        )

        return PipelineResult(
            source_path=source,
            domain=domain,
            transcript=transcript,
            segments=segments,
            scored_segments=scored,
            clips=selected,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _overlaps_existing(candidate, existing: list, threshold: float = 0.5) -> bool:
        """Check if candidate segment overlaps significantly with any existing segment."""
        from distillation.models import Segment
        for seg in existing:
            overlap_start = max(candidate.start, seg.start)
            overlap_end = min(candidate.end, seg.end)
            overlap = max(0, overlap_end - overlap_start)
            shorter_dur = min(candidate.duration, seg.duration)
            if shorter_dur > 0 and overlap / shorter_dur > threshold:
                return True
        return False

    def _ingest(self, source: str) -> tuple[Path, Path]:
        """Download (if URL) and extract audio.  Returns (video_path, audio_path)."""
        if source.startswith("http://") or source.startswith("https://"):
            from distillation.ingestion.downloader import Downloader
            dl = Downloader(self.cfg)
            video_path = dl.download(source)
        else:
            video_path = Path(source)

        from distillation.ingestion.extractor import AudioExtractor
        extractor = AudioExtractor(self.cfg)
        audio_path = extractor.extract(video_path)
        return video_path, audio_path
