"""Tests for the pipeline orchestration — all stages are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from distillation.models import (
    Clip,
    ClipMetadata,
    Domain,
    EngagementAxes,
    Language,
    Segment,
    ScoredSegment,
    SubtitleLine,
    Transcript,
    TranscriptSegment,
)
from distillation.pipeline import Pipeline


def _make_transcript() -> Transcript:
    return Transcript(
        source_path="/tmp/test.wav",
        duration=60.0,
        language=Language.EN,
        segments=[
            TranscriptSegment(id=0, text="First segment.", start=0.0, end=20.0, words=[]),
            TranscriptSegment(id=1, text="Second segment.", start=20.0, end=40.0, words=[]),
            TranscriptSegment(id=2, text="Third segment.", start=40.0, end=60.0, words=[]),
        ],
    )


def _make_segments() -> list[Segment]:
    return [
        Segment(segment_id=0, start=0.0, end=30.0, text="First segment.", transcript_segment_ids=[0]),
        Segment(segment_id=1, start=30.0, end=60.0, text="Second segment.", transcript_segment_ids=[1]),
    ]


def _make_scored_segments(segments: list[Segment]) -> list[ScoredSegment]:
    axes = EngagementAxes(
        semantic_density=0.8, emotional_resonance=0.7, standalone_coherence=0.9,
        narrative_completeness=0.85, domain_integrity=0.95, hook_strength=0.6,
    )
    return [ScoredSegment(segment=s, scores=axes) for s in segments]


def _make_clips(scored: list[ScoredSegment]) -> list[Clip]:
    return [
        Clip(clip_id=i, source_path="/tmp/source.mp4", start=ss.segment.start,
             end=ss.segment.end, scored_segment=ss)
        for i, ss in enumerate(scored)
    ]


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.default_domain = Domain.KHUTBA
    cfg.nomination_strategy = "semantic"
    cfg.segmentation_strategy = "semantic"
    cfg.target_clip_count = 2
    cfg.min_clip_duration = 15.0
    cfg.max_clip_duration = 90.0
    cfg.output_dir = MagicMock()
    return cfg


@pytest.fixture
def mock_stages():
    transcript = _make_transcript()
    segments = _make_segments()
    scored = _make_scored_segments(segments)
    clips = _make_clips(scored)

    transcriber = MagicMock()
    transcriber.transcribe.return_value = transcript

    segmenter = MagicMock()
    segmenter.segment.return_value = segments

    moment_detector = MagicMock()
    moment_detector.detect.return_value = segments

    scorer = MagicMock()
    scorer.score.return_value = scored

    selector = MagicMock()
    selector.select.return_value = clips

    subtitle_gen = MagicMock()
    subtitle_gen.generate.return_value = [SubtitleLine(index=1, start=0.0, end=2.0, text="Sub")]

    metadata_gen = MagicMock()
    metadata_gen.generate.return_value = ClipMetadata(
        hook_caption="Hook", description="Desc", hashtags=["#test"],
        thumbnail_suggestion="Thumb",
    )

    exporter = MagicMock()

    return {
        "transcriber": transcriber,
        "segmenter": segmenter,
        "moment_detector": moment_detector,
        "scorer": scorer,
        "selector": selector,
        "subtitle_gen": subtitle_gen,
        "metadata_gen": metadata_gen,
        "exporter": exporter,
        "transcript": transcript,
        "segments": segments,
        "scored": scored,
        "clips": clips,
    }


class TestPipelineSemanticStrategy:
    def test_run_returns_pipeline_result(self, config, mock_stages):
        config.nomination_strategy = "semantic"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        result = pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        assert result.domain == Domain.KHUTBA
        assert len(result.clips) == 2
        assert result.transcript.duration == 60.0

    def test_semantic_strategy_calls_segmenter(self, config, mock_stages):
        config.nomination_strategy = "semantic"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        mock_stages["segmenter"].segment.assert_called_once()

    def test_semantic_strategy_does_not_call_moment_detector(self, config, mock_stages):
        config.nomination_strategy = "semantic"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            segmenter=mock_stages["segmenter"],
            moment_detector=mock_stages["moment_detector"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        mock_stages["moment_detector"].detect.assert_not_called()


class TestPipelineLLMStrategy:
    def test_llm_strategy_calls_moment_detector(self, config, mock_stages):
        config.nomination_strategy = "llm"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            moment_detector=mock_stages["moment_detector"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        mock_stages["moment_detector"].detect.assert_called_once()

    def test_llm_strategy_does_not_call_segmenter(self, config, mock_stages):
        config.nomination_strategy = "llm"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            moment_detector=mock_stages["moment_detector"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        mock_stages["segmenter"].segment.assert_not_called()


class TestPipelineHybridStrategy:
    def test_hybrid_calls_both(self, config, mock_stages):
        config.nomination_strategy = "hybrid"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            moment_detector=mock_stages["moment_detector"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        mock_stages["moment_detector"].detect.assert_called_once()
        mock_stages["segmenter"].segment.assert_called_once()


class TestPipelineExport:
    def test_export_called_when_enabled(self, config, mock_stages):
        config.nomination_strategy = "semantic"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=True)

        assert mock_stages["exporter"].export.call_count == 2

    def test_export_skipped_when_disabled(self, config, mock_stages):
        config.nomination_strategy = "semantic"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        mock_stages["exporter"].export.assert_not_called()


class TestPipelineIngestion:
    def test_audio_path_skips_ingestion(self, config, mock_stages):
        config.nomination_strategy = "semantic"
        pipeline = Pipeline(
            config=config,
            transcriber=mock_stages["transcriber"],
            segmenter=mock_stages["segmenter"],
            scorer=mock_stages["scorer"],
            selector=mock_stages["selector"],
            subtitle_generator=mock_stages["subtitle_gen"],
            metadata_generator=mock_stages["metadata_gen"],
            exporter=mock_stages["exporter"],
        )

        result = pipeline.run("/tmp/source.mp4", audio_path="/tmp/test.wav", export_video=False)

        # Transcriber should receive the provided audio path
        mock_stages["transcriber"].transcribe.assert_called_once_with("/tmp/test.wav")


class TestPipelineOverlapHelper:
    def test_no_overlap(self):
        a = Segment(segment_id=0, start=0.0, end=10.0, text="A", transcript_segment_ids=[0])
        b = Segment(segment_id=1, start=20.0, end=30.0, text="B", transcript_segment_ids=[1])
        assert Pipeline._overlaps_existing(a, [b]) is False

    def test_full_overlap(self):
        a = Segment(segment_id=0, start=0.0, end=10.0, text="A", transcript_segment_ids=[0])
        b = Segment(segment_id=1, start=0.0, end=10.0, text="B", transcript_segment_ids=[1])
        assert Pipeline._overlaps_existing(a, [b]) is True

    def test_partial_overlap_above_threshold(self):
        a = Segment(segment_id=0, start=0.0, end=10.0, text="A", transcript_segment_ids=[0])
        b = Segment(segment_id=1, start=4.0, end=14.0, text="B", transcript_segment_ids=[1])
        # Overlap = 6s, shorter = 10s, ratio = 0.6 > 0.5
        assert Pipeline._overlaps_existing(a, [b]) is True

    def test_partial_overlap_below_threshold(self):
        a = Segment(segment_id=0, start=0.0, end=10.0, text="A", transcript_segment_ids=[0])
        b = Segment(segment_id=1, start=8.0, end=30.0, text="B", transcript_segment_ids=[1])
        # Overlap = 2s, shorter = 10s, ratio = 0.2 < 0.5
        assert Pipeline._overlaps_existing(a, [b]) is False

    def test_empty_existing_list(self):
        a = Segment(segment_id=0, start=0.0, end=10.0, text="A", transcript_segment_ids=[0])
        assert Pipeline._overlaps_existing(a, []) is False
