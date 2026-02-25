"""Tests for the video exporter — ffmpeg calls are mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distillation.models import (
    Clip,
    ClipMetadata,
    EngagementAxes,
    Segment,
    ScoredSegment,
    SubtitleLine,
)
from distillation.export.video import VideoExporter


def make_clip(clip_id: int = 0, start: float = 10.0, end: float = 40.0, with_subs: bool = False, with_meta: bool = False) -> Clip:
    seg = Segment(segment_id=0, start=start, end=end, text="Test text.", transcript_segment_ids=[0])
    axes = EngagementAxes(
        semantic_density=0.8, emotional_resonance=0.7, standalone_coherence=0.9,
        narrative_completeness=0.85, domain_integrity=0.95, hook_strength=0.6,
    )
    scored = ScoredSegment(segment=seg, scores=axes, acoustic_energy=0.5, speech_rate_wpm=130.0)
    clip = Clip(clip_id=clip_id, source_path="/tmp/source.mp4", start=start, end=end, scored_segment=scored)
    if with_subs:
        clip.subtitles = [
            SubtitleLine(index=1, start=0.0, end=2.0, text="Test subtitle"),
        ]
    if with_meta:
        clip.metadata = ClipMetadata(
            hook_caption="Hook caption",
            description="Description",
            hashtags=["#test"],
            thumbnail_suggestion="Thumbnail",
        )
    return clip


@pytest.fixture
def config(tmp_path):
    cfg = MagicMock()
    cfg.output_dir = tmp_path / "outputs"
    return cfg


@pytest.fixture
def exporter(config):
    return VideoExporter(config, force_reencode=True)


@pytest.fixture
def stream_copy_exporter(config):
    return VideoExporter(config, force_reencode=False)


class TestVideoExporter:
    def test_output_dir_created(self, config, tmp_path):
        VideoExporter(config)
        assert (tmp_path / "outputs").is_dir()

    def test_default_is_reencode(self, config):
        exp = VideoExporter(config)
        assert exp.force_reencode is True

    def test_stream_copy_mode(self, config):
        exp = VideoExporter(config, force_reencode=False)
        assert exp.force_reencode is False

    @patch("distillation.export.video.subprocess.run")
    def test_export_creates_clip_dir(self, mock_run, exporter, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip(clip_id=3)
        exporter.export(clip, source_video="/tmp/source.mp4")

        clip_dir = tmp_path / "outputs" / "clip_003"
        assert clip_dir.is_dir()

    @patch("distillation.export.video.subprocess.run")
    def test_export_sets_output_path(self, mock_run, exporter):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip()
        exporter.export(clip, source_video="/tmp/source.mp4")

        assert clip.output_path is not None
        assert "clip_000.mp4" in clip.output_path

    @patch("distillation.export.video.subprocess.run")
    def test_reencode_uses_libx264(self, mock_run, exporter):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip()
        exporter.export(clip, source_video="/tmp/source.mp4")

        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd
        assert "-c:v" in cmd

    @patch("distillation.export.video.subprocess.run")
    def test_stream_copy_uses_c_copy(self, mock_run, stream_copy_exporter):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip()
        stream_copy_exporter.export(clip, source_video="/tmp/source.mp4")

        cmd = mock_run.call_args[0][0]
        assert "copy" in cmd

    @patch("distillation.export.video.subprocess.run")
    def test_ffmpeg_failure_raises(self, mock_run, exporter):
        mock_run.return_value = MagicMock(returncode=1, stderr="encoding error")
        clip = make_clip()

        with pytest.raises(RuntimeError, match="ffmpeg clip cut failed"):
            exporter.export(clip, source_video="/tmp/source.mp4")

    @patch("distillation.export.video.subprocess.run")
    def test_srt_written_when_subtitles_present(self, mock_run, exporter, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip(with_subs=True)
        exporter.export(clip, source_video="/tmp/source.mp4")

        srt_path = tmp_path / "outputs" / "clip_000" / "clip_000.srt"
        assert srt_path.exists()
        assert "Test subtitle" in srt_path.read_text(encoding="utf-8")

    @patch("distillation.export.video.subprocess.run")
    def test_srt_not_written_when_no_subtitles(self, mock_run, exporter, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip(with_subs=False)
        exporter.export(clip, source_video="/tmp/source.mp4")

        srt_path = tmp_path / "outputs" / "clip_000" / "clip_000.srt"
        assert not srt_path.exists()

    @patch("distillation.export.video.subprocess.run")
    def test_metadata_json_written(self, mock_run, exporter, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip(with_meta=True)
        exporter.export(clip, source_video="/tmp/source.mp4")

        meta_path = tmp_path / "outputs" / "clip_000" / "clip_000_metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert data["clip_id"] == 0
        assert "scores" in data
        assert "metadata" in data

    @patch("distillation.export.video.subprocess.run")
    def test_metadata_json_without_clip_metadata(self, mock_run, exporter, tmp_path):
        mock_run.return_value = MagicMock(returncode=0)
        clip = make_clip(with_meta=False)
        exporter.export(clip, source_video="/tmp/source.mp4")

        meta_path = tmp_path / "outputs" / "clip_000" / "clip_000_metadata.json"
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "metadata" not in data
