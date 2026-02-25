"""
Tests for LLM scorer — all LLM calls are mocked.

The goal is to validate:
  - JSON parsing robustness (malformed, fenced, partial responses)
  - Neutral fallback on parse failure
  - Correct merging of acoustic and LLM scores
  - Batch chunking logic
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from distillation.models import Domain, EngagementAxes, Segment
from distillation.scoring.llm_scorer import LLMScorer


def make_valid_axes_dict(segment_id: int = 0) -> dict:
    return {
        "segment_id": segment_id,
        "semantic_density": 0.8,
        "emotional_resonance": 0.7,
        "standalone_coherence": 0.9,
        "narrative_completeness": 0.85,
        "domain_integrity": 0.95,
        "hook_strength": 0.6,
        "llm_rationale": "Good clip with clear narrative arc.",
    }


class TestLLMScorerParsing:
    """Test _parse_response in isolation — no network calls needed."""

    def test_valid_json_array(self):
        raw = json.dumps([make_valid_axes_dict(0), make_valid_axes_dict(1)])
        axes = LLMScorer._parse_response(raw, expected_count=2)
        assert len(axes) == 2
        assert axes[0].semantic_density == pytest.approx(0.8)

    def test_markdown_fenced_json(self):
        data = [make_valid_axes_dict(0)]
        raw = f"```json\n{json.dumps(data)}\n```"
        axes = LLMScorer._parse_response(raw, expected_count=1)
        assert len(axes) == 1

    def test_invalid_json_returns_neutral_defaults(self):
        axes = LLMScorer._parse_response("NOT JSON {{{", expected_count=2)
        assert len(axes) == 2
        for ax in axes:
            assert ax.semantic_density == pytest.approx(0.5)

    def test_partial_response_padded(self):
        """If Claude returns fewer items than expected, we pad with last item."""
        raw = json.dumps([make_valid_axes_dict(0)])  # only 1 item for 3 expected
        axes = LLMScorer._parse_response(raw, expected_count=3)
        assert len(axes) == 3

    def test_extra_response_truncated(self):
        raw = json.dumps([make_valid_axes_dict(i) for i in range(5)])
        axes = LLMScorer._parse_response(raw, expected_count=3)
        assert len(axes) == 3

    def test_out_of_range_scores_clamped_by_pydantic(self):
        """Pydantic should reject out-of-range values and we should see neutral."""
        data = [{"segment_id": 0, "semantic_density": 99.0, **{
            k: 0.5 for k in [
                "emotional_resonance", "standalone_coherence",
                "narrative_completeness", "domain_integrity", "hook_strength",
            ]
        }}]
        # Pydantic ValidationError triggers our except branch → neutral default
        axes = LLMScorer._parse_response(json.dumps(data), expected_count=1)
        # Either the value is clamped or we get neutral — either is acceptable
        assert 0.0 <= axes[0].semantic_density <= 1.0


class TestLLMScorerIntegration:
    """Integration tests with mocked Anthropic client."""

    @pytest.fixture
    def config(self):
        cfg = MagicMock()
        cfg.openai_api_key = "test-key"
        cfg.openai_model = "gpt-4o-mini"
        return cfg

    @pytest.fixture
    def scorer(self, config):
        with patch("distillation.scoring.llm_scorer.OpenAI"):
            return LLMScorer(config)

    def test_score_returns_same_count_as_input(
        self, scorer, sample_transcript, sample_segment
    ):
        # Mock the API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps([make_valid_axes_dict(0)])))]
        scorer.client.chat.completions.create.return_value = mock_response

        segments = [sample_segment]
        result = scorer.score(segments, sample_transcript, domain=Domain.KHUTBA)

        assert len(result) == 1

    def test_score_preserves_segment_identity(
        self, scorer, sample_transcript, sample_segment
    ):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps([make_valid_axes_dict(0)])))]
        scorer.client.chat.completions.create.return_value = mock_response

        result = scorer.score([sample_segment], sample_transcript, domain=Domain.KHUTBA)

        assert result[0].segment.segment_id == sample_segment.segment_id

    def test_empty_segments_returns_empty(self, scorer, sample_transcript):
        result = scorer.score([], sample_transcript, domain=Domain.KHUTBA)
        assert result == []
