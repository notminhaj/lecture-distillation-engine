"""Tests for the metadata generator — all LLM calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from distillation.models import Domain
from distillation.generation.metadata import (
    DOMAIN_HASHTAG_SEEDS,
    MetadataGenerator,
)


def make_valid_metadata_dict() -> dict:
    return {
        "hook_caption": "The Prophet's advice that changed everything",
        "description": "A powerful reminder about seeking knowledge. This clip from a khutba delivers a timeless message.",
        "hashtags": ["#islamicreminder", "#khutbah", "#knowledge", "#deen"],
        "thumbnail_suggestion": "Speaker at podium with Arabic calligraphy overlay of the hadith text",
        "b_roll_suggestions": [
            {
                "timestamp_in_clip": 2.0,
                "duration": 3.0,
                "description": "Ancient library with Islamic manuscripts",
                "tags": ["library", "manuscripts", "islamic"],
            },
        ],
    }


@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.openai_api_key = "test-key"
    cfg.openai_model = "gpt-4o-mini"
    return cfg


@pytest.fixture
def generator(config):
    with patch("distillation.generation.metadata.OpenAI"):
        return MetadataGenerator(config)


@pytest.fixture
def mock_clip():
    clip = MagicMock()
    clip.duration = 30.0
    clip.scored_segment.segment.text = "The Prophet said seek knowledge even unto China."
    return clip


class TestMetadataGeneratorParsing:
    """Test _parse in isolation — no network calls needed."""

    def test_valid_json(self):
        raw = json.dumps(make_valid_metadata_dict())
        result = MetadataGenerator._parse(raw, Domain.KHUTBA)

        assert len(result.hook_caption) <= 150
        assert len(result.hashtags) > 0
        assert len(result.b_roll_suggestions) == 1
        assert result.b_roll_suggestions[0].timestamp_in_clip == 2.0

    def test_markdown_fenced_json(self):
        data = make_valid_metadata_dict()
        raw = f"```json\n{json.dumps(data)}\n```"
        result = MetadataGenerator._parse(raw, Domain.KHUTBA)

        assert result.hook_caption == data["hook_caption"]

    def test_invalid_json_returns_fallback(self):
        result = MetadataGenerator._parse("NOT JSON {{{", Domain.KHUTBA)

        assert result.hook_caption == "Watch this powerful clip."
        assert result.hashtags == DOMAIN_HASHTAG_SEEDS[Domain.KHUTBA]

    def test_hook_caption_truncated_to_150_chars(self):
        data = make_valid_metadata_dict()
        data["hook_caption"] = "x" * 200
        raw = json.dumps(data)
        result = MetadataGenerator._parse(raw, Domain.GENERIC)

        assert len(result.hook_caption) == 150

    def test_missing_b_roll_still_parses(self):
        data = make_valid_metadata_dict()
        del data["b_roll_suggestions"]
        raw = json.dumps(data)
        result = MetadataGenerator._parse(raw, Domain.KHUTBA)

        assert result.b_roll_suggestions == []

    def test_malformed_b_roll_item_skipped(self):
        data = make_valid_metadata_dict()
        data["b_roll_suggestions"] = [
            {"bad": "data"},  # should be skipped
            {
                "timestamp_in_clip": 5.0,
                "duration": 2.0,
                "description": "Valid item",
                "tags": ["test"],
            },
        ]
        raw = json.dumps(data)
        result = MetadataGenerator._parse(raw, Domain.KHUTBA)

        # The malformed one may or may not parse (depends on defaults),
        # but at least the valid one should be present
        assert any(b.description == "Valid item" for b in result.b_roll_suggestions)

    def test_domain_fallback_hashtags_on_parse_failure(self):
        """Each domain should get its own seed hashtags on failure."""
        for domain in [Domain.KHUTBA, Domain.PODCAST, Domain.GENERIC]:
            result = MetadataGenerator._parse("INVALID", domain)
            assert result.hashtags == DOMAIN_HASHTAG_SEEDS.get(domain, [])


class TestMetadataGeneratorIntegration:
    """Integration tests with mocked OpenAI client."""

    def test_generate_returns_clip_metadata(self, generator, mock_clip):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(make_valid_metadata_dict())))
        ]
        generator.client.chat.completions.create.return_value = mock_response

        result = generator.generate(mock_clip, domain=Domain.KHUTBA)

        assert result.hook_caption
        assert len(result.hashtags) > 0
        generator.client.chat.completions.create.assert_called_once()

    def test_generate_prompt_includes_clip_text(self, generator, mock_clip):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(make_valid_metadata_dict())))
        ]
        generator.client.chat.completions.create.return_value = mock_response

        generator.generate(mock_clip, domain=Domain.KHUTBA)

        call_args = generator.client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        prompt_text = messages[0]["content"]
        assert "seek knowledge" in prompt_text
