"""
LLM-based metadata generator.

Produces per-clip:
  - Hook caption (≤150 chars, designed to stop a scroll)
  - Description (YouTube Shorts / IG Reels description)
  - Hashtags (platform-optimised, domain-aware)
  - Thumbnail concept (text description)
  - B-roll suggestions (timestamped visual concept per major point)

Why Claude for this and not template-based generation?
  - Hook captions require understanding the specific emotional payoff of the clip.
  - Hashtag sets need to be contextually relevant, not just generic.
  - B-roll ideas require understanding what the speaker is describing
    (e.g., "he's describing the Day of Judgement" → suggest epic celestial imagery).
  - Templates produce identical-sounding captions; LLMs produce variety that
    avoids platform shadow-banning from repetition signals.
"""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from distillation.config import Config
from distillation.models import BRollSuggestion, Clip, ClipMetadata, Domain

logger = logging.getLogger(__name__)

DOMAIN_HASHTAG_SEEDS: dict[Domain, list[str]] = {
    Domain.KHUTBA: [
        "#islamicreminder", "#khutbah", "#fridaysermon", "#muslimtiktok",
        "#islamicquotes", "#quran", "#deen",
    ],
    Domain.PODCAST: ["#podcast", "#podcastclips"],
    Domain.BUSINESS_TALK: ["#businessadvice", "#leadership", "#entrepreneurship"],
    Domain.LECTURE: ["#education", "#learning"],
    Domain.GENERIC: [],
}


class MetadataGenerator:
    """Generates platform-ready metadata for each clip using Claude."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.client = Anthropic(api_key=config.anthropic_api_key)

    def generate(self, clip: Clip, domain: Domain = Domain.GENERIC) -> ClipMetadata:
        """Generate full metadata for a single clip."""
        raw = self._call_llm(clip, domain)
        return self._parse(raw, domain)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_llm(self, clip: Clip, domain: Domain) -> str:
        seed_tags = DOMAIN_HASHTAG_SEEDS.get(domain, [])
        clip_text = clip.scored_segment.segment.text

        prompt = f"""You are a short-form video strategist.

CLIP TEXT ({clip.duration:.0f} seconds):
{clip_text}

DOMAIN: {domain.value}

Generate platform metadata for this clip. Return ONLY valid JSON:
{{
  "hook_caption": "<max 150 chars; must start with the most compelling idea; no filler>",
  "description": "<2-3 sentences for YouTube Shorts description>",
  "hashtags": ["<10-15 hashtags including {', '.join(seed_tags[:3])} if relevant>"],
  "thumbnail_suggestion": "<one-sentence visual description of an ideal thumbnail>",
  "b_roll_suggestions": [
    {{
      "timestamp_in_clip": <float seconds from clip start>,
      "duration": <float seconds>,
      "description": "<stock footage concept>",
      "tags": ["<search keyword>"]
    }}
  ]
}}

Rules:
- hook_caption must NOT start with "In this clip" or "Watch as" — start with the idea.
- thumbnail_suggestion should be a specific visual, not generic.
- Provide 2-4 b_roll_suggestions tied to specific moments in the transcript.
- hashtags: mix popular + niche, no spaces in tags."""

        response = self.client.messages.create(
            model=self.cfg.claude_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    @staticmethod
    def _parse(raw: str, domain: Domain) -> ClipMetadata:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse metadata JSON from LLM")
            return ClipMetadata(
                hook_caption="Watch this powerful clip.",
                description="",
                hashtags=DOMAIN_HASHTAG_SEEDS.get(domain, []),
                thumbnail_suggestion="Speaker in mid-sentence, text overlay with key quote.",
            )

        b_roll = []
        for item in data.get("b_roll_suggestions", []):
            try:
                b_roll.append(BRollSuggestion(
                    timestamp_in_clip=float(item.get("timestamp_in_clip", 0)),
                    duration=float(item.get("duration", 3)),
                    description=str(item.get("description", "")),
                    tags=list(item.get("tags", [])),
                ))
            except Exception:
                pass

        return ClipMetadata(
            hook_caption=str(data.get("hook_caption", ""))[:150],
            description=str(data.get("description", "")),
            hashtags=list(data.get("hashtags", DOMAIN_HASHTAG_SEEDS.get(domain, []))),
            thumbnail_suggestion=str(data.get("thumbnail_suggestion", "")),
            b_roll_suggestions=b_roll,
        )
