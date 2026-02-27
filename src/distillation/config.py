"""
Centralised configuration via Pydantic Settings.

All environment variables are read here and nowhere else.  Every module
imports Config and reads from the singleton — no scattered os.getenv() calls.
"""

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from distillation.models import Domain


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API credentials ────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", description="Anthropic API key")
    gemini_api_key: str = Field("", description="Google Gemini API key")
    openai_api_key: str = Field("", description="OpenAI API key")

    # ── Model selection ────────────────────────────────────────────────────────
    whisper_model: str = Field("large-v3", description="faster-whisper model name")
    whisper_device: str = Field("cpu", description="'cpu', 'cuda', or 'auto'")
    whisper_compute_type: str = Field(
        "int8", description="Quantisation: float16 (GPU) or int8 (CPU)"
    )
    claude_model: str = Field("claude-sonnet-4-6")
    gemini_model: str = Field("gemini-2.0-flash")
    openai_model: str = Field("gpt-4o-mini")

    # ── Clip constraints ───────────────────────────────────────────────────────
    min_clip_duration: float = Field(30.0, description="Minimum clip length in seconds")
    max_clip_duration: float = Field(90.0, description="Maximum clip length in seconds")
    target_clip_count: int = Field(5, description="Clips to produce per lecture")

    # ── Selection gates ───────────────────────────────────────────────────────
    min_narrative_completeness: float = Field(
        0.5, description="Segments below this narrative_completeness are excluded before ranking"
    )

    # ── Scoring weights (sum need not equal 1; they are L1-normalised) ─────────
    weight_semantic_density: float = 0.20
    weight_emotional_resonance: float = 0.20
    weight_standalone_coherence: float = 0.20
    weight_narrative_completeness: float = 0.15
    weight_domain_integrity: float = 0.15
    weight_hook_strength: float = 0.10

    # ── Nomination ────────────────────────────────────────────────────────────
    nomination_strategy: str = Field(
        "llm",
        description="'llm' (LLM moment detection), 'semantic' (old segmentation), or 'hybrid' (union)",
    )

    # ── Segmentation ───────────────────────────────────────────────────────────
    segmentation_strategy: str = Field(
        "semantic",
        description="'sliding_window' or 'semantic' (embedding-based TextTiling)",
    )
    # Sliding window fallback
    sliding_window_size: float = Field(60.0, description="Window size in seconds")
    sliding_window_stride: float = Field(15.0, description="Stride in seconds")
    # Semantic segmentation
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence encoder for cosine similarity boundaries",
    )
    semantic_similarity_threshold: float = Field(
        0.35,
        description="Cosine distance above which a new segment starts",
    )

    # ── Boundary refinement ──────────────────────────────────────────────────
    sentence_boundary_window: float = Field(
        5.0, description="±seconds to search for sentence boundary when snapping"
    )
    silence_threshold_ms: float = Field(
        600.0, description="Minimum silence gap (ms) to prefer as boundary"
    )
    hook_protection_window: float = Field(
        3.0, description="Seconds at segment start to check for hook energy"
    )
    enable_boundary_refinement: bool = Field(
        True, description="Toggle boundary refinement on/off"
    )
    enable_word_boundary_refinement: bool = Field(
        True,
        description=(
            "Run WordBoundaryRefiner after selection to snap clip edges "
            "to exact Whisper word boundaries"
        ),
    )

    # ── Download ─────────────────────────────────────────────────────────────
    cookies_from_browser: str = Field(
        "", description="Browser to extract cookies from for yt-dlp (e.g. 'chrome', 'firefox', 'edge')"
    )
    cookies_file: str = Field(
        "", description="Path to a Netscape-format cookies.txt file for yt-dlp"
    )

    # ── Domain ────────────────────────────────────────────────────────────────
    default_domain: Domain = Field(Domain.KHUTBA)

    # ── Paths ──────────────────────────────────────────────────────────────────
    output_dir: Path = Field(Path("outputs"))
    cache_dir: Path = Field(Path("cache"))

    # ── LLM retry ─────────────────────────────────────────────────────────────
    llm_max_retries: int = Field(3)
    llm_retry_wait_seconds: float = Field(2.0)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the singleton Config, cached after first load."""
    return Config()  # type: ignore[call-arg]
