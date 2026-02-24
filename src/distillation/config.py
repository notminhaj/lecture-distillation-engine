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
    anthropic_api_key: str = Field(..., description="Anthropic API key")

    # ── Model selection ────────────────────────────────────────────────────────
    whisper_model: str = Field("large-v3", description="faster-whisper model name")
    whisper_device: str = Field("cpu", description="'cpu', 'cuda', or 'auto'")
    whisper_compute_type: str = Field(
        "int8", description="Quantisation: float16 (GPU) or int8 (CPU)"
    )
    claude_model: str = Field("claude-sonnet-4-6")

    # ── Clip constraints ───────────────────────────────────────────────────────
    min_clip_duration: float = Field(30.0, description="Minimum clip length in seconds")
    max_clip_duration: float = Field(90.0, description="Maximum clip length in seconds")
    target_clip_count: int = Field(5, description="Clips to produce per lecture")

    # ── Scoring weights (sum need not equal 1; they are L1-normalised) ─────────
    weight_semantic_density: float = 0.20
    weight_emotional_resonance: float = 0.20
    weight_standalone_coherence: float = 0.20
    weight_narrative_completeness: float = 0.15
    weight_domain_integrity: float = 0.15
    weight_hook_strength: float = 0.10

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
