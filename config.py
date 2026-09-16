"""
DocuMind configuration.

All settings are driven by environment variables (or a .env file) and have
sensible defaults so the system works out-of-the-box without any configuration.
"""

from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for DocuMind.

    Values are read from environment variables with the ``DOCUMIND_`` prefix
    (e.g. ``DOCUMIND_LOG_LEVEL=DEBUG``), then from a ``.env`` file, then from
    the defaults defined here.
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCUMIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Ingestion ─────────────────────────────────────────────────────────────
    corpus_dir: Path = Path("data/corpus")
    """Default corpus root used by the CLI when --input is not supplied."""

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    """Python logging level name."""

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    """Root data directory (corpus lives inside this)."""

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 512
    """Target/maximum tokens per chunk.

    Chosen as a starting point for technical documentation: large enough to
    contain a full API description or procedure, small enough to remain a
    precise retrieval unit.  Empirical tuning happens in a later evaluation
    phase.
    """

    chunk_overlap: int = 50
    """Tokens of overlap between consecutive sub-chunks of an oversized section.

    Overlap preserves local context across sub-chunk boundaries so that a
    sentence split between two chunks is still intelligible in both.  Applied
    only when a section is large enough to require subdivision — never between
    independent sections.
    """

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, v: str) -> str:
        return v.upper()

    @field_validator("chunk_size", mode="before")
    @classmethod
    def _validate_chunk_size(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError(f"chunk_size must be > 0, got {v}")
        return int(v)

    @field_validator("chunk_overlap", mode="before")
    @classmethod
    def _validate_chunk_overlap(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {v}")
        return int(v)

    @model_validator(mode="after")
    def _validate_overlap_less_than_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self


# Module-level singleton — callers import ``settings`` directly.
settings = Settings()
