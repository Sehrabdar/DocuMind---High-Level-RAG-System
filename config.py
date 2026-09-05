"""
DocuMind configuration.

All settings are driven by environment variables (or a .env file) and have
sensible defaults so the system works out-of-the-box without any configuration.
"""

from pathlib import Path

from pydantic import field_validator
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

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, v: str) -> str:
        return v.upper()


# Module-level singleton — callers import ``settings`` directly.
settings = Settings()
