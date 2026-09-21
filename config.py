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

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    """HuggingFace model identifier for the local embedding model.

    BAAI/bge-small-en-v1.5 was selected as the Phase 3 baseline because it:
    - Runs locally (no per-document API cost)
    - Produces 384-dimensional vectors (manageable pgvector index size)
    - Performs competitively on MTEB semantic similarity benchmarks
    - Recommends L2-normalized cosine similarity (matching pgvector HNSW setup)
    - Is reproducible: same input → same vector, deterministically
    """

    embedding_dimension: int = 384
    """Output vector dimension for the configured embedding model.

    This is the single authoritative dimension value.  The SQLAlchemy model
    and Alembic migration both read from this configuration.  If the model
    changes, update this value and generate a new migration — the discrepancy
    will be explicit rather than scattered across the codebase.
    """

    embedding_batch_size: int = 64
    """Number of texts to encode per batch.

    Balances GPU/CPU utilisation against memory pressure.  The default of 64
    is conservative and works on CPU-only machines.  Increase for GPU inference.
    """

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://documind:documind@localhost:5434/documind"
    )
    """SQLAlchemy async database URL.

    Uses asyncpg as the async driver, matching the FastAPI-oriented Phase 8
    architecture.  Starting async now avoids a painful sync→async migration later.

    Host port 5434 maps to the Docker Compose container's port 5432 to avoid
    conflict with any local PostgreSQL instance on the standard port.
    Credentials default to the Docker Compose values — never commit a real
    credentials-bearing URL.
    """

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_default_top_k: int = 5
    """Default number of chunks returned by a single retrieval call.

    Five chunks is a sensible starting point for RAG: enough context for a
    complete answer, small enough to keep the LLM prompt compact.  Users can
    override per-query with --top-k.
    """

    retrieval_max_top_k: int = 100
    """Hard ceiling on top_k accepted by the retrieval API.

    Prevents accidental full-table scans via the retrieval layer.  The HNSW
    index is efficient for small K; very large K degrades to a linear scan.
    If higher recall is needed, adjust this value and monitor query latency.
    """

    # ── Validators ────────────────────────────────────────────────────────────

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

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def _validate_embedding_dimension(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError(f"embedding_dimension must be > 0, got {v}")
        return int(v)

    @field_validator("embedding_batch_size", mode="before")
    @classmethod
    def _validate_embedding_batch_size(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError(f"embedding_batch_size must be > 0, got {v}")
        return int(v)

    @field_validator("retrieval_default_top_k", mode="before")
    @classmethod
    def _validate_retrieval_default_top_k(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError(f"retrieval_default_top_k must be > 0, got {v}")
        return int(v)

    @field_validator("retrieval_max_top_k", mode="before")
    @classmethod
    def _validate_retrieval_max_top_k(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError(f"retrieval_max_top_k must be > 0, got {v}")
        return int(v)

    @model_validator(mode="after")
    def _validate_overlap_less_than_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.retrieval_default_top_k > self.retrieval_max_top_k:
            raise ValueError(
                f"retrieval_default_top_k ({self.retrieval_default_top_k}) "
                f"must be <= retrieval_max_top_k ({self.retrieval_max_top_k})"
            )
        return self


# Module-level singleton — callers import ``settings`` directly.
settings = Settings()
