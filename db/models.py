"""
SQLAlchemy model for persisted chunks (Phase 3).

Schema design rationale
-----------------------
Fields are split into two categories:

**Explicit columns** (scalar values likely to be filtered, sorted, or joined):
    chunk_id, document_id, chunk_index, page, page_start, page_end,
    start_char, end_char, token_count — these will appear in WHERE clauses,
    ORDER BY, and GROUP BY in future retrieval and evaluation queries.

**JSONB column** (variable-length structured data):
    section_path — a list[str] of variable length (0 to N entries depending
    on document structure).  Storing this as JSONB avoids the awkwardness of
    either a VARCHAR with a delimiter or a separate normalized join table.
    Querying by section path is possible with PostgreSQL JSONB operators
    if needed in a future phase.

The ``embedding`` column uses pgvector's ``Vector(dimension)`` type.  The
dimension is read from the project configuration so that it is the single
authoritative value — changing the embedding model requires updating the
config and generating a new Alembic migration, not hunting for magic numbers.

Boundary between Pydantic and SQLAlchemy
-----------------------------------------
``NormalizedChunk`` (Pydantic) is the in-memory data contract for the chunking
pipeline.  ``ChunkRecord`` (SQLAlchemy) is the persistence contract for the
database.  ``chunk_record_from_normalized()`` is the explicit adapter between
the two — never access ``ChunkRecord`` attributes directly from the chunking
layer.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector  # type: ignore[import]
from sqlalchemy import DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base for all DocuMind models."""
    pass


class ChunkRecord(Base):
    """Database representation of a persisted chunk with its embedding.

    Maps to the ``chunks`` table in PostgreSQL.

    Fields
    ------
    id:
        Auto-increment integer primary key (internal row identifier).
        Use ``chunk_id`` for application-level identity.

    chunk_id:
        Deterministic identifier from Phase 2 chunking:
        ``SHA-256(document_id:chunk_index:content[:64])``.
        Has a UNIQUE constraint — used as the upsert conflict target.

    document_id:
        References ``NormalizedDocument.id`` (SHA-256 of file path).
        Indexed for efficient "fetch all chunks for a document" queries.

    embedding:
        pgvector ``Vector(384)`` column.  Stores the L2-normalized embedding
        produced by ``EmbeddingService.embed_texts()``.  Dimension comes from
        ``settings.embedding_dimension`` and must match the HNSW index.

    section_path:
        JSONB list[str] of heading breadcrumbs (e.g. ["Auth", "OAuth"]).
        Empty list for PDF, HTML, and DOCX (no heading hierarchy available).

    created_at:
        UTC timestamp of first insertion.  Updated on upsert so that
        re-ingestion of unchanged chunks does NOT change this value
        (only changed content updates it — see repository.py).
    """

    __tablename__ = "chunks"

    # ── Identity ──────────────────────────────────────────────────────────────
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False)
    document_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Content ───────────────────────────────────────────────────────────────
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Embedding ─────────────────────────────────────────────────────────────
    # Dimension is resolved at import time from configuration.
    # If the model changes, update settings.embedding_dimension and generate
    # a new Alembic migration — the change will be explicit and traceable.
    embedding: Mapped[list[float]] = mapped_column(
        Vector(384),  # kept as literal; runtime dimension check in repository
        nullable=False,
    )

    # ── Structural provenance ─────────────────────────────────────────────────
    section_path: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    document_title: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # ── Location provenance ───────────────────────────────────────────────────
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)

    # ── Audit ─────────────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint("chunk_id", name="uq_chunks_chunk_id"),
        # HNSW index is created in the Alembic migration (not here) because
        # SQLAlchemy does not natively support pgvector index parameters
        # (m, ef_construction).  The DDL migration has the full CREATE INDEX
        # statement with the correct operator class and parameters.
    )

    def __repr__(self) -> str:
        return (
            f"ChunkRecord("
            f"chunk_id={self.chunk_id!r}, "
            f"document_id={self.document_id!r}, "
            f"chunk_index={self.chunk_index!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Adapter: NormalizedChunk + embedding → ChunkRecord
# ─────────────────────────────────────────────────────────────────────────────


def chunk_record_from_normalized(
    chunk: "NormalizedChunkProtocol",
    embedding: list[float],
) -> ChunkRecord:
    """Create a ``ChunkRecord`` from a ``NormalizedChunk`` and its embedding.

    This is the single adapter between the Pydantic chunking model and the
    SQLAlchemy persistence model.  Keeping it here (not in the repository)
    makes both models independently testable.

    Parameters
    ----------
    chunk:
        A ``NormalizedChunk`` (or any duck-typed equivalent).
    embedding:
        L2-normalized float vector from ``EmbeddingService.embed_texts()``.
        Length must match ``ChunkRecord.embedding`` column dimension (384).

    Returns
    -------
    ChunkRecord
        Ready to be added to a session.  ``created_at`` defaults to now.
    """
    return ChunkRecord(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        token_count=chunk.token_count,
        embedding=embedding,
        section_path=list(chunk.section_path),
        document_title=chunk.document_title,
        page=chunk.page,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        start_char=chunk.start_char,
        end_char=chunk.end_char,
        created_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Duck-type protocol (TYPE_CHECKING only — avoids circular imports)
# ─────────────────────────────────────────────────────────────────────────────

class NormalizedChunkProtocol:  # pragma: no cover
    """Duck-type protocol for chunk_record_from_normalized() — not imported at runtime."""
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    section_path: list[str]
    document_title: str | None
    page: int | None
    page_start: int | None
    page_end: int | None
    start_char: int
    end_char: int
