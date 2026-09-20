"""Create chunks table with pgvector embedding column and HNSW index.

Revision ID: 0001
Revises:
Create Date: 2026-09-20

Migration overview
------------------
1. Enable the pgvector extension (idempotent — ``IF NOT EXISTS``).
2. Create the ``chunks`` table with all provenance columns.
3. Create a UNIQUE index on ``chunk_id`` (idempotency contract).
4. Create a plain B-tree index on ``document_id`` (fetch-by-doc queries).
5. Create an HNSW vector index on ``embedding`` using cosine distance.

HNSW index parameters
---------------------
- ``m=16``: Maximum number of connections per layer.  Controls the tradeoff
  between index size, build time, and recall.  16 is the pgvector default
  and a well-regarded starting point for document retrieval workloads.
- ``ef_construction=64``: Size of the candidate list during construction.
  Higher values improve recall at the cost of build time.  64 is a sensible
  default that can be tuned in a later evaluation phase.
- ``vector_cosine_ops``: Operator class for cosine distance.  Matches the
  L2-normalized embeddings from ``EmbeddingService`` — after normalization,
  cosine similarity equals dot product, making pgvector's cosine ops optimal.

Downgrade
---------
Drops the ``chunks`` table entirely.  The vector extension is NOT dropped
on downgrade (it may be used by other tables in the future).
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector  # type: ignore[import]

# revision identifiers
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable pgvector extension
    # This is idempotent — safe to run on a database that already has it.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Create the chunks table
    op.create_table(
        "chunks",
        # ── Identity ──────────────────────────────────────────────────────
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("chunk_id", sa.String(64), nullable=False),
        sa.Column("document_id", sa.String(64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        # ── Content ───────────────────────────────────────────────────────
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        # ── Embedding ─────────────────────────────────────────────────────
        # Dimension 384 matches BAAI/bge-small-en-v1.5.
        # If the model changes, create a new migration to ALTER the column.
        sa.Column("embedding", Vector(384), nullable=False),
        # ── Structural provenance ─────────────────────────────────────────
        sa.Column("section_path", JSONB(), nullable=False, server_default="[]"),
        sa.Column("document_title", sa.String(1024), nullable=True),
        # ── Location provenance ───────────────────────────────────────────
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("start_char", sa.Integer(), nullable=False),
        sa.Column("end_char", sa.Integer(), nullable=False),
        # ── Audit ─────────────────────────────────────────────────────────
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        # ── Constraints ───────────────────────────────────────────────────
        sa.PrimaryKeyConstraint("id", name="pk_chunks"),
        sa.UniqueConstraint("chunk_id", name="uq_chunks_chunk_id"),
    )

    # 3. B-tree index on document_id for "fetch all chunks for a document"
    op.create_index(
        "ix_chunks_document_id",
        "chunks",
        ["document_id"],
        unique=False,
    )

    # 4. HNSW vector index for cosine similarity search (Phase 4+)
    # Using raw SQL because SQLAlchemy's Index() does not support pgvector's
    # custom operator classes and HNSW-specific parameters.
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding_hnsw
        ON chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.drop_table("chunks")
    # The vector extension is intentionally NOT dropped on downgrade
    # because other tables might depend on it in the future.
