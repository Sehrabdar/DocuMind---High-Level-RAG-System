"""Add search_vector tsvector column and GIN index for PostgreSQL FTS.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-26

Migration overview
------------------
Phase 5 adds keyword retrieval using PostgreSQL native full-text search (FTS).
This migration adds the minimum schema required to support efficient FTS:

1. Add a ``search_vector`` column of type ``tsvector`` to the ``chunks`` table.
   It is defined as ``GENERATED ALWAYS AS (...) STORED`` so PostgreSQL
   automatically populates and maintains it whenever a row is inserted or
   updated — no application code changes are needed.

2. Create a GIN index on ``search_vector`` for fast FTS queries.

Text indexed
------------
Only ``content`` is indexed.  Section path strings are short (1–3 words)
and their lexical signal is already contained in the chunk content.
Extending the index to include section_path is possible in a future phase
if evaluation shows it improves recall.

Language configuration
----------------------
``english`` is used, which applies:
- English stop-word removal (articles, prepositions, etc.)
- Snowball stemming (e.g. "connecting" → "connect")

This is appropriate for an English technical-documentation corpus.
If multi-language support is needed, change the language and regenerate
the column (ALTER TABLE … ALTER COLUMN … SET ...).

Generated column semantics
---------------------------
``GENERATED ALWAYS AS (to_tsvector('english', content)) STORED`` means:
- The column is maintained by PostgreSQL, not the application.
- Every INSERT and UPDATE that changes ``content`` automatically refreshes
  ``search_vector``.
- The existing ``upsert_chunks`` method (which updates ``content`` on conflict)
  automatically benefits without any code change.
- ``STORED`` means the tsvector value is physically written to disk and the
  GIN index can be built on top of it efficiently.

Existing rows
-------------
When the column is added, PostgreSQL populates ``search_vector`` for all
existing rows immediately during the ALTER TABLE statement.  No backfill
migration step is required.

Downgrade
---------
The GIN index is dropped, then the column is dropped.  Existing chunk rows
and their pgvector embeddings are completely unaffected.

Compatibility
-------------
- ``GENERATED ALWAYS AS ... STORED`` requires PostgreSQL 12+.
  The project already requires PostgreSQL 16 (docker-compose.yml).
- ``to_tsvector`` is part of PostgreSQL core — no extension needed.
- The GIN index works correctly alongside the existing HNSW vector index.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

# revision identifiers
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add the generated tsvector column.
    #
    # Using raw SQL because SQLAlchemy's Column() does not support
    # PostgreSQL's GENERATED ALWAYS AS ... STORED syntax for tsvector columns.
    # This is a deliberate, minimal use of raw SQL — kept in the migration
    # rather than scattered across model code.
    op.execute(
        """
        ALTER TABLE chunks
        ADD COLUMN search_vector tsvector
            GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
        """
    )

    # 2. Create a GIN index for fast full-text search.
    #
    # GIN (Generalised Inverted Index) is the standard index type for tsvector.
    # It maps each lexeme to the set of rows containing it, enabling O(log N)
    # lookup per query term.
    #
    # Alternative: GiST — GiST indexes are smaller but slower for pure FTS.
    # GIN is preferred when queries are frequent and inserts are batched (as
    # with the DocuMind ingestion pipeline).
    op.execute(
        """
        CREATE INDEX ix_chunks_search_vector_gin
        ON chunks
        USING gin(search_vector)
        """
    )


def downgrade() -> None:
    # Drop the GIN index first (it depends on the column)
    op.execute("DROP INDEX IF EXISTS ix_chunks_search_vector_gin")

    # Drop the generated column
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector")
