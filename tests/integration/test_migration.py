"""
Integration tests for Alembic migration and database schema.

Verifies that after `alembic upgrade head`:
- The pgvector extension is enabled
- The chunks table exists
- All expected columns are present with correct types
- The unique constraint on chunk_id exists
- The HNSW index on embedding exists
- The B-tree index on document_id exists

Requires a running PostgreSQL + pgvector instance.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import requires_db


@pytest.mark.integration
@requires_db
class TestMigrationSchema:
    """Verify the database schema after alembic upgrade head."""

    @pytest.mark.asyncio
    async def test_vector_extension_exists(self, db_session: AsyncSession):
        """pgvector extension must be installed."""
        result = await db_session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        row = result.fetchone()
        assert row is not None, "pgvector extension not found — run: CREATE EXTENSION IF NOT EXISTS vector"

    @pytest.mark.asyncio
    async def test_chunks_table_exists(self, db_session: AsyncSession):
        """The chunks table must exist after migration."""
        result = await db_session.execute(
            text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = 'chunks'"
            )
        )
        row = result.fetchone()
        assert row is not None, "chunks table does not exist"

    @pytest.mark.asyncio
    async def test_required_columns_exist(self, db_session: AsyncSession):
        """All expected columns must be present in the chunks table."""
        result = await db_session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'chunks'"
            )
        )
        columns = {row[0] for row in result.fetchall()}

        expected = {
            "id", "chunk_id", "document_id", "chunk_index",
            "content", "token_count", "embedding",
            "section_path", "document_title",
            "page", "page_start", "page_end",
            "start_char", "end_char", "created_at",
        }
        missing = expected - columns
        assert not missing, f"Missing columns: {missing}"

    @pytest.mark.asyncio
    async def test_unique_constraint_on_chunk_id(self, db_session: AsyncSession):
        """The uq_chunks_chunk_id unique constraint must exist."""
        result = await db_session.execute(
            text(
                "SELECT constraint_name FROM information_schema.table_constraints "
                "WHERE table_name = 'chunks' AND constraint_type = 'UNIQUE'"
            )
        )
        constraints = {row[0] for row in result.fetchall()}
        assert "uq_chunks_chunk_id" in constraints, (
            f"uq_chunks_chunk_id not found. Constraints found: {constraints}"
        )

    @pytest.mark.asyncio
    async def test_hnsw_index_exists(self, db_session: AsyncSession):
        """The HNSW index on the embedding column must exist."""
        result = await db_session.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'chunks' AND indexname = 'ix_chunks_embedding_hnsw'"
            )
        )
        row = result.fetchone()
        assert row is not None, "HNSW index ix_chunks_embedding_hnsw not found"

    @pytest.mark.asyncio
    async def test_document_id_index_exists(self, db_session: AsyncSession):
        """The B-tree index on document_id must exist."""
        result = await db_session.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'chunks' AND indexname = 'ix_chunks_document_id'"
            )
        )
        row = result.fetchone()
        assert row is not None, "B-tree index ix_chunks_document_id not found"

    @pytest.mark.asyncio
    async def test_embedding_column_is_vector_type(self, db_session: AsyncSession):
        """The embedding column must use the pgvector type."""
        result = await db_session.execute(
            text(
                "SELECT data_type, udt_name FROM information_schema.columns "
                "WHERE table_name = 'chunks' AND column_name = 'embedding'"
            )
        )
        row = result.fetchone()
        assert row is not None
        # pgvector type appears as USER-DEFINED with udt_name = 'vector'
        assert row[1] == "vector", f"Expected 'vector' type, got '{row[1]}'"

    @pytest.mark.asyncio
    async def test_section_path_is_jsonb(self, db_session: AsyncSession):
        """section_path must be stored as JSONB."""
        result = await db_session.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'chunks' AND column_name = 'section_path'"
            )
        )
        row = result.fetchone()
        assert row is not None
        assert row[0] == "jsonb", f"Expected 'jsonb', got '{row[0]}'"
