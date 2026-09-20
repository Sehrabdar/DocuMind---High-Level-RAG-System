"""
Integration tests for ChunkRepository persistence operations.

Covers:
- upsert_chunks inserts new rows
- upsert_chunks with the same chunk_id updates the row (idempotency)
- count() returns correct total
- get_by_document_id() returns chunks in chunk_index order
- delete_by_document_id() removes rows
- get_by_document_id returns empty list for unknown document

Requires running PostgreSQL + pgvector + alembic upgrade head.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from db.models import ChunkRecord
from db.repository import ChunkRepository
from tests.integration.conftest import requires_db


# ─────────────────────────────────────────────────────────────────────────────
# Test ID helpers — all IDs are exactly 64 hex-safe chars
# ─────────────────────────────────────────────────────────────────────────────

def _cid(prefix: str, suffix: str = "") -> str:
    """Return a 64-char chunk_id padded with 'f'."""
    raw = prefix + suffix
    return (raw + "f" * 64)[:64]


def _did(prefix: str, suffix: str = "") -> str:
    """Return a 64-char document_id padded with 'e'."""
    raw = prefix + suffix
    return (raw + "e" * 64)[:64]


def _make_embedding(dim: int = 384, seed: int = 0) -> list[float]:
    """Deterministic unit-norm embedding."""
    raw = [float((i + seed) % 10 + 1) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw))
    return [v / norm for v in raw]


def _make_chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int = 0,
    content: str = "Test content.",
    section_path: list[str] | None = None,
    page: int | None = None,
) -> ChunkRecord:
    assert len(chunk_id) <= 64, f"chunk_id too long: {len(chunk_id)}"
    assert len(document_id) <= 64, f"document_id too long: {len(document_id)}"
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        token_count=len(content.split()),
        embedding=_make_embedding(seed=chunk_index),
        section_path=section_path or [],
        document_title="Test Document",
        page=page,
        page_start=None,
        page_end=None,
        start_char=0,
        end_char=len(content),
        created_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Upsert
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
class TestChunkRepositoryUpsert:

    @pytest.mark.asyncio
    async def test_upsert_inserts_new_chunk(self, db_session):
        repo = ChunkRepository()
        record = _make_chunk(_cid("insert_test"), _did("doc_a"))
        count = await repo.upsert_chunks([record], db_session)
        assert count >= 1

    @pytest.mark.asyncio
    async def test_upsert_returns_count(self, db_session):
        repo = ChunkRepository()
        doc_id = _did("count_doc")
        records = [
            _make_chunk(_cid("count_chunk", str(i)), doc_id, chunk_index=i)
            for i in range(5)
        ]
        count = await repo.upsert_chunks(records, db_session)
        assert count == 5

    @pytest.mark.asyncio
    async def test_upsert_empty_list_returns_zero(self, db_session):
        repo = ChunkRepository()
        count = await repo.upsert_chunks([], db_session)
        assert count == 0

    @pytest.mark.asyncio
    async def test_upsert_idempotent_same_chunk_id(self, db_session):
        """Upserting the same chunk_id twice must not create duplicate rows."""
        repo = ChunkRepository()
        cid = _cid("idem_chunk")
        doc_id = _did("idem_doc")

        record1 = _make_chunk(cid, doc_id, content="Original content.")
        await repo.upsert_chunks([record1], db_session)

        record2 = _make_chunk(cid, doc_id, content="Updated content.")
        await repo.upsert_chunks([record2], db_session)

        chunks = await repo.get_by_document_id(doc_id, db_session)
        matching = [c for c in chunks if c.chunk_id == cid]
        assert len(matching) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_content_on_conflict(self, db_session):
        """On conflict, content is updated to the new value."""
        repo = ChunkRepository()
        cid = _cid("update_content_chunk")
        doc_id = _did("update_content_doc")

        record1 = _make_chunk(cid, doc_id, content="Original content.")
        await repo.upsert_chunks([record1], db_session)

        record2 = _make_chunk(cid, doc_id, content="Updated content after re-ingestion.")
        await repo.upsert_chunks([record2], db_session)

        chunks = await repo.get_by_document_id(doc_id, db_session)
        matching = [c for c in chunks if c.chunk_id == cid]
        assert matching[0].content == "Updated content after re-ingestion."


# ─────────────────────────────────────────────────────────────────────────────
# Fetch
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
class TestChunkRepositoryFetch:

    @pytest.mark.asyncio
    async def test_get_by_document_id_returns_correct_chunks(self, db_session):
        repo = ChunkRepository()
        doc_id = _did("fetch_doc")
        records = [
            _make_chunk(_cid("fetch_chunk", str(i)), doc_id, chunk_index=i)
            for i in range(3)
        ]
        await repo.upsert_chunks(records, db_session)
        retrieved = await repo.get_by_document_id(doc_id, db_session)
        assert len(retrieved) == 3

    @pytest.mark.asyncio
    async def test_get_by_document_id_ordered_by_chunk_index(self, db_session):
        repo = ChunkRepository()
        doc_id = _did("order_doc")
        # Insert in reverse order to verify the query sorts correctly
        records = [
            _make_chunk(_cid("order_chunk", str(i)), doc_id, chunk_index=i)
            for i in [2, 0, 1]
        ]
        await repo.upsert_chunks(records, db_session)
        retrieved = await repo.get_by_document_id(doc_id, db_session)
        indexes = [r.chunk_index for r in retrieved]
        assert indexes == sorted(indexes), f"Expected sorted indexes, got {indexes}"

    @pytest.mark.asyncio
    async def test_get_by_document_id_empty_for_unknown_doc(self, db_session):
        repo = ChunkRepository()
        result = await repo.get_by_document_id(_did("no_such_doc"), db_session)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_by_document_id_does_not_return_other_docs(self, db_session):
        repo = ChunkRepository()
        doc_a = _did("iso_doc_a")
        doc_b = _did("iso_doc_b")
        record_a = _make_chunk(_cid("iso_chunk_a"), doc_a, content="Doc A content.")
        record_b = _make_chunk(_cid("iso_chunk_b"), doc_b, content="Doc B content.")
        await repo.upsert_chunks([record_a, record_b], db_session)
        result_a = await repo.get_by_document_id(doc_a, db_session)
        assert all(r.document_id == doc_a for r in result_a)


# ─────────────────────────────────────────────────────────────────────────────
# Count
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
class TestChunkRepositoryCount:

    @pytest.mark.asyncio
    async def test_count_returns_integer(self, db_session):
        repo = ChunkRepository()
        result = await repo.count(db_session)
        assert isinstance(result, int)
        assert result >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Delete
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
class TestChunkRepositoryDelete:

    @pytest.mark.asyncio
    async def test_delete_by_document_id_removes_chunks(self, db_session):
        repo = ChunkRepository()
        doc_id = _did("delete_doc")
        records = [
            _make_chunk(_cid("del_chunk", str(i)), doc_id, chunk_index=i)
            for i in range(3)
        ]
        await repo.upsert_chunks(records, db_session)
        deleted = await repo.delete_by_document_id(doc_id, db_session)
        assert deleted == 3
        remaining = await repo.get_by_document_id(doc_id, db_session)
        assert remaining == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_zero(self, db_session):
        repo = ChunkRepository()
        deleted = await repo.delete_by_document_id(_did("no_such_doc2"), db_session)
        assert deleted == 0


# ─────────────────────────────────────────────────────────────────────────────
# Section path persistence
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
class TestSectionPathPersistence:

    @pytest.mark.asyncio
    async def test_section_path_is_persisted_correctly(self, db_session):
        repo = ChunkRepository()
        path = ["Authentication", "OAuth", "Token Refresh"]
        doc_id = _did("section_path_doc")
        record = _make_chunk(_cid("section_path_chunk"), doc_id, section_path=path)
        await repo.upsert_chunks([record], db_session)
        retrieved = await repo.get_by_document_id(doc_id, db_session)
        assert len(retrieved) == 1
        assert retrieved[0].section_path == path

    @pytest.mark.asyncio
    async def test_empty_section_path_persisted(self, db_session):
        repo = ChunkRepository()
        doc_id = _did("empty_path_doc")
        record = _make_chunk(_cid("empty_path_chunk"), doc_id, section_path=[])
        await repo.upsert_chunks([record], db_session)
        retrieved = await repo.get_by_document_id(doc_id, db_session)
        assert retrieved[0].section_path == []
