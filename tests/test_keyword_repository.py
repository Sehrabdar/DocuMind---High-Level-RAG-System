"""
Unit tests for Phase 5 — KeywordRepository.

Tests verify the repository's SQL construction strategy using a mocked
AsyncSession.  We verify:
- The repository calls session.execute() (not Python-side filtering)
- The SQL string contains expected FTS operators (@@ and websearch_to_tsquery)
- Parameters passed to execute() match the input arguments
- The document_id filter clause is included/excluded correctly
- Results are returned as list[dict]

We do NOT test PostgreSQL FTS correctness here — that is the job of integration tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from db.keyword_repository import KeywordRepository


def _make_session(rows: list[dict] | None = None) -> MagicMock:
    """Return a mocked AsyncSession that yields the given rows."""
    rows = rows or []
    mapping_mock = MagicMock()
    mapping_mock.fetchall.return_value = [dict(r) for r in rows]

    result_mock = MagicMock()
    result_mock.mappings.return_value = mapping_mock

    session = MagicMock()
    session.execute = AsyncMock(return_value=result_mock)
    return session


def _sample_row(chunk_id: str = "a" * 64, fts_score: float = 0.1) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "b" * 64,
        "content": "PostgreSQL connection pooling.",
        "token_count": 4,
        "chunk_index": 0,
        "section_path": [],
        "document_title": None,
        "page": None,
        "page_start": None,
        "page_end": None,
        "start_char": 0,
        "end_char": 30,
        "fts_score": fts_score,
    }


class TestKeywordRepositorySQL:
    """Verify that the repository uses PostgreSQL FTS, not Python filtering."""

    @pytest.mark.asyncio
    async def test_session_execute_called(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("connection pooling", top_k=5, session=session)
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_sql_contains_fts_operator(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("connection pooling", top_k=5, session=session)
        call_args = session.execute.call_args
        sql_obj = call_args[0][0]
        sql_text = str(sql_obj)
        assert "@@" in sql_text

    @pytest.mark.asyncio
    async def test_sql_contains_websearch_to_tsquery(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("JWT authentication", top_k=5, session=session)
        call_args = session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "websearch_to_tsquery" in sql_text

    @pytest.mark.asyncio
    async def test_sql_contains_ts_rank_cd(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("rate limiting", top_k=5, session=session)
        call_args = session.execute.call_args
        sql_text = str(call_args[0][0])
        assert "ts_rank_cd" in sql_text

    @pytest.mark.asyncio
    async def test_sql_contains_order_by_desc(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("query", top_k=5, session=session)
        call_args = session.execute.call_args
        sql_text = str(call_args[0][0]).upper()
        assert "ORDER BY" in sql_text
        assert "DESC" in sql_text

    @pytest.mark.asyncio
    async def test_sql_contains_limit(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("query", top_k=5, session=session)
        call_args = session.execute.call_args
        sql_text = str(call_args[0][0]).upper()
        assert "LIMIT" in sql_text

    @pytest.mark.asyncio
    async def test_query_text_in_params(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("PostgreSQL async sessions", top_k=5, session=session)
        call_args = session.execute.call_args
        params = call_args[0][1]
        assert params.get("query_text") == "PostgreSQL async sessions"

    @pytest.mark.asyncio
    async def test_top_k_in_params(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("query", top_k=7, session=session)
        call_args = session.execute.call_args
        params = call_args[0][1]
        assert params.get("top_k") == 7


class TestKeywordRepositoryDocumentFilter:

    @pytest.mark.asyncio
    async def test_document_id_filter_in_sql(self):
        session = _make_session()
        repo = KeywordRepository()
        doc_id = "d" * 64
        await repo.search("query", top_k=5, session=session, document_id=doc_id)
        call_args = session.execute.call_args
        sql_text = str(call_args[0][0])
        params = call_args[0][1]
        assert "document_id" in sql_text
        assert params.get("document_id") == doc_id

    @pytest.mark.asyncio
    async def test_no_document_id_no_filter_in_params(self):
        session = _make_session()
        repo = KeywordRepository()
        await repo.search("query", top_k=5, session=session)
        call_args = session.execute.call_args
        params = call_args[0][1]
        assert "document_id" not in params


class TestKeywordRepositoryResults:

    @pytest.mark.asyncio
    async def test_returns_list(self):
        session = _make_session([_sample_row()])
        repo = KeywordRepository()
        result = await repo.search("query", top_k=5, session=session)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_returns_dicts(self):
        session = _make_session([_sample_row()])
        repo = KeywordRepository()
        result = await repo.search("query", top_k=5, session=session)
        assert all(isinstance(r, dict) for r in result)

    @pytest.mark.asyncio
    async def test_empty_database_returns_empty_list(self):
        session = _make_session([])
        repo = KeywordRepository()
        result = await repo.search("query", top_k=5, session=session)
        assert result == []

    @pytest.mark.asyncio
    async def test_result_contains_expected_keys(self):
        session = _make_session([_sample_row()])
        repo = KeywordRepository()
        result = await repo.search("query", top_k=5, session=session)
        row = result[0]
        for key in ("chunk_id", "document_id", "content", "fts_score"):
            assert key in row, f"Expected key {key!r} in row"

    @pytest.mark.asyncio
    async def test_multiple_rows_preserved(self):
        rows = [_sample_row("a" * 64, fts_score=0.3), _sample_row("b" * 64, fts_score=0.1)]
        session = _make_session(rows)
        repo = KeywordRepository()
        result = await repo.search("query", top_k=5, session=session)
        assert len(result) == 2
