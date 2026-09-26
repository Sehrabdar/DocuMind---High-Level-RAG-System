"""
Unit tests for Phase 5 — KeywordRetriever service.

All tests use a mocked KeywordRepository (no database connection).

Covers:
- Query validation (empty, whitespace, valid)
- top_k validation (zero, negative, exceeds max, None→default, bool rejected)
- Delegation to KeywordRepository with correct arguments
- Rank assignment (1-indexed, ordered by fts_score descending from repo)
- RetrievedChunk assembly (retrieval_method, fts_score, None distance/similarity)
- Provenance metadata preservation (content, document_id, section_path, etc.)
- document_id filter passed through to repository
- Empty database → empty result
- QueryValidationError is catchable as ValueError
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from documind.retrieval.keyword import KeywordRetriever
from documind.retrieval.models import RetrievedChunk
from documind.retrieval.validation import QueryValidationError

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_row(
    chunk_id: str = "a" * 64,
    document_id: str = "b" * 64,
    content: str = "PostgreSQL connection pooling example.",
    fts_score: float = 0.0759,
    chunk_index: int = 0,
    section_path: list | None = None,
) -> dict:
    """Return a dict that mimics a KeywordRepository row."""
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "content": content,
        "token_count": len(content.split()),
        "chunk_index": chunk_index,
        "section_path": section_path or [],
        "document_title": "Tech Docs",
        "page": None,
        "page_start": None,
        "page_end": None,
        "start_char": 0,
        "end_char": len(content),
        "fts_score": fts_score,
    }


def _make_mock_repo(rows: list[dict] | None = None) -> MagicMock:
    repo = MagicMock()
    repo.search = AsyncMock(return_value=rows or [])
    return repo


def _make_session() -> MagicMock:
    return MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# Query validation
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryValidation:

    @pytest.mark.asyncio
    async def test_empty_string_raises(self):
        retriever = KeywordRetriever(_make_mock_repo())
        with pytest.raises(QueryValidationError):
            await retriever.retrieve("", session=_make_session())

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self):
        retriever = KeywordRetriever(_make_mock_repo())
        with pytest.raises(QueryValidationError):
            await retriever.retrieve("   ", session=_make_session())

    @pytest.mark.asyncio
    async def test_tab_newline_raises(self):
        retriever = KeywordRetriever(_make_mock_repo())
        with pytest.raises(QueryValidationError):
            await retriever.retrieve("\t\n", session=_make_session())

    @pytest.mark.asyncio
    async def test_valid_query_does_not_raise(self):
        retriever = KeywordRetriever(_make_mock_repo())
        result = await retriever.retrieve("PostgreSQL connection pooling", session=_make_session())
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_query_stripped_before_passing_to_repo(self):
        repo = _make_mock_repo()
        retriever = KeywordRetriever(repo)
        await retriever.retrieve("  valid query  ", session=_make_session())
        _, kwargs = repo.search.call_args
        assert kwargs["query_text"] == "valid query"

    def test_query_validation_error_is_value_error(self):
        assert issubclass(QueryValidationError, ValueError)


# ─────────────────────────────────────────────────────────────────────────────
# top_k validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTopKValidation:

    def _retriever(self, max_top_k: int = 50) -> KeywordRetriever:
        return KeywordRetriever(_make_mock_repo(), max_top_k=max_top_k)

    @pytest.mark.asyncio
    async def test_none_uses_default_top_k(self):
        from config import settings
        retriever = self._retriever()
        await retriever.retrieve("query", top_k=None, session=_make_session())
        _, kwargs = retriever._repo.search.call_args
        assert kwargs["top_k"] == settings.retrieval_default_top_k

    @pytest.mark.asyncio
    async def test_valid_top_k_passed_through(self):
        retriever = self._retriever()
        await retriever.retrieve("query", top_k=7, session=_make_session())
        _, kwargs = retriever._repo.search.call_args
        assert kwargs["top_k"] == 7

    @pytest.mark.asyncio
    async def test_top_k_zero_raises(self):
        retriever = self._retriever()
        with pytest.raises(ValueError, match="top_k must be > 0"):
            await retriever.retrieve("query", top_k=0, session=_make_session())

    @pytest.mark.asyncio
    async def test_top_k_negative_raises(self):
        retriever = self._retriever()
        with pytest.raises(ValueError, match="top_k must be > 0"):
            await retriever.retrieve("query", top_k=-1, session=_make_session())

    @pytest.mark.asyncio
    async def test_top_k_exceeds_max_raises(self):
        retriever = self._retriever(max_top_k=10)
        with pytest.raises(ValueError, match="exceeds maximum"):
            await retriever.retrieve("query", top_k=11, session=_make_session())

    @pytest.mark.asyncio
    async def test_top_k_at_max_is_valid(self):
        retriever = self._retriever(max_top_k=10)
        await retriever.retrieve("query", top_k=10, session=_make_session())

    @pytest.mark.asyncio
    async def test_top_k_one_is_valid(self):
        retriever = self._retriever()
        result = await retriever.retrieve("query", top_k=1, session=_make_session())
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_top_k_bool_raises(self):
        retriever = self._retriever()
        with pytest.raises((ValueError, TypeError)):
            await retriever.retrieve("query", top_k=True, session=_make_session())  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Repository delegation
# ─────────────────────────────────────────────────────────────────────────────

class TestRepositoryDelegation:

    @pytest.mark.asyncio
    async def test_search_called_with_query_text(self):
        repo = _make_mock_repo()
        retriever = KeywordRetriever(repo)
        await retriever.retrieve("JWT authentication", top_k=3, session=_make_session())
        repo.search.assert_called_once()
        _, kwargs = repo.search.call_args
        assert kwargs["query_text"] == "JWT authentication"

    @pytest.mark.asyncio
    async def test_search_called_with_top_k(self):
        repo = _make_mock_repo()
        retriever = KeywordRetriever(repo)
        await retriever.retrieve("query", top_k=7, session=_make_session())
        _, kwargs = repo.search.call_args
        assert kwargs["top_k"] == 7

    @pytest.mark.asyncio
    async def test_document_id_passed_to_repo(self):
        repo = _make_mock_repo()
        retriever = KeywordRetriever(repo)
        doc_id = "d" * 64
        await retriever.retrieve("query", top_k=5, session=_make_session(), document_id=doc_id)
        _, kwargs = repo.search.call_args
        assert kwargs["document_id"] == doc_id

    @pytest.mark.asyncio
    async def test_no_document_id_passes_none(self):
        repo = _make_mock_repo()
        retriever = KeywordRetriever(repo)
        await retriever.retrieve("query", top_k=5, session=_make_session())
        _, kwargs = repo.search.call_args
        assert kwargs.get("document_id") is None


# ─────────────────────────────────────────────────────────────────────────────
# Empty results
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyResults:

    @pytest.mark.asyncio
    async def test_empty_repo_returns_empty_list(self):
        retriever = KeywordRetriever(_make_mock_repo(rows=[]))
        result = await retriever.retrieve("no match", session=_make_session())
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_result_is_list(self):
        retriever = KeywordRetriever(_make_mock_repo(rows=[]))
        result = await retriever.retrieve("query", session=_make_session())
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# RetrievedChunk assembly
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkAssembly:

    @pytest.mark.asyncio
    async def test_results_are_retrieved_chunks(self):
        rows = [_make_row(fts_score=0.1)]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert all(isinstance(r, RetrievedChunk) for r in results)

    @pytest.mark.asyncio
    async def test_retrieval_method_is_keyword(self):
        rows = [_make_row()]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].retrieval_method == "keyword"

    @pytest.mark.asyncio
    async def test_distance_is_none(self):
        rows = [_make_row()]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].distance is None

    @pytest.mark.asyncio
    async def test_similarity_is_none(self):
        rows = [_make_row()]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].similarity is None

    @pytest.mark.asyncio
    async def test_fts_score_populated(self):
        rows = [_make_row(fts_score=0.0759)]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert abs(results[0].fts_score - 0.0759) < 1e-6

    @pytest.mark.asyncio
    async def test_rank_starts_at_one(self):
        rows = [_make_row(chunk_id="a" * 64, fts_score=0.3),
                _make_row(chunk_id="b" * 64, fts_score=0.2)]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].rank == 1
        assert results[1].rank == 2

    @pytest.mark.asyncio
    async def test_rank_order_matches_repo_order(self):
        """Repo returns rows in descending fts_score order; retriever preserves it."""
        rows = [
            _make_row(chunk_id="a" * 64, fts_score=0.5),
            _make_row(chunk_id="b" * 64, fts_score=0.3),
            _make_row(chunk_id="c" * 64, fts_score=0.1),
        ]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        fts_scores = [r.fts_score for r in results]
        assert fts_scores == sorted(fts_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_content_preserved(self):
        content = "Redis sliding window rate limiting tracks request timestamps."
        rows = [_make_row(content=content)]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].content == content

    @pytest.mark.asyncio
    async def test_document_id_preserved(self):
        doc_id = "d" * 64
        rows = [_make_row(document_id=doc_id)]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].document_id == doc_id

    @pytest.mark.asyncio
    async def test_section_path_preserved(self):
        rows = [_make_row(section_path=["Auth", "JWT"])]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].section_path == ["Auth", "JWT"]

    @pytest.mark.asyncio
    async def test_chunk_id_preserved(self):
        cid = "c" * 64
        rows = [_make_row(chunk_id=cid)]
        retriever = KeywordRetriever(_make_mock_repo(rows=rows))
        results = await retriever.retrieve("query", session=_make_session())
        assert results[0].chunk_id == cid


# ─────────────────────────────────────────────────────────────────────────────
# RetrievedChunk model: Phase 5 extension correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievedChunkPhase5:

    def _kw_chunk(self, **kwargs) -> RetrievedChunk:
        defaults = dict(
            chunk_id="a" * 64,
            document_id="b" * 64,
            content="test",
            token_count=1,
            chunk_index=0,
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=4,
            fts_score=0.1,
            rank=1,
        )
        defaults.update(kwargs)
        return RetrievedChunk.from_fts_row(**defaults)

    def _dense_chunk(self, **kwargs) -> RetrievedChunk:
        defaults = dict(
            chunk_id="a" * 64,
            document_id="b" * 64,
            content="test",
            token_count=1,
            chunk_index=0,
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=4,
            distance=0.25,
            rank=1,
        )
        defaults.update(kwargs)
        return RetrievedChunk.from_row(**defaults)

    def test_keyword_chunk_method_is_keyword(self):
        assert self._kw_chunk().retrieval_method == "keyword"

    def test_dense_chunk_method_is_dense(self):
        assert self._dense_chunk().retrieval_method == "dense"

    def test_keyword_chunk_distance_is_none(self):
        assert self._kw_chunk().distance is None

    def test_keyword_chunk_similarity_is_none(self):
        assert self._kw_chunk().similarity is None

    def test_dense_chunk_fts_score_is_none(self):
        assert self._dense_chunk().fts_score is None

    def test_dense_chunk_distance_set(self):
        chunk = self._dense_chunk(distance=0.3)
        assert abs(chunk.distance - 0.3) < 1e-9

    def test_dense_chunk_similarity_derived(self):
        chunk = self._dense_chunk(distance=0.3)
        assert abs(chunk.similarity - 0.7) < 1e-9

    def test_keyword_chunk_fts_score_set(self):
        chunk = self._kw_chunk(fts_score=0.0759)
        assert abs(chunk.fts_score - 0.0759) < 1e-9

    def test_rank_zero_raises(self):
        with pytest.raises(ValueError):
            self._kw_chunk(rank=0)

    def test_rank_negative_raises(self):
        with pytest.raises(ValueError):
            self._kw_chunk(rank=-1)

    def test_keyword_format_summary_contains_fts(self):
        chunk = self._kw_chunk(fts_score=0.0759, rank=1)
        summary = chunk.format_summary()
        assert "keyword" in summary
        assert "fts=" in summary

    def test_dense_format_summary_contains_dist(self):
        chunk = self._dense_chunk(distance=0.25, rank=1)
        summary = chunk.format_summary()
        assert "dense" in summary
        assert "dist=" in summary

    def test_json_serializable(self):
        import json
        chunk = self._kw_chunk()
        j = chunk.model_dump_json()
        d = json.loads(j)
        assert d["retrieval_method"] == "keyword"
        assert d["distance"] is None
        assert d["similarity"] is None
        assert d["fts_score"] is not None
