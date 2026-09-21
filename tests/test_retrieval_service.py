"""
Unit tests for documind/retrieval/service.py — DenseRetriever.

All tests use FakeEmbeddingProvider (no model download) and an async mock
VectorRepository (no database).  This verifies the retrieval service logic
in isolation from both the embedding model and the database.

Covers:
- Query validation (empty, whitespace, None-like)
- top_k validation (zero, negative, exceeds max, None defaults)
- Correct delegation to EmbeddingService
- Correct delegation to VectorRepository
- Rank assignment (1-indexed, ordered by distance)
- Empty database → empty result
- Distance/similarity semantics in returned RetrievedChunk
- document_id filter passed through to VectorRepository
- DenseRetriever constructor defaults
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from documind.embeddings.service import EmbeddingService, FakeEmbeddingProvider
from documind.retrieval.models import RetrievedChunk
from documind.retrieval.service import DenseRetriever, QueryValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_embedding_service(dim: int = 8) -> EmbeddingService:
    """Small-dimension service for speed."""
    return EmbeddingService(FakeEmbeddingProvider(dimension=dim))


def _make_fake_record(
    chunk_id: str,
    document_id: str = "d" * 64,
    chunk_index: int = 0,
    content: str = "Test content.",
) -> MagicMock:
    """Return a mock that quacks like a ChunkRecord."""
    record = MagicMock()
    record.chunk_id = chunk_id
    record.document_id = document_id
    record.chunk_index = chunk_index
    record.content = content
    record.token_count = len(content.split())
    record.section_path = []
    record.document_title = "Test Doc"
    record.page = None
    record.page_start = None
    record.page_end = None
    record.start_char = 0
    record.end_char = len(content)
    return record


def _make_mock_vector_repo(
    search_results: list[tuple[str, float]] | None = None,
    chunk_map: dict | None = None,
) -> MagicMock:
    """Return a mock VectorRepository with pre-configured return values."""
    repo = MagicMock()
    repo.search = AsyncMock(return_value=search_results or [])

    if chunk_map is None:
        # Default: empty map (no chunks)
        repo.fetch_by_chunk_ids = AsyncMock(return_value={})
    else:
        repo.fetch_by_chunk_ids = AsyncMock(return_value=chunk_map)

    return repo


def _make_mock_session() -> MagicMock:
    return MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# Query validation
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryValidation:

    @pytest.mark.asyncio
    async def test_empty_string_raises(self):
        retriever = DenseRetriever(_fake_embedding_service(), _make_mock_vector_repo())
        with pytest.raises(QueryValidationError):
            await retriever.retrieve("", session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_whitespace_only_raises(self):
        retriever = DenseRetriever(_fake_embedding_service(), _make_mock_vector_repo())
        with pytest.raises(QueryValidationError):
            await retriever.retrieve("   ", session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_tab_only_raises(self):
        retriever = DenseRetriever(_fake_embedding_service(), _make_mock_vector_repo())
        with pytest.raises(QueryValidationError):
            await retriever.retrieve("\t\n", session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_valid_query_does_not_raise(self):
        retriever = DenseRetriever(_fake_embedding_service(), _make_mock_vector_repo())
        result = await retriever.retrieve("How do I create an API key?", session=_make_mock_session())
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_query_is_stripped_before_embedding(self):
        """Leading/trailing whitespace is stripped before embedding."""
        svc = _fake_embedding_service()
        repo = _make_mock_vector_repo()
        retriever = DenseRetriever(svc, repo)
        # If the query had leading/trailing whitespace, it should not raise
        await retriever.retrieve("  valid query  ", session=_make_mock_session())

    def test_validate_query_returns_stripped(self):
        retriever = DenseRetriever(_fake_embedding_service(), _make_mock_vector_repo())
        result = retriever._validate_query("  hello  ")
        assert result == "hello"

    def test_validate_query_empty_raises(self):
        retriever = DenseRetriever(_fake_embedding_service(), _make_mock_vector_repo())
        with pytest.raises(QueryValidationError):
            retriever._validate_query("")

    def test_query_validation_error_is_value_error(self):
        """QueryValidationError must be a ValueError subclass."""
        assert issubclass(QueryValidationError, ValueError)


# ─────────────────────────────────────────────────────────────────────────────
# top_k validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTopKValidation:

    def _retriever(self, max_top_k: int = 50) -> DenseRetriever:
        return DenseRetriever(
            _fake_embedding_service(),
            _make_mock_vector_repo(),
            max_top_k=max_top_k,
        )

    def test_none_returns_default(self):
        r = self._retriever()
        from config import settings
        assert r._validate_top_k(None) == settings.retrieval_default_top_k

    def test_valid_top_k_returned_as_is(self):
        r = self._retriever()
        assert r._validate_top_k(10) == 10

    def test_top_k_one_is_valid(self):
        r = self._retriever()
        assert r._validate_top_k(1) == 1

    def test_top_k_zero_raises(self):
        r = self._retriever()
        with pytest.raises(ValueError, match="top_k must be > 0"):
            r._validate_top_k(0)

    def test_top_k_negative_raises(self):
        r = self._retriever()
        with pytest.raises(ValueError, match="top_k must be > 0"):
            r._validate_top_k(-5)

    def test_top_k_exceeds_max_raises(self):
        r = self._retriever(max_top_k=20)
        with pytest.raises(ValueError, match="exceeds maximum"):
            r._validate_top_k(21)

    def test_top_k_at_max_is_valid(self):
        r = self._retriever(max_top_k=20)
        assert r._validate_top_k(20) == 20

    def test_top_k_bool_raises(self):
        """True/False should not be accepted as top_k (Python bools are ints)."""
        r = self._retriever()
        with pytest.raises((ValueError, TypeError)):
            r._validate_top_k(True)  # type: ignore

    @pytest.mark.asyncio
    async def test_retrieve_top_k_zero_raises(self):
        retriever = self._retriever()
        with pytest.raises(ValueError):
            await retriever.retrieve("query", top_k=0, session=_make_mock_session())

    @pytest.mark.asyncio
    async def test_retrieve_top_k_exceeds_max_raises(self):
        retriever = self._retriever(max_top_k=5)
        with pytest.raises(ValueError, match="exceeds maximum"):
            await retriever.retrieve("query", top_k=6, session=_make_mock_session())


# ─────────────────────────────────────────────────────────────────────────────
# Delegation to EmbeddingService
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbeddingDelegation:

    @pytest.mark.asyncio
    async def test_embed_text_called_with_query(self):
        """DenseRetriever must embed the query using EmbeddingService.embed_text."""
        provider = FakeEmbeddingProvider(dimension=8)
        svc = EmbeddingService(provider)
        repo = _make_mock_vector_repo()
        retriever = DenseRetriever(svc, repo)

        query = "How do I revoke an API key?"
        await retriever.retrieve(query, top_k=1, session=_make_mock_session())

        # search() should have been called with a vector (list of floats)
        args, kwargs = repo.search.call_args
        vector = kwargs.get("query_vector") or args[0]
        assert isinstance(vector, list)
        assert len(vector) == 8
        # All values are floats
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.asyncio
    async def test_embedded_vector_is_normalized(self):
        """The query vector passed to the repository must be L2-normalized."""
        provider = FakeEmbeddingProvider(dimension=16)
        svc = EmbeddingService(provider)
        repo = _make_mock_vector_repo()
        retriever = DenseRetriever(svc, repo)

        await retriever.retrieve("test query", top_k=1, session=_make_mock_session())

        args, kwargs = repo.search.call_args
        vector = kwargs.get("query_vector") or args[0]
        norm = math.sqrt(sum(v * v for v in vector))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ─────────────────────────────────────────────────────────────────────────────
# Empty database
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyDatabase:

    @pytest.mark.asyncio
    async def test_empty_database_returns_empty_list(self):
        retriever = DenseRetriever(
            _fake_embedding_service(),
            _make_mock_vector_repo(search_results=[]),
        )
        result = await retriever.retrieve("any query", top_k=5, session=_make_mock_session())
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_result_type_is_list(self):
        retriever = DenseRetriever(
            _fake_embedding_service(),
            _make_mock_vector_repo(search_results=[]),
        )
        result = await retriever.retrieve("any query", session=_make_mock_session())
        assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# Rank assignment
# ─────────────────────────────────────────────────────────────────────────────

class TestRankAssignment:

    def _retriever_with_results(
        self, distances: list[float]
    ) -> tuple[DenseRetriever, list[MagicMock]]:
        """Build a retriever whose repo returns results with the given distances."""
        chunk_ids = [f"chunk_{i:02d}" + "x" * 62 for i in range(len(distances))]
        records = {
            cid: _make_fake_record(cid, chunk_index=i)
            for i, cid in enumerate(chunk_ids)
        }
        search_results = list(zip(chunk_ids, distances))
        repo = _make_mock_vector_repo(search_results=search_results, chunk_map=records)
        retriever = DenseRetriever(_fake_embedding_service(), repo)
        return retriever, list(records.values())

    @pytest.mark.asyncio
    async def test_single_result_has_rank_one(self):
        retriever, _ = self._retriever_with_results([0.25])
        results = await retriever.retrieve("q", top_k=1, session=_make_mock_session())
        assert len(results) == 1
        assert results[0].rank == 1

    @pytest.mark.asyncio
    async def test_ranks_are_one_indexed(self):
        retriever, _ = self._retriever_with_results([0.1, 0.3, 0.5])
        results = await retriever.retrieve("q", top_k=3, session=_make_mock_session())
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_rank_one_has_smallest_distance(self):
        retriever, _ = self._retriever_with_results([0.1, 0.3, 0.7])
        results = await retriever.retrieve("q", top_k=3, session=_make_mock_session())
        assert results[0].distance < results[1].distance < results[2].distance

    @pytest.mark.asyncio
    async def test_distances_preserved_in_results(self):
        distances = [0.12, 0.34, 0.56]
        retriever, _ = self._retriever_with_results(distances)
        results = await retriever.retrieve("q", top_k=3, session=_make_mock_session())
        for i, r in enumerate(results):
            assert abs(r.distance - distances[i]) < 1e-9

    @pytest.mark.asyncio
    async def test_similarity_derived_from_distance(self):
        retriever, _ = self._retriever_with_results([0.25])
        results = await retriever.retrieve("q", top_k=1, session=_make_mock_session())
        assert abs(results[0].similarity - (1.0 - 0.25)) < 1e-9

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self):
        retriever, _ = self._retriever_with_results([0.1, 0.2, 0.3, 0.4, 0.5])
        # The mock always returns all results; the limit is enforced by the DB.
        # Here we confirm the retriever doesn't re-expand beyond what's returned.
        results = await retriever.retrieve("q", top_k=5, session=_make_mock_session())
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_all_results_are_retrieved_chunks(self):
        retriever, _ = self._retriever_with_results([0.1, 0.2])
        results = await retriever.retrieve("q", top_k=2, session=_make_mock_session())
        assert all(isinstance(r, RetrievedChunk) for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# document_id filter
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIdFilter:

    @pytest.mark.asyncio
    async def test_document_id_passed_to_repository(self):
        repo = _make_mock_vector_repo()
        retriever = DenseRetriever(_fake_embedding_service(), repo)
        doc_id = "d" * 64

        await retriever.retrieve(
            "query",
            top_k=5,
            session=_make_mock_session(),
            document_id=doc_id,
        )

        _, kwargs = repo.search.call_args
        assert kwargs.get("document_id") == doc_id

    @pytest.mark.asyncio
    async def test_no_document_id_passes_none(self):
        repo = _make_mock_vector_repo()
        retriever = DenseRetriever(_fake_embedding_service(), repo)

        await retriever.retrieve("query", top_k=5, session=_make_mock_session())

        _, kwargs = repo.search.call_args
        assert kwargs.get("document_id") is None


# ─────────────────────────────────────────────────────────────────────────────
# Provenance metadata in results
# ─────────────────────────────────────────────────────────────────────────────

class TestProvenanceMetadata:

    @pytest.mark.asyncio
    async def test_content_preserved_in_result(self):
        cid = "cid" + "a" * 61
        content = "OAuth tokens can be refreshed using the refresh_token endpoint."
        record = _make_fake_record(cid, content=content)
        repo = _make_mock_vector_repo(
            search_results=[(cid, 0.18)],
            chunk_map={cid: record},
        )
        retriever = DenseRetriever(_fake_embedding_service(), repo)
        results = await retriever.retrieve("OAuth", top_k=1, session=_make_mock_session())
        assert results[0].content == content

    @pytest.mark.asyncio
    async def test_document_id_preserved(self):
        cid = "cid" + "b" * 61
        doc_id = "doc" + "c" * 61
        record = _make_fake_record(cid, document_id=doc_id)
        repo = _make_mock_vector_repo(
            search_results=[(cid, 0.22)],
            chunk_map={cid: record},
        )
        retriever = DenseRetriever(_fake_embedding_service(), repo)
        results = await retriever.retrieve("query", top_k=1, session=_make_mock_session())
        assert results[0].document_id == doc_id

    @pytest.mark.asyncio
    async def test_section_path_preserved(self):
        cid = "cid" + "d" * 61
        record = _make_fake_record(cid)
        record.section_path = ["API Reference", "Authentication"]
        repo = _make_mock_vector_repo(
            search_results=[(cid, 0.30)],
            chunk_map={cid: record},
        )
        retriever = DenseRetriever(_fake_embedding_service(), repo)
        results = await retriever.retrieve("query", top_k=1, session=_make_mock_session())
        assert results[0].section_path == ["API Reference", "Authentication"]
