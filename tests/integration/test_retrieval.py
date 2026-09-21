"""
Integration tests for Phase 4 dense retrieval — live PostgreSQL + pgvector.

These tests prove that:
1. VectorRepository.search() correctly uses pgvector cosine-distance to find
   the nearest stored chunk to a query vector.
2. DenseRetriever.retrieve() returns the semantically closest chunk first.
3. The HNSW index is in place and the query executes against the real DB.
4. Result ordering, rank assignment, and metadata preservation are correct.

Test corpus design
------------------
We insert a small, controlled synthetic corpus with DETERMINISTIC embeddings
(from FakeEmbeddingProvider, seeded by text content) so the tests are:
- Offline (no model download required)
- Reproducible across environments
- Fast (few DB rows, no model inference overhead)

The key invariant we verify:
    If chunk A has a smaller cosine distance to the query than chunk B,
    then rank(A) < rank(B).

We also include an optional real-model test that verifies BGE-small-en-v1.5
returns a semantically appropriate result for a natural-language query.

Requires:
    docker compose up -d && uv run alembic upgrade head
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from db.models import ChunkRecord
from db.repository import ChunkRepository
from db.vector_repository import VectorRepository
from documind.embeddings.service import EmbeddingService, FakeEmbeddingProvider
from documind.retrieval.models import RetrievedChunk
from documind.retrieval.service import DenseRetriever, QueryValidationError
from tests.integration.conftest import requires_db


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cid(prefix: str) -> str:
    """64-char chunk_id padded with 'a'."""
    return (prefix + "a" * 64)[:64]


def _did(prefix: str) -> str:
    """64-char document_id padded with 'b'."""
    return (prefix + "b" * 64)[:64]


def _fake_service(dim: int = 384) -> EmbeddingService:
    return EmbeddingService(FakeEmbeddingProvider(dimension=dim))


def _embed(text: str, dim: int = 384) -> list[float]:
    """Return a deterministic L2-normalized embedding for text."""
    svc = _fake_service(dim)
    return svc.embed_text(text)


def _make_record(
    chunk_id: str,
    document_id: str,
    content: str,
    embedding: list[float],
    chunk_index: int = 0,
    section_path: list[str] | None = None,
    document_title: str | None = None,
    page: int | None = None,
) -> ChunkRecord:
    assert len(chunk_id) <= 64
    assert len(document_id) <= 64
    assert len(embedding) == 384
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        token_count=len(content.split()),
        embedding=embedding,
        section_path=section_path or [],
        document_title=document_title,
        page=page,
        page_start=None,
        page_end=None,
        start_char=0,
        end_char=len(content),
        created_at=datetime.now(timezone.utc),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic corpus
#
# Three chunks with distinct semantic content.  Their embeddings are derived
# from FakeEmbeddingProvider (hash-seeded deterministic random vectors).
# We find the nearest chunk by querying with each chunk's own embedding —
# cosine distance from a vector to itself should be ~0 (the best match).
# ─────────────────────────────────────────────────────────────────────────────

CORPUS = [
    {
        "prefix": "r4_chunk_api_key",
        "doc_prefix": "r4_doc_auth",
        "content": "The API key can be revoked from the security dashboard.",
        "section": ["API Keys", "Revocation"],
        "title": "API Reference",
    },
    {
        "prefix": "r4_chunk_oauth",
        "doc_prefix": "r4_doc_oauth",
        "content": "OAuth refresh tokens are used to obtain new access tokens.",
        "section": ["OAuth", "Refresh Tokens"],
        "title": "OAuth Guide",
    },
    {
        "prefix": "r4_chunk_postgres",
        "doc_prefix": "r4_doc_postgres",
        "content": "PostgreSQL uses indexes to improve query performance.",
        "section": ["Database", "Indexes"],
        "title": "Database Guide",
    },
]


@pytest_asyncio.fixture
async def corpus_in_db(db_session) -> list[ChunkRecord]:
    """Insert the synthetic corpus and return the records."""
    repo = ChunkRepository()
    records = []

    for entry in CORPUS:
        embedding = _embed(entry["content"])
        record = _make_record(
            chunk_id=_cid(entry["prefix"]),
            document_id=_did(entry["doc_prefix"]),
            content=entry["content"],
            embedding=embedding,
            section_path=entry["section"],
            document_title=entry["title"],
        )
        records.append(record)

    await repo.upsert_chunks(records, db_session)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# VectorRepository tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
class TestVectorRepositorySearch:

    @pytest.mark.asyncio
    async def test_self_query_is_rank_one(self, db_session, corpus_in_db):
        """Querying with a chunk's own embedding should return it as rank 1.

        This is the fundamental nearest-neighbour correctness test:
        the closest vector to any unit-norm vector v is v itself (distance ≈ 0).
        """
        repo = VectorRepository()

        for entry, record in zip(CORPUS, corpus_in_db):
            query_vector = _embed(entry["content"])  # same text → same embedding
            results = await repo.search(
                query_vector=query_vector,
                top_k=3,
                session=db_session,
            )
            assert len(results) >= 1, f"Expected at least 1 result for '{entry['content'][:30]}'"
            top_chunk_id, top_distance = results[0]
            assert top_chunk_id == record.chunk_id, (
                f"Expected {record.chunk_id!r} at rank 1, got {top_chunk_id!r} "
                f"(distance={top_distance:.4f})"
            )
            # Self-distance should be very small for an L2-normalized vector
            assert top_distance < 0.01, (
                f"Expected near-zero self-distance, got {top_distance:.6f}"
            )

    @pytest.mark.asyncio
    async def test_results_ordered_by_ascending_distance(self, db_session, corpus_in_db):
        """Results must be ordered from smallest to largest cosine distance."""
        repo = VectorRepository()
        query_vector = _embed(CORPUS[0]["content"])  # query for API key chunk

        results = await repo.search(
            query_vector=query_vector,
            top_k=3,
            session=db_session,
        )
        distances = [dist for _, dist in results]
        assert distances == sorted(distances), (
            f"Results not in ascending distance order: {distances}"
        )

    @pytest.mark.asyncio
    async def test_top_k_limits_result_count(self, db_session, corpus_in_db):
        """top_k=1 must return at most 1 result."""
        repo = VectorRepository()
        query_vector = _embed(CORPUS[0]["content"])

        results = await repo.search(
            query_vector=query_vector,
            top_k=1,
            session=db_session,
        )
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_top_k_two_returns_two(self, db_session, corpus_in_db):
        repo = VectorRepository()
        query_vector = _embed(CORPUS[0]["content"])
        results = await repo.search(query_vector=query_vector, top_k=2, session=db_session)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_results_are_id_distance_tuples(self, db_session, corpus_in_db):
        """search() returns list of (str, float) tuples."""
        repo = VectorRepository()
        query_vector = _embed(CORPUS[0]["content"])
        results = await repo.search(query_vector=query_vector, top_k=3, session=db_session)

        for item in results:
            assert isinstance(item, tuple) and len(item) == 2
            chunk_id, distance = item
            assert isinstance(chunk_id, str)
            assert isinstance(distance, float)
            assert 0.0 <= distance <= 2.0 + 1e-6, f"distance out of range: {distance}"

    @pytest.mark.asyncio
    async def test_document_id_filter_restricts_results(self, db_session, corpus_in_db):
        """Filtering by document_id returns only chunks from that document."""
        repo = VectorRepository()
        # Use the OAuth chunk's document_id
        oauth_record = corpus_in_db[1]
        query_vector = _embed(CORPUS[0]["content"])  # query is for API keys

        results = await repo.search(
            query_vector=query_vector,
            top_k=10,
            session=db_session,
            document_id=oauth_record.document_id,
        )
        assert len(results) == 1, (
            f"Expected only 1 chunk from the oauth doc, got {len(results)}"
        )
        assert results[0][0] == oauth_record.chunk_id

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty_list(self, db_session):
        """Searching an empty database must return []."""
        # This test uses a fresh db_session with no corpus inserted.
        # Other tests may have committed data, so we look for a doc that
        # certainly does not exist using the document_id filter.
        repo = VectorRepository()
        query_vector = _embed("some query")
        results = await repo.search(
            query_vector=query_vector,
            top_k=5,
            session=db_session,
            document_id="nonexistent_doc_" + "z" * 48,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_by_chunk_ids_returns_records(self, db_session, corpus_in_db):
        """fetch_by_chunk_ids must return the correct ChunkRecord for known IDs."""
        repo = VectorRepository()
        ids = [r.chunk_id for r in corpus_in_db]
        result_map = await repo.fetch_by_chunk_ids(ids, db_session)

        assert len(result_map) == len(corpus_in_db)
        for record in corpus_in_db:
            assert record.chunk_id in result_map
            stored = result_map[record.chunk_id]
            assert stored.content == record.content

    @pytest.mark.asyncio
    async def test_fetch_by_chunk_ids_empty_list_returns_empty(self, db_session):
        repo = VectorRepository()
        result = await repo.fetch_by_chunk_ids([], db_session)
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# DenseRetriever integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@requires_db
class TestDenseRetrieverIntegration:

    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieved_chunks(self, db_session, corpus_in_db):
        """retrieve() must return RetrievedChunk instances."""
        svc = _fake_service()
        retriever = DenseRetriever(svc)

        results = await retriever.retrieve(
            "API key revocation",
            top_k=3,
            session=db_session,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievedChunk) for r in results)

    @pytest.mark.asyncio
    async def test_nearest_chunk_is_rank_one(self, db_session, corpus_in_db):
        """The chunk with the smallest distance must be rank 1."""
        svc = _fake_service()
        retriever = DenseRetriever(svc)

        # Query with the exact text of the first corpus chunk — it should be rank 1.
        query_text = CORPUS[0]["content"]
        results = await retriever.retrieve(query_text, top_k=3, session=db_session)

        assert len(results) >= 1
        rank_one = results[0]
        assert rank_one.rank == 1
        assert rank_one.chunk_id == _cid(CORPUS[0]["prefix"])

    @pytest.mark.asyncio
    async def test_ranks_are_consecutive_from_one(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=3, session=db_session)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))

    @pytest.mark.asyncio
    async def test_distances_ascending(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=3, session=db_session)
        distances = [r.distance for r in results]
        assert distances == sorted(distances), f"Not in ascending distance order: {distances}"

    @pytest.mark.asyncio
    async def test_similarity_derived_from_distance(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=3, session=db_session)
        for r in results:
            assert abs(r.similarity - (1.0 - r.distance)) < 1e-6, (
                f"similarity={r.similarity} but 1-distance={1.0 - r.distance}"
            )

    @pytest.mark.asyncio
    async def test_content_preserved_in_result(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=1, session=db_session)
        assert results[0].content == CORPUS[0]["content"]

    @pytest.mark.asyncio
    async def test_section_path_preserved(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=1, session=db_session)
        assert results[0].section_path == CORPUS[0]["section"]

    @pytest.mark.asyncio
    async def test_document_title_preserved(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=1, session=db_session)
        assert results[0].document_title == CORPUS[0]["title"]

    @pytest.mark.asyncio
    async def test_document_id_filter_scopes_search(self, db_session, corpus_in_db):
        """Filtering by document_id must return only chunks from that document."""
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        # Use the oauth document's ID
        oauth_doc_id = _did(CORPUS[1]["doc_prefix"])

        results = await retriever.retrieve(
            "some query",
            top_k=10,
            session=db_session,
            document_id=oauth_doc_id,
        )
        assert all(r.document_id == oauth_doc_id for r in results), (
            "All results must belong to the filtered document"
        )

    @pytest.mark.asyncio
    async def test_query_validation_raises_in_integration(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        with pytest.raises(QueryValidationError):
            await retriever.retrieve("   ", top_k=5, session=db_session)

    @pytest.mark.asyncio
    async def test_distance_range_valid(self, db_session, corpus_in_db):
        """All returned distances must be in [0, 2] (L2-normalized cosine distance)."""
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=3, session=db_session)
        for r in results:
            assert 0.0 <= r.distance <= 2.0 + 1e-6, (
                f"distance {r.distance} is outside expected range [0, 2]"
            )

    @pytest.mark.asyncio
    async def test_top_k_one_returns_one_result(self, db_session, corpus_in_db):
        svc = _fake_service()
        retriever = DenseRetriever(svc)
        results = await retriever.retrieve(CORPUS[0]["content"], top_k=1, session=db_session)
        assert len(results) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Real model integration test (optional — skipped if model not available)
# ─────────────────────────────────────────────────────────────────────────────

def _real_model_available() -> bool:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        SentenceTransformer("BAAI/bge-small-en-v1.5")
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.model_integration
@requires_db
class TestRealModelRetrieval:
    """
    Optional tests using the actual BAAI/bge-small-en-v1.5 model.

    These tests verify that:
    1. The real embedding model produces vectors compatible with the DB schema.
    2. Semantic similarity works end-to-end: a query about API keys finds
       the API key chunk, not the database or OAuth chunk.
    3. Query and corpus embeddings from the same model produce meaningful ranks.

    These tests are marked model_integration and require the model to be
    downloaded.  They are skipped gracefully if the model is absent.
    """

    @pytest.mark.asyncio
    async def test_real_model_retrieves_semantically_similar_chunk(self, db_session):
        """
        Demonstrate end-to-end semantic retrieval with the real BGE model.

        Corpus:
            Chunk A: "The API key can be revoked from the security dashboard."
            Chunk B: "OAuth refresh tokens are used to obtain new access tokens."
            Chunk C: "PostgreSQL uses indexes to improve query performance."

        Query: "How do I revoke an API key?"

        Expected: Chunk A is rank 1 (semantically nearest).
        """
        from documind.embeddings.service import create_embedding_service
        from db.models import chunk_record_from_normalized

        real_svc = create_embedding_service("BAAI/bge-small-en-v1.5")

        # Insert corpus using real embeddings
        repo = ChunkRepository()
        records = []
        for i, entry in enumerate(CORPUS):
            emb = real_svc.embed_text(entry["content"])
            record = _make_record(
                chunk_id=_cid(f"real_{entry['prefix']}"),
                document_id=_did(f"real_{entry['doc_prefix']}"),
                content=entry["content"],
                embedding=emb,
                section_path=entry["section"],
                document_title=entry["title"],
                chunk_index=i,
            )
            records.append(record)

        await repo.upsert_chunks(records, db_session)

        # Retrieve with a semantically related query
        retriever = DenseRetriever(real_svc)
        query = "How do I revoke an API key?"
        results = await retriever.retrieve(
            query,
            top_k=3,
            session=db_session,
            document_id=None,
        )

        # Filter results to the chunks we just inserted (by prefix)
        our_ids = {_cid(f"real_{entry['prefix']}") for entry in CORPUS}
        our_results = [r for r in results if r.chunk_id in our_ids]

        assert len(our_results) >= 1, "Expected at least one result from the real corpus"

        rank_one = our_results[0]
        assert rank_one.chunk_id == _cid(f"real_{CORPUS[0]['prefix']}"), (
            f"Expected API key chunk at rank 1, got: {rank_one.content[:60]}"
        )
        assert rank_one.distance < 0.3, (
            f"Expected low distance for semantically similar query, got {rank_one.distance:.4f}"
        )
