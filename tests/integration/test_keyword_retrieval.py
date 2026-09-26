"""
Integration tests for Phase 5 — PostgreSQL FTS keyword retrieval.

Requirements:
    - PostgreSQL + pgvector running on port 5434 (docker compose up -d)
    - Both migrations applied (uv run alembic upgrade head)

Isolation strategy
------------------
Each test run inserts 4 synthetic chunks with content prefixed by a unique
``RUNID_<hex>`` token.  All FTS queries include this token so results are
scoped to exactly the 4 chunks from the current invocation, even when the
database contains stale rows from previous test runs.

The fixture deletes its rows in teardown (before the session commits), so
each test leaves zero net rows in the database.

The tests are automatically skipped if the database is not reachable.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import text

from db.keyword_repository import KeywordRepository
from documind.retrieval.keyword import KeywordRetriever
from tests.integration.conftest import requires_db

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic corpus — 4 chunks with clearly distinguishable technical topics
# ─────────────────────────────────────────────────────────────────────────────

_CORPUS = [
    {
        "tag": "postgres",
        "content": (
            "PostgreSQL connection pooling uses async database sessions "
            "to manage concurrent queries efficiently. "
            "PgBouncer and SQLAlchemy async engines are common pooling strategies."
        ),
    },
    {
        "tag": "redis",
        "content": (
            "Redis sliding window rate limiting tracks request timestamps "
            "to enforce per-user API quotas. "
            "Tokens bucket and leaky bucket algorithms are popular alternatives."
        ),
    },
    {
        "tag": "jwt",
        "content": (
            "JWT authentication middleware validates bearer tokens on every request. "
            "The payload is base64-encoded and contains claims such as subject and expiry."
        ),
    },
    {
        "tag": "docker",
        "content": (
            "Docker Compose starts container services with a single command. "
            "Networking between containers uses the default bridge network."
        ),
    },
]


@pytest_asyncio.fixture
async def fts_corpus(db_session):
    """
    Insert 4 synthetic chunks and yield a corpus descriptor dict.

    Yields
    ------
    dict with keys:
        chunk_ids  : dict[tag, chunk_id]  — chunk IDs inserted by this run
        doc_ids    : dict[tag, doc_id]    — document IDs per tag
        run_token  : str                  — unique token embedded in all contents

    Isolation strategy
    ------------------
    Each chunk's content is prefixed with ``run_token`` (e.g. ``RUNID_a1b2c3d4``).
    All FTS queries in tests include this token so results are scoped to
    exactly the 4 chunks from this run, even when the database has stale rows
    from previous runs.

    Teardown deletes the 4 inserted rows before the session commits, so
    INSERT + DELETE net to zero persisted rows after each test.
    """
    run_id = uuid.uuid4().hex[:8]
    run_token = f"RUNID{run_id}"  # unique FTS-queryable token (no underscore = single lexeme)
    tag_to_chunk_id: dict[str, str] = {}
    tag_to_doc_id: dict[str, str] = {}

    # Fake 384-d zero vector; FTS does not use embeddings
    fake_vector = "[" + ",".join(["0"] * 384) + "]"

    for i, item in enumerate(_CORPUS):
        chunk_id = f"{run_id}-{item['tag']}-{'x' * (60 - len(run_id) - len(item['tag']))}"
        chunk_id = chunk_id[:64]
        doc_id = f"{run_id}-doc-{item['tag']}"
        tag_to_chunk_id[item["tag"]] = chunk_id
        tag_to_doc_id[item["tag"]] = doc_id

        # Prefix content with run_token for query isolation.
        # asyncpg cannot parse ::type casts immediately after bind params, so
        # we use CAST(:param AS type) instead.
        scoped_content = f"{run_token} {item['content']}"

        await db_session.execute(
            text(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, content, token_count,
                    embedding, section_path, document_title,
                    page, page_start, page_end, start_char, end_char, created_at
                )
                VALUES (
                    :chunk_id, :document_id, :chunk_index, :content, :token_count,
                    CAST(:embedding AS vector),
                    CAST(:section_path AS jsonb),
                    :document_title,
                    NULL, NULL, NULL, 0, :end_char, NOW()
                )
                ON CONFLICT (chunk_id) DO NOTHING
                """
            ),
            {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": i,
                "content": scoped_content,
                "token_count": len(scoped_content.split()),
                "embedding": fake_vector,
                "section_path": "[]",
                "document_title": f"{item['tag'].capitalize()} Guide",
                "end_char": len(scoped_content),
            },
        )

    await db_session.flush()
    yield {
        "chunk_ids": tag_to_chunk_id,
        "doc_ids": tag_to_doc_id,
        "run_token": run_token,
    }

    # Teardown: delete inserted rows before the session commits.
    # INSERT + DELETE net to zero rows persisted after each test.
    for chunk_id in tag_to_chunk_id.values():
        await db_session.execute(
            text("DELETE FROM chunks WHERE chunk_id = :cid"),
            {"cid": chunk_id},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Schema — verify migration 0002 was applied
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSchemaAfterMigration:

    @requires_db
    @pytest.mark.asyncio
    async def test_search_vector_column_exists(self, db_session):
        """search_vector column must exist after migration 0002."""
        result = await db_session.execute(
            text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'chunks'
                  AND column_name = 'search_vector'
                """
            )
        )
        rows = result.fetchall()
        assert rows, (
            "search_vector column not found in chunks table. "
            "Run: uv run alembic upgrade head"
        )

    @requires_db
    @pytest.mark.asyncio
    async def test_gin_index_exists(self, db_session):
        """GIN index ix_chunks_search_vector_gin must exist after migration 0002."""
        result = await db_session.execute(
            text(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = 'chunks'
                  AND indexname = 'ix_chunks_search_vector_gin'
                """
            )
        )
        rows = result.fetchall()
        assert rows, (
            "GIN index ix_chunks_search_vector_gin not found. "
            "Run: uv run alembic upgrade head"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Basic retrieval
# All queries include `run_token` to scope to the current test's 4 chunks.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestKeywordRetrievalBasic:

    def _q(self, fts_corpus: dict, extra: str) -> str:
        """Build a run-scoped query: run_token AND extra terms."""
        return f"{fts_corpus['run_token']} {extra}"

    @requires_db
    @pytest.mark.asyncio
    async def test_postgres_query_returns_postgres_chunk(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=4,
            session=db_session,
        )
        assert results, "Expected at least one result for the PostgreSQL query"
        chunk_ids = [r.chunk_id for r in results]
        assert fts_corpus["chunk_ids"]["postgres"] in chunk_ids

    @requires_db
    @pytest.mark.asyncio
    async def test_postgres_chunk_is_rank_1(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "PostgreSQL connection pooling async sessions"),
            top_k=4,
            session=db_session,
        )
        assert results
        assert results[0].chunk_id == fts_corpus["chunk_ids"]["postgres"]

    @requires_db
    @pytest.mark.asyncio
    async def test_redis_query_returns_redis_chunk(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "rate limiting sliding window"),
            top_k=4,
            session=db_session,
        )
        assert results
        chunk_ids = [r.chunk_id for r in results]
        assert fts_corpus["chunk_ids"]["redis"] in chunk_ids

    @requires_db
    @pytest.mark.asyncio
    async def test_redis_chunk_is_rank_1(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "Redis rate limiting sliding window API quotas"),
            top_k=4,
            session=db_session,
        )
        assert results
        assert results[0].chunk_id == fts_corpus["chunk_ids"]["redis"]

    @requires_db
    @pytest.mark.asyncio
    async def test_jwt_query_returns_jwt_chunk(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "JWT authentication bearer token"),
            top_k=4,
            session=db_session,
        )
        assert results
        chunk_ids = [r.chunk_id for r in results]
        assert fts_corpus["chunk_ids"]["jwt"] in chunk_ids

    @requires_db
    @pytest.mark.asyncio
    async def test_docker_query_returns_docker_chunk(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "Docker Compose container services"),
            top_k=4,
            session=db_session,
        )
        assert results
        chunk_ids = [r.chunk_id for r in results]
        assert fts_corpus["chunk_ids"]["docker"] in chunk_ids


# ─────────────────────────────────────────────────────────────────────────────
# Ranking and scoring
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestKeywordRanking:

    def _q(self, fts_corpus: dict, extra: str) -> str:
        return f"{fts_corpus['run_token']} {extra}"

    @requires_db
    @pytest.mark.asyncio
    async def test_results_ordered_by_descending_fts_score(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=4,
            session=db_session,
        )
        scores = [r.fts_score for r in results if r.fts_score is not None]
        assert scores == sorted(scores, reverse=True), (
            "Results must be ordered by descending fts_score"
        )

    @requires_db
    @pytest.mark.asyncio
    async def test_rank_is_one_indexed(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=4,
            session=db_session,
        )
        assert results
        assert results[0].rank == 1

    @requires_db
    @pytest.mark.asyncio
    async def test_rank_is_sequential(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        # run_token alone matches all 4 chunks; enough for sequential rank check
        results = await retriever.retrieve(
            fts_corpus["run_token"],
            top_k=4,
            session=db_session,
        )
        assert results
        for i, r in enumerate(results, start=1):
            assert r.rank == i

    @requires_db
    @pytest.mark.asyncio
    async def test_fts_scores_are_positive(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=4,
            session=db_session,
        )
        assert results
        assert all(r.fts_score > 0 for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Result model fields
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestKeywordResultFields:

    def _q(self, fts_corpus: dict, extra: str) -> str:
        return f"{fts_corpus['run_token']} {extra}"

    @requires_db
    @pytest.mark.asyncio
    async def test_retrieval_method_is_keyword(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=2,
            session=db_session,
        )
        assert results
        assert all(r.retrieval_method == "keyword" for r in results)

    @requires_db
    @pytest.mark.asyncio
    async def test_distance_is_none(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=2,
            session=db_session,
        )
        assert results
        assert all(r.distance is None for r in results)

    @requires_db
    @pytest.mark.asyncio
    async def test_similarity_is_none(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=2,
            session=db_session,
        )
        assert results
        assert all(r.similarity is None for r in results)

    @requires_db
    @pytest.mark.asyncio
    async def test_chunk_id_present(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=2,
            session=db_session,
        )
        assert results
        assert all(r.chunk_id for r in results)

    @requires_db
    @pytest.mark.asyncio
    async def test_content_present(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=2,
            session=db_session,
        )
        assert results
        assert all(r.content for r in results)

    @requires_db
    @pytest.mark.asyncio
    async def test_content_contains_run_token(self, db_session, fts_corpus):
        """All results contain the run_token prefix — confirms scoping is working."""
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=4,
            session=db_session,
        )
        assert results
        # run_token is embedded in content as the first word
        assert all(fts_corpus["run_token"] in r.content for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestKeywordEdgeCases:

    def _q(self, fts_corpus: dict, extra: str) -> str:
        return f"{fts_corpus['run_token']} {extra}"

    @requires_db
    @pytest.mark.asyncio
    async def test_no_match_query_returns_empty(self, db_session, fts_corpus):
        """A query with no matching terms returns an empty list."""
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            "quantum photonic compiler xyz123abc",
            top_k=5,
            session=db_session,
        )
        assert results == []

    @requires_db
    @pytest.mark.asyncio
    async def test_top_k_one_returns_one_result(self, db_session, fts_corpus):
        # Run-token alone matches all 4 chunks. With top_k=1, exactly 1 is returned.
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            fts_corpus["run_token"],
            top_k=1,
            session=db_session,
        )
        assert len(results) == 1

    @requires_db
    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, db_session, fts_corpus):
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            fts_corpus["run_token"],
            top_k=2,
            session=db_session,
        )
        assert len(results) <= 2

    @requires_db
    @pytest.mark.asyncio
    async def test_punctuation_in_query_does_not_crash(self, db_session, fts_corpus):
        """websearch_to_tsquery handles punctuation gracefully."""
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            f"{fts_corpus['run_token']}, connection pooling!",
            top_k=4,
            session=db_session,
        )
        # May or may not return results — but must not raise
        assert isinstance(results, list)

    @requires_db
    @pytest.mark.asyncio
    async def test_stop_words_only_query_returns_empty(self, db_session, fts_corpus):
        """A query of only English stop words becomes an empty tsquery → 0 results."""
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            "the a is",
            top_k=5,
            session=db_session,
        )
        assert results == []

    @requires_db
    @pytest.mark.asyncio
    async def test_quoted_phrase_query(self, db_session, fts_corpus):
        """websearch_to_tsquery supports quoted phrase matching."""
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            f'"{fts_corpus["run_token"]}" "connection pooling"',
            top_k=4,
            session=db_session,
        )
        assert isinstance(results, list)

    @requires_db
    @pytest.mark.asyncio
    async def test_multiple_terms_boost_relevant_chunk(self, db_session, fts_corpus):
        """The JWT chunk should rank highest for a JWT-specific query."""
        retriever = KeywordRetriever()
        results = await retriever.retrieve(
            self._q(fts_corpus, "JWT bearer token authentication middleware"),
            top_k=4,
            session=db_session,
        )
        assert results
        assert results[0].chunk_id == fts_corpus["chunk_ids"]["jwt"]

    @requires_db
    @pytest.mark.asyncio
    async def test_document_id_filter_restricts_results(self, db_session, fts_corpus):
        """document_id filter returns only chunks from that document."""
        retriever = KeywordRetriever()
        postgres_doc_id = fts_corpus["doc_ids"]["postgres"]

        # run_token + "connection pooling" matches the postgres chunk.
        results = await retriever.retrieve(
            self._q(fts_corpus, "connection pooling"),
            top_k=10,
            session=db_session,
            document_id=postgres_doc_id,
        )
        assert results, "Expected at least one result scoped to the postgres document"
        assert all(r.document_id == postgres_doc_id for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# KeywordRepository integration
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestKeywordRepositoryIntegration:

    def _q(self, fts_corpus: dict, extra: str) -> str:
        return f"{fts_corpus['run_token']} {extra}"

    @requires_db
    @pytest.mark.asyncio
    async def test_repo_returns_list_of_dicts(self, db_session, fts_corpus):
        repo = KeywordRepository()
        rows = await repo.search(
            self._q(fts_corpus, "connection pooling"),
            top_k=5,
            session=db_session,
        )
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    @requires_db
    @pytest.mark.asyncio
    async def test_repo_rows_have_fts_score(self, db_session, fts_corpus):
        repo = KeywordRepository()
        rows = await repo.search(
            self._q(fts_corpus, "connection pooling"),
            top_k=5,
            session=db_session,
        )
        assert rows
        assert all("fts_score" in r for r in rows)

    @requires_db
    @pytest.mark.asyncio
    async def test_repo_fts_score_decreasing(self, db_session, fts_corpus):
        repo = KeywordRepository()
        rows = await repo.search(
            self._q(fts_corpus, "connection pooling"),
            top_k=10,
            session=db_session,
        )
        scores = [r["fts_score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    @requires_db
    @pytest.mark.asyncio
    async def test_repo_no_match_returns_empty(self, db_session, fts_corpus):
        repo = KeywordRepository()
        rows = await repo.search("zzzquantumnonexistent", top_k=5, session=db_session)
        assert rows == []
