"""
KeywordRetriever — Phase 5 PostgreSQL FTS keyword retrieval service.

Architecture
------------
The retrieval pipeline is:

    KeywordRetriever.retrieve(query, top_k, session)
        │
        ├── 1. Validate query string (shared validation module)
        ├── 2. Validate top_k (shared validation module)
        ├── 3. KeywordRepository.search(query, top_k, session)
        │       └── PostgreSQL websearch_to_tsquery + ts_rank_cd
        │           → list[dict]  (raw DB rows with fts_score)
        ├── 4. Assign ranks (1-indexed, rank 1 = highest fts_score)
        └── 5. Return list[RetrievedChunk]

Boundary contract
-----------------
- KeywordRetriever does NOT generate embeddings.
- KeywordRetriever does NOT call EmbeddingService.
- KeywordRetriever does NOT perform any Python-side text matching or ranking.
- KeywordRetriever does NOT know about pgvector or cosine distance.
- All keyword search and ranking runs inside PostgreSQL.

This makes it independently testable with a mocked KeywordRepository and
independently operable — no model download required.

Score semantics
---------------
All scores use PostgreSQL ts_rank_cd():
    fts_score ∈ [0, ∞)  higher = better keyword match
    rank 1 = highest fts_score (best match)

This is fundamentally different from dense retrieval:
    Dense:   lower cosine distance = better
    Keyword: higher fts_score = better

The two scores cannot be compared or merged directly.
Score fusion is the responsibility of Phase 6 (hybrid retrieval + RRF).

Phase independence
------------------
This service is completely independent from DenseRetriever and EmbeddingService.
Both retrievers expose the same retrieve() interface so Phase 6 can call them
symmetrically without conditional logic.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.keyword_repository import KeywordRepository
from documind.retrieval.models import RetrievedChunk
from documind.retrieval.validation import validate_query, validate_top_k

logger = logging.getLogger(__name__)


class KeywordRetriever:
    """PostgreSQL FTS keyword retrieval service.

    Parameters
    ----------
    keyword_repository:
        KeywordRepository instance.  Defaults to a new KeywordRepository().
        Inject a different implementation for testing.
    max_top_k:
        Hard ceiling on top_k.  Defaults to settings.retrieval_max_top_k.
        Override in tests to test boundary behaviour.
    """

    def __init__(
        self,
        keyword_repository: KeywordRepository | None = None,
        *,
        max_top_k: int | None = None,
    ) -> None:
        self._repo = keyword_repository or KeywordRepository()
        self._max_top_k = (
            max_top_k if max_top_k is not None else settings.retrieval_max_top_k
        )

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        session: AsyncSession | None = None,
        *,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top_k most keyword-relevant chunks for query.

        Uses PostgreSQL full-text search (websearch_to_tsquery + ts_rank_cd).
        No embeddings are generated or required.

        Parameters
        ----------
        query:
            Natural-language or technical query string.
            Must be non-empty and non-whitespace.
            Passed directly to ``websearch_to_tsquery('english', query)``.
            Stop words are filtered by PostgreSQL (e.g. "the a" → 0 results).
        top_k:
            Number of chunks to return.  Defaults to settings.retrieval_default_top_k.
            Must satisfy 1 ≤ top_k ≤ max_top_k.
        session:
            Active AsyncSession.  The caller manages the session lifecycle.
            If None, a new session is created from the default engine.
        document_id:
            Optional filter: only search chunks from this document.

        Returns
        -------
        list[RetrievedChunk]
            Ordered by descending fts_score (rank 1 = best keyword match).
            ``retrieval_method == "keyword"`` for all results.
            ``distance`` and ``similarity`` are None (not applicable).
            ``fts_score`` is the raw ts_rank_cd() value (higher = better).
            Empty list if no chunks match or the database is empty.

        Raises
        ------
        QueryValidationError
            If query is empty or whitespace-only.
        ValueError
            If top_k is out of the valid range.
        """
        resolved_query = validate_query(query)
        resolved_top_k = validate_top_k(top_k, self._max_top_k)

        logger.debug(
            "KeywordRetriever.retrieve: query=%r top_k=%d doc_filter=%r",
            resolved_query[:60],
            resolved_top_k,
            document_id,
        )

        if session is not None:
            return await self._retrieve_with_session(
                resolved_query, resolved_top_k, session, document_id=document_id
            )
        return await self._retrieve_new_session(
            resolved_query, resolved_top_k, document_id=document_id
        )

    async def _retrieve_with_session(
        self,
        query: str,
        top_k: int,
        session: AsyncSession,
        *,
        document_id: str | None,
    ) -> list[RetrievedChunk]:
        """Execute FTS retrieval using a caller-supplied session."""
        rows = await self._repo.search(
            query_text=query,
            top_k=top_k,
            session=session,
            document_id=document_id,
        )

        if not rows:
            logger.debug("KeywordRetriever: no matching chunks")
            return []

        results: list[RetrievedChunk] = []
        for rank, row in enumerate(rows, start=1):
            results.append(
                RetrievedChunk.from_fts_row(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    content=row["content"],
                    token_count=row["token_count"],
                    chunk_index=row["chunk_index"],
                    section_path=list(row["section_path"]),
                    document_title=row["document_title"],
                    page=row["page"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                    fts_score=float(row["fts_score"]),
                    rank=rank,
                )
            )

        logger.debug("KeywordRetriever: returning %d results", len(results))
        return results

    async def _retrieve_new_session(
        self,
        query: str,
        top_k: int,
        *,
        document_id: str | None,
    ) -> list[RetrievedChunk]:
        """Create a session from the default engine and execute retrieval."""
        from db.session import get_async_session, get_engine, get_session_factory

        engine = get_engine()
        factory = get_session_factory(engine)
        async with get_async_session(factory) as session:
            return await self._retrieve_with_session(
                query, top_k, session, document_id=document_id
            )
