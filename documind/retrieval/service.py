"""
DenseRetriever — Phase 4 dense retrieval service.

Architecture
------------
The retrieval pipeline is:

    DenseRetriever.retrieve(query, top_k)
        │
        ├── 1. Validate query string
        ├── 2. EmbeddingService.embed_text(query) → 384-d L2-normalized vector
        ├── 3. VectorRepository.search(vector, top_k) → [(chunk_id, distance)]
        ├── 4. VectorRepository.fetch_by_chunk_ids(ids) → ChunkRecord map
        ├── 5. Sort by distance ASC (database already returns sorted, but we
        │      enforce order here to make the contract explicit)
        ├── 6. Assign ranks (1-indexed, rank 1 = closest)
        └── 7. Return list[RetrievedChunk]

Boundary contract
-----------------
- DenseRetriever does NOT know about SentenceTransformer.
  It depends on EmbeddingService (which abstracts the provider).
- DenseRetriever does NOT know about asyncpg or SQL.
  It depends on VectorRepository.
- DenseRetriever does NOT generate answers or call LLMs.

This makes it independently testable with FakeEmbeddingProvider +
an in-process mock or real integration DB.

Score semantics
---------------
All scores use cosine distance from pgvector's <=> operator:
    distance ∈ [0, 2] for L2-normalized vectors
    0.0 = identical direction → best match
    rank 1 = smallest distance

similarity = 1 - distance is also computed and stored on each result.
See documind/retrieval/models.py for full score documentation.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from db.vector_repository import VectorRepository
from documind.embeddings.service import EmbeddingService
from documind.retrieval.models import RetrievedChunk
from documind.retrieval.validation import (
    QueryValidationError,
    validate_query,
    validate_top_k,
)

# Re-export: callers that do `from documind.retrieval.service import QueryValidationError`
# continue to work. The canonical definition is in documind.retrieval.validation.
__all__ = ["DenseRetriever", "QueryValidationError"]

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Semantic chunk retrieval using pgvector cosine-distance search.

    Parameters
    ----------
    embedding_service:
        Pre-constructed EmbeddingService.  Use FakeEmbeddingProvider for
        tests; use create_embedding_service() for production.
    vector_repository:
        VectorRepository instance.  Defaults to a new VectorRepository().
        Inject a different implementation for testing.
    max_top_k:
        Hard ceiling on top_k.  Defaults to settings.retrieval_max_top_k.
        Override in tests to test boundary behaviour.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_repository: VectorRepository | None = None,
        *,
        max_top_k: int | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_repo = vector_repository or VectorRepository()
        self._max_top_k = max_top_k if max_top_k is not None else settings.retrieval_max_top_k

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        session: AsyncSession | None = None,
        *,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top_k most semantically relevant chunks for query.

        Parameters
        ----------
        query:
            Natural-language query string.  Must be non-empty and non-whitespace.
        top_k:
            Number of chunks to return.  Defaults to settings.retrieval_default_top_k.
            Must satisfy 1 ≤ top_k ≤ max_top_k.
        session:
            Active AsyncSession.  The caller manages the session lifecycle.
            If None, a new session is created from the default engine.
        document_id:
            Optional filter: only search chunks from this document.
            Useful for focused document Q&A; ignored if None.

        Returns
        -------
        list[RetrievedChunk]
            Ordered by ascending cosine distance (rank 1 = closest).
            Empty list if no chunks exist in the database.

        Raises
        ------
        QueryValidationError
            If query is empty, whitespace-only, or None.
        ValueError
            If top_k is out of the valid range.
        """
        # 1. Validate query
        resolved_query = self._validate_query(query)

        # 2. Validate top_k
        resolved_top_k = self._validate_top_k(top_k)

        logger.debug(
            "DenseRetriever.retrieve: query=%r top_k=%d doc_filter=%r",
            resolved_query[:60],
            resolved_top_k,
            document_id,
        )

        # 3–7. Execute retrieval
        if session is not None:
            return await self._retrieve_with_session(
                resolved_query, resolved_top_k, session, document_id=document_id
            )
        else:
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
        """Execute the full retrieval pipeline using a caller-supplied session."""
        # 3. Embed the query — uses the same model/normalization as document chunks
        query_vector = self._embedding_service.embed_text(query)

        # 4. Nearest-neighbour search → [(chunk_id, distance)]
        search_results = await self._vector_repo.search(
            query_vector=query_vector,
            top_k=top_k,
            session=session,
            document_id=document_id,
        )

        if not search_results:
            logger.debug("DenseRetriever: no chunks found in database")
            return []

        # 5. Fetch full chunk data for the returned IDs
        chunk_ids = [cid for cid, _ in search_results]
        chunk_map = await self._vector_repo.fetch_by_chunk_ids(chunk_ids, session)

        # 6. Assemble results, preserving distance order from the search
        results: list[RetrievedChunk] = []
        for rank, (chunk_id, distance) in enumerate(search_results, start=1):
            record = chunk_map.get(chunk_id)
            if record is None:
                # Should not happen in a consistent database, but be defensive
                logger.warning(
                    "DenseRetriever: chunk_id %r returned by search but not found "
                    "in fetch_by_chunk_ids — skipping",
                    chunk_id,
                )
                continue

            results.append(
                RetrievedChunk.from_row(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    content=record.content,
                    token_count=record.token_count,
                    chunk_index=record.chunk_index,
                    section_path=list(record.section_path),
                    document_title=record.document_title,
                    page=record.page,
                    page_start=record.page_start,
                    page_end=record.page_end,
                    start_char=record.start_char,
                    end_char=record.end_char,
                    distance=distance,
                    rank=rank,
                )
            )

        logger.debug("DenseRetriever: returning %d results", len(results))
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

    # ── Validation helpers — delegated to shared module ──────────────────────

    def _validate_query(self, query: str) -> str:
        """Delegate to shared validation module."""
        return validate_query(query)

    def _validate_top_k(self, top_k: int | None) -> int:
        """Delegate to shared validation module."""
        return validate_top_k(top_k, self._max_top_k)
