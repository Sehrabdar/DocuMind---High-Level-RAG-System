"""
VectorRepository — pgvector nearest-neighbour search for Phase 4.

Responsibilities
----------------
- Execute cosine-distance nearest-neighbour queries against the ``chunks`` table.
- Return typed results that include both the chunk data and the distance score.
- Support an optional ``document_id`` filter for scoped retrieval.

What this class does NOT do:
- Generate embeddings (that is EmbeddingService's responsibility).
- Validate queries (that is DenseRetriever's responsibility).
- Assign ranks (that is DenseRetriever's responsibility after sorting).
- Manage sessions (the caller passes sessions in).

pgvector query design
---------------------
The nearest-neighbour query is:

    SELECT <columns>, embedding <=> :query_vector AS distance
    FROM chunks
    [WHERE document_id = :doc_id]
    ORDER BY distance ASC
    LIMIT :top_k

The ``<=>`` operator is pgvector's cosine-distance operator.  It returns
values in [0, 2] for L2-normalized unit vectors:
    0.0  = identical direction (perfect match)
    1.0  = orthogonal (unrelated)
    2.0  = opposite direction

We use ``text()`` for the cosine-distance expression because SQLAlchemy does
not natively generate pgvector operator syntax.  This is a deliberate,
minimal use of raw SQL — the rest of the query uses the ORM column references.

HNSW index usage
----------------
pgvector will use the HNSW index created in migration 0001 when:
1. The query vector has the same dimension as the index (384).
2. The operator class matches (vector_cosine_ops for <=>).
3. top_k is small (the HNSW index is approximate; it degrades at very large K).

The index is NOT used when:
- A WHERE clause is added that forces a sequential scan on a small table.
- The table is very small (the planner may prefer a sequential scan).
Run ``EXPLAIN ANALYZE`` to verify index usage in production.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class VectorRepository:
    """Thin database layer for nearest-neighbour vector search.

    All methods are ``async`` and accept an ``AsyncSession`` from the caller.
    This makes them independently testable with a session from an integration
    test database.
    """

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        session: AsyncSession,
        *,
        document_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find the top_k chunks closest to query_vector by cosine distance.

        Returns a list of (chunk_id, distance) tuples in ascending distance
        order (closest first).  The caller (DenseRetriever) is responsible for
        fetching full chunk data and assembling RetrievedChunk objects.

        We separate the search step (returns IDs + distances) from the data
        hydration step (fetches full rows) to keep this method testable in
        isolation without constructing full ChunkRecord objects.

        Parameters
        ----------
        query_vector:
            L2-normalized embedding of the query (length must match stored dim).
        top_k:
            Maximum number of results to return.  Must be > 0.
        session:
            Active ``AsyncSession``.
        document_id:
            Optional filter.  If supplied, only chunks from this document are
            searched.  Useful for document-scoped Q&A.

        Returns
        -------
        list[tuple[str, float]]
            List of (chunk_id, cosine_distance) in ascending distance order.
            Empty list if no chunks exist in the database.
        """
        # Build the cosine-distance expression using pgvector's <=> operator.
        # The CAST to vector is required so asyncpg knows the parameter type.
        # ::vector is PostgreSQL cast syntax; sqlalchemy text() passes it through.
        where_clause = ""
        params: dict = {
            "query_vector": str(query_vector),
            "top_k": top_k,
        }
        if document_id is not None:
            where_clause = "WHERE document_id = :document_id"
            params["document_id"] = document_id

        sql = text(
            f"""
            SELECT
                chunk_id,
                (embedding <=> CAST(:query_vector AS vector)) AS distance
            FROM chunks
            {where_clause}
            ORDER BY distance ASC
            LIMIT :top_k
            """
        )

        result = await session.execute(sql, params)
        rows = result.fetchall()

        logger.debug(
            "VectorRepository.search: top_k=%d, doc_filter=%r → %d rows",
            top_k,
            document_id,
            len(rows),
        )
        return [(row.chunk_id, float(row.distance)) for row in rows]

    async def fetch_by_chunk_ids(
        self,
        chunk_ids: list[str],
        session: AsyncSession,
    ) -> dict[str, "ChunkRowData"]:
        """Fetch full chunk rows for a list of chunk_ids.

        Returns a dict keyed by chunk_id for O(1) lookup.  Order is not
        guaranteed — the caller is responsible for re-ordering by the
        distance ranking from search().

        Parameters
        ----------
        chunk_ids:
            List of chunk_id values to look up.
        session:
            Active ``AsyncSession``.

        Returns
        -------
        dict[str, ChunkRowData]
            Maps chunk_id → a dataclass with all chunk columns.
            Missing chunk_ids are silently absent (should not happen in practice).
        """
        if not chunk_ids:
            return {}

        from sqlalchemy import select
        from db.models import ChunkRecord

        stmt = select(ChunkRecord).where(ChunkRecord.chunk_id.in_(chunk_ids))
        result = await session.execute(stmt)
        records = result.scalars().all()

        return {r.chunk_id: r for r in records}


# Type alias for the return type of fetch_by_chunk_ids — helps mypy
# understand that we return ChunkRecord instances without a circular import.
ChunkRowData = "ChunkRecord"  # noqa: F821
