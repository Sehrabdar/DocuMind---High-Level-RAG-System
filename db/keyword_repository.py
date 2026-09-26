"""
KeywordRepository — PostgreSQL full-text search for Phase 5.

Responsibilities
----------------
- Build and execute PostgreSQL FTS queries against the ``chunks`` table.
- Rank matching rows using ``ts_rank_cd`` (cover density ranking).
- Return typed results for the KeywordRetriever to assemble into RetrievedChunk objects.

What this class does NOT do:
- Parse or preprocess queries beyond passing them to ``websearch_to_tsquery``.
- Validate query strings (that is KeywordRetriever's responsibility).
- Validate top_k (that is KeywordRetriever's responsibility).
- Generate embeddings (not relevant for keyword search).
- Manage sessions (the caller passes sessions in).

Query design
------------
The FTS query uses PostgreSQL's native operators:

    SELECT <columns>, ts_rank_cd(search_vector, query) AS fts_score
    FROM chunks, websearch_to_tsquery('english', :query_text) AS query
    WHERE search_vector @@ query
    ORDER BY fts_score DESC
    LIMIT :top_k

Key design decisions:

1. ``websearch_to_tsquery`` over ``plainto_tsquery``/``to_tsquery``
   ``websearch_to_tsquery`` is the most robust choice for user-supplied queries:
   - Handles quoted phrases:  "connection pooling"
   - Handles negation:        PostgreSQL -Oracle
   - Handles OR:              JWT OR OAuth
   - Does NOT raise errors on punctuation or stop-word-only input
   - Returns an empty query on no-match input (gracefully returns 0 rows)

2. ``ts_rank_cd`` over ``ts_rank``
   ts_rank_cd (cover density) rewards chunks where matching terms appear
   clustered together, which is a better relevance signal for technical
   documentation than raw term frequency (ts_rank).

3. FROM … AS query pattern
   Binding the query expression once via ``FROM … AS query`` avoids
   calling websearch_to_tsquery twice (in WHERE and in ts_rank_cd).
   This is a PostgreSQL idiom for efficient FTS queries.

4. ``search_vector`` column
   The tsvector is a GENERATED ALWAYS AS STORED column (added in migration
   0002).  PostgreSQL maintains it automatically on every INSERT/UPDATE.
   The GIN index on this column makes the @@ operator O(log N).

Score semantics
---------------
``fts_score`` is the raw ``ts_rank_cd()`` output:
- Range: [0, ∞) — not normalised, not bounded at 1.0
- Higher is better.
- Absolute values have no fixed semantic meaning.
- NOT comparable to cosine distance or cosine similarity.
- Results are ordered by ``fts_score DESC`` (best match first → rank 1).

Stop words
----------
PostgreSQL's 'english' text search configuration removes common English stop
words (a, the, is, etc.) before indexing.  A query consisting entirely of
stop words (e.g. "the a") will be converted to an empty tsquery and match
zero rows.  This is correct behaviour; the KeywordRetriever handles it
gracefully (returns an empty list).
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class KeywordRepository:
    """PostgreSQL FTS search layer for keyword retrieval.

    All methods are ``async`` and accept an ``AsyncSession`` from the caller.
    """

    async def search(
        self,
        query_text: str,
        top_k: int,
        session: AsyncSession,
        *,
        document_id: str | None = None,
    ) -> list[dict]:
        """Search for chunks matching query_text using PostgreSQL FTS.

        Returns a list of row-dicts in descending fts_score order (best first).
        Each dict contains all columns needed to construct a RetrievedChunk.

        Parameters
        ----------
        query_text:
            Raw user query.  Passed directly to ``websearch_to_tsquery``.
            Stop words are filtered by PostgreSQL.  Punctuation is handled.
            An all-stop-word or empty query returns zero results gracefully.
        top_k:
            Maximum number of results.  Must be > 0.
        session:
            Active ``AsyncSession``.
        document_id:
            Optional filter.  If supplied, only chunks from this document
            are searched.

        Returns
        -------
        list[dict]
            Dicts with keys matching ChunkRecord columns plus ``fts_score``.
            Empty list if no chunks match.
        """
        where_clause = ""
        params: dict = {
            "query_text": query_text,
            "top_k": top_k,
        }

        if document_id is not None:
            where_clause = "AND chunks.document_id = :document_id"
            params["document_id"] = document_id

        # The FROM … AS query pattern binds websearch_to_tsquery once and
        # reuses it in both WHERE (@@) and SELECT (ts_rank_cd).
        # If websearch_to_tsquery produces an empty tsquery (e.g. all stop
        # words), the @@ operator returns false for every row → 0 results.
        sql = text(
            f"""
            SELECT
                chunks.chunk_id,
                chunks.document_id,
                chunks.content,
                chunks.token_count,
                chunks.chunk_index,
                chunks.section_path,
                chunks.document_title,
                chunks.page,
                chunks.page_start,
                chunks.page_end,
                chunks.start_char,
                chunks.end_char,
                ts_rank_cd(chunks.search_vector, query) AS fts_score
            FROM
                chunks,
                websearch_to_tsquery('english', :query_text) AS query
            WHERE
                chunks.search_vector @@ query
                {where_clause}
            ORDER BY fts_score DESC
            LIMIT :top_k
            """
        )

        result = await session.execute(sql, params)
        rows = result.mappings().fetchall()

        logger.debug(
            "KeywordRepository.search: query=%r top_k=%d doc_filter=%r → %d rows",
            query_text[:60],
            top_k,
            document_id,
            len(rows),
        )

        return [dict(row) for row in rows]
