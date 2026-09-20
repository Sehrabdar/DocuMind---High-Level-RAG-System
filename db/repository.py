"""
ChunkRepository — persistence layer for Phase 3.

Responsibilities
----------------
- Insert / upsert chunk records into PostgreSQL.
- Fetch chunk records by document ID.
- Count total stored chunks.

What this class does NOT do:
- Generate embeddings (that is EmbeddingService's responsibility).
- Build SQL search queries (that is Phase 4's responsibility).
- Manage sessions (that is the caller's responsibility — sessions are passed in).

Idempotency
-----------
The upsert uses ``INSERT ... ON CONFLICT (chunk_id) DO UPDATE SET ...``.
Because ``chunk_id`` is deterministic (Phase 2 SHA-256 ID), re-ingesting
the same unchanged document will update the row in-place rather than
creating a duplicate.  This makes the ingestion pipeline idempotent.

Document update semantics
--------------------------
If a document changes and its chunks receive new content (and therefore
new SHA-256 ``chunk_id`` values), the new chunks are inserted as new rows.
Old chunk rows are NOT automatically deleted — this is a known Phase 3
limitation.  Full document-versioning (delete-and-reinsert on document
change) is deferred to a future phase.  This is safe for a technical
documentation corpus that evolves infrequently.
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import ChunkRecord

logger = logging.getLogger(__name__)


class ChunkRepository:
    """Thin persistence layer for ``ChunkRecord`` rows.

    All methods are ``async`` and accept an ``AsyncSession`` from the caller.
    This makes them individually testable with a session from a test transaction
    that is rolled back after the test.
    """

    async def upsert_chunks(
        self,
        records: list[ChunkRecord],
        session: AsyncSession,
    ) -> int:
        """Insert or update chunk records, returning the count of affected rows.

        Uses ``INSERT ... ON CONFLICT (chunk_id) DO UPDATE`` (PostgreSQL upsert).
        On conflict, the embedding and content are updated but ``created_at``
        is preserved (i.e. the first-insertion timestamp is kept).

        Parameters
        ----------
        records:
            List of ``ChunkRecord`` instances to persist.  May be empty.
        session:
            Active ``AsyncSession``.  The caller manages commit/rollback.

        Returns
        -------
        int
            Number of rows inserted or updated.
        """
        if not records:
            return 0

        # Build INSERT statements with ON CONFLICT using PostgreSQL dialect
        values = [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "chunk_index": r.chunk_index,
                "content": r.content,
                "token_count": r.token_count,
                "embedding": r.embedding,
                "section_path": r.section_path,
                "document_title": r.document_title,
                "page": r.page,
                "page_start": r.page_start,
                "page_end": r.page_end,
                "start_char": r.start_char,
                "end_char": r.end_char,
                "created_at": r.created_at,
            }
            for r in records
        ]

        stmt = pg_insert(ChunkRecord).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["chunk_id"],
            set_={
                # Update mutable fields on re-ingestion
                "content": stmt.excluded.content,
                "token_count": stmt.excluded.token_count,
                "embedding": stmt.excluded.embedding,
                "section_path": stmt.excluded.section_path,
                "document_title": stmt.excluded.document_title,
                "page": stmt.excluded.page,
                "page_start": stmt.excluded.page_start,
                "page_end": stmt.excluded.page_end,
                "start_char": stmt.excluded.start_char,
                "end_char": stmt.excluded.end_char,
                # created_at deliberately NOT updated — preserves first-insertion time
            },
        )

        result = await session.execute(stmt)
        affected = result.rowcount
        logger.debug("upsert_chunks: %d records affected", affected)
        return affected

    async def get_by_document_id(
        self,
        document_id: str,
        session: AsyncSession,
    ) -> list[ChunkRecord]:
        """Fetch all chunk records for a given document ID, ordered by chunk_index.

        Parameters
        ----------
        document_id:
            The ``NormalizedDocument.id`` (SHA-256 hex) to look up.
        session:
            Active ``AsyncSession``.

        Returns
        -------
        list[ChunkRecord]
            Chunks in source order (ascending ``chunk_index``).
            Empty list if no chunks exist for the document.
        """
        stmt = (
            select(ChunkRecord)
            .where(ChunkRecord.document_id == document_id)
            .order_by(ChunkRecord.chunk_index)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def count(self, session: AsyncSession) -> int:
        """Return the total number of chunk rows in the database.

        Parameters
        ----------
        session:
            Active ``AsyncSession``.

        Returns
        -------
        int
            Total row count in the ``chunks`` table.
        """
        stmt = select(func.count()).select_from(ChunkRecord)
        result = await session.execute(stmt)
        return result.scalar_one()

    async def delete_by_document_id(
        self,
        document_id: str,
        session: AsyncSession,
    ) -> int:
        """Delete all chunk records for a given document.

        Provided for completeness and future use (e.g. document removal or
        re-ingestion with explicit purge semantics).  Not called automatically
        by ``upsert_chunks`` — see the idempotency notes in the module docstring.

        Parameters
        ----------
        document_id:
            The document whose chunks to delete.
        session:
            Active ``AsyncSession``.

        Returns
        -------
        int
            Number of rows deleted.
        """
        stmt = delete(ChunkRecord).where(ChunkRecord.document_id == document_id)
        result = await session.execute(stmt)
        affected = result.rowcount
        logger.debug("delete_by_document_id(%r): %d rows deleted", document_id, affected)
        return affected
