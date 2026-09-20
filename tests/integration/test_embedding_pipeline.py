"""
End-to-end integration test: NormalizedChunk → embedding → database persistence.

This is the primary Phase 3 integration test demonstrating the full pipeline:

    NormalizedChunk (from Phase 2 chunking)
          ↓
    EmbeddingService with FakeEmbeddingProvider (deterministic, no model download)
          ↓
    chunk_record_from_normalized() adapter
          ↓
    ChunkRepository.upsert_chunks()
          ↓
    PostgreSQL (pgvector)
          ↓
    Verify: correct content, embedding, metadata stored

A separate model_integration test (test_real_embeddings.py) verifies the
actual BAAI/bge-small-en-v1.5 model when it is available.
"""

from __future__ import annotations

import math

import pytest

from db.models import chunk_record_from_normalized
from db.repository import ChunkRepository
from documind.embeddings.service import EmbeddingService, FakeEmbeddingProvider
from documind.ingestion.chunking import NormalizedChunk
from tests.integration.conftest import requires_db


def _cid(prefix: str) -> str:
    """64-char chunk_id, padded with 'a'."""
    return (prefix + "a" * 64)[:64]


def _did(prefix: str) -> str:
    """64-char document_id, padded with 'b'."""
    return (prefix + "b" * 64)[:64]


def _make_normalized_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    chunk_index: int = 0,
    section_path: list[str] | None = None,
    page: int | None = None,
) -> NormalizedChunk:
    assert len(chunk_id) <= 64
    assert len(document_id) <= 64
    return NormalizedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        token_count=len(content.split()),
        section_path=section_path or [],
        document_title="Integration Test Document",
        start_char=0,
        end_char=len(content),
        page=page,
        page_start=None,
        page_end=None,
    )


@pytest.fixture
def embedding_service() -> EmbeddingService:
    """EmbeddingService backed by FakeEmbeddingProvider — no model download."""
    provider = FakeEmbeddingProvider(dimension=384)
    return EmbeddingService(provider)


@pytest.mark.integration
@requires_db
class TestEndToEndPipeline:

    @pytest.mark.asyncio
    async def test_chunk_embed_persist_round_trip(self, db_session, embedding_service):
        """The primary Phase 3 integration test.

        NormalizedChunk → EmbeddingService → ChunkRepository → verify in DB.
        """
        chunk = _make_normalized_chunk(
            chunk_id=_cid("e2e_round_trip"),
            document_id=_did("e2e_doc_rt"),
            content="Authentication allows applications to verify user identity.",
            chunk_index=0,
            section_path=["Authentication"],
        )

        embedding = embedding_service.embed_text(chunk.content)
        assert len(embedding) == 384

        norm = math.sqrt(sum(v * v for v in embedding))
        assert abs(norm - 1.0) < 1e-5

        record = chunk_record_from_normalized(chunk, embedding)
        repo = ChunkRepository()
        count = await repo.upsert_chunks([record], db_session)
        assert count >= 1

        retrieved = await repo.get_by_document_id(chunk.document_id, db_session)
        assert len(retrieved) == 1

        stored = retrieved[0]
        assert stored.chunk_id == chunk.chunk_id
        assert stored.document_id == chunk.document_id
        assert stored.content == chunk.content
        assert stored.chunk_index == chunk.chunk_index
        assert stored.section_path == chunk.section_path
        assert stored.token_count == chunk.token_count
        assert stored.start_char == chunk.start_char
        assert stored.end_char == chunk.end_char

        assert len(stored.embedding) == 384
        for i in range(min(10, len(embedding))):
            assert abs(stored.embedding[i] - embedding[i]) < 1e-6

    @pytest.mark.asyncio
    async def test_batch_embed_persist(self, db_session, embedding_service):
        """Multiple chunks embedded and persisted in batch."""
        doc_id = _did("batch_e2e_doc")
        contents = [
            "API keys are used for server-to-server authentication.",
            "OAuth provides delegated authorization for third-party applications.",
            "JWT tokens contain encoded claims about the authenticated user.",
        ]
        chunks = [
            _make_normalized_chunk(
                chunk_id=_cid(f"batch_e2e_{i}"),
                document_id=doc_id,
                content=content,
                chunk_index=i,
                section_path=["Authentication", f"Section {i}"],
            )
            for i, content in enumerate(contents)
        ]

        texts = [c.content for c in chunks]
        embeddings = embedding_service.embed_texts(texts)
        assert len(embeddings) == len(chunks)

        records = [
            chunk_record_from_normalized(chunk, emb)
            for chunk, emb in zip(chunks, embeddings)
        ]

        repo = ChunkRepository()
        count = await repo.upsert_chunks(records, db_session)
        assert count == len(chunks)

        retrieved = await repo.get_by_document_id(doc_id, db_session)
        assert len(retrieved) == len(chunks)
        assert [r.chunk_index for r in retrieved] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_pdf_chunk_page_metadata_persisted(self, db_session, embedding_service):
        """PDF page number must survive the full persist/retrieve cycle."""
        doc_id = _did("pdf_e2e_doc")
        chunk = _make_normalized_chunk(
            chunk_id=_cid("pdf_e2e_chunk"),
            document_id=doc_id,
            content="This is the content of page 3 of a PDF document.",
            page=3,
        )
        emb = embedding_service.embed_text(chunk.content)
        record = chunk_record_from_normalized(chunk, emb)

        repo = ChunkRepository()
        await repo.upsert_chunks([record], db_session)
        retrieved = await repo.get_by_document_id(doc_id, db_session)

        assert retrieved[0].page == 3

    @pytest.mark.asyncio
    async def test_idempotent_reingest_same_chunks(self, db_session, embedding_service):
        """Re-ingesting the same chunks produces no duplicates."""
        doc_id = _did("idem_e2e_doc")
        chunk = _make_normalized_chunk(
            chunk_id=_cid("idem_e2e_chunk"),
            document_id=doc_id,
            content="This content does not change between ingestion runs.",
        )
        emb = embedding_service.embed_text(chunk.content)
        record = chunk_record_from_normalized(chunk, emb)

        repo = ChunkRepository()
        await repo.upsert_chunks([record], db_session)
        await repo.upsert_chunks([record], db_session)

        retrieved = await repo.get_by_document_id(doc_id, db_session)
        assert len(retrieved) == 1, "Idempotent re-ingestion must not create duplicates"

    @pytest.mark.asyncio
    async def test_embedding_determinism_across_persist_cycles(self, db_session, embedding_service):
        """Same text always produces the same embedding in the DB."""
        text = "Determinism is a core requirement for reproducible RAG pipelines."
        doc_id = _did("deter_e2e_doc")

        emb1 = embedding_service.embed_text(text)
        chunk = _make_normalized_chunk(
            chunk_id=_cid("deter_e2e_chunk"),
            document_id=doc_id,
            content=text,
        )
        record = chunk_record_from_normalized(chunk, emb1)

        repo = ChunkRepository()
        await repo.upsert_chunks([record], db_session)

        retrieved = await repo.get_by_document_id(doc_id, db_session)
        stored_emb = retrieved[0].embedding

        emb2 = embedding_service.embed_text(text)
        for i in range(min(20, len(emb1))):
            assert abs(stored_emb[i] - emb2[i]) < 1e-6
