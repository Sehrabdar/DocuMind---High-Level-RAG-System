"""
Unit tests for the db models layer (no database connection required).

Covers:
- ChunkRecord can be constructed with all fields
- chunk_record_from_normalized() adapter correctness
- section_path is stored as a list (JSONB)
- document_title can be None
- page fields can be None
- created_at is set on construction
- ChunkRecord.__repr__ is sane
- Field names match between NormalizedChunk and ChunkRecord
- Embedding field accepts list[float]
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from db.models import ChunkRecord, chunk_record_from_normalized
from documind.ingestion.chunking import NormalizedChunk


def _make_chunk(**overrides) -> NormalizedChunk:
    """Build a minimal valid NormalizedChunk for testing."""
    defaults = dict(
        chunk_id="a" * 64,
        document_id="b" * 64,
        chunk_index=0,
        content="This is test chunk content for database model testing.",
        token_count=10,
        section_path=["Authentication", "OAuth"],
        document_title="API Reference",
        start_char=0,
        end_char=54,
        page=None,
        page_start=None,
        page_end=None,
    )
    defaults.update(overrides)
    return NormalizedChunk(**defaults)


def _make_embedding(dim: int = 384) -> list[float]:
    """Return a simple deterministic embedding of the given dimension."""
    import math
    raw = [float(i % 10) + 1.0 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw))
    return [v / norm for v in raw]


class TestChunkRecord:
    def test_can_construct_chunk_record(self):
        record = ChunkRecord(
            chunk_id="a" * 64,
            document_id="b" * 64,
            chunk_index=0,
            content="Some content.",
            token_count=3,
            embedding=_make_embedding(),
            section_path=["Auth"],
            document_title="Test Doc",
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=13,
            created_at=datetime.now(timezone.utc),
        )
        assert record.chunk_id == "a" * 64
        assert record.chunk_index == 0

    def test_chunk_id_is_string(self):
        record = ChunkRecord(
            chunk_id="abc123" + "x" * 58,
            document_id="d" * 64,
            chunk_index=1,
            content="content",
            token_count=1,
            embedding=_make_embedding(),
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=7,
            created_at=datetime.now(timezone.utc),
        )
        assert isinstance(record.chunk_id, str)

    def test_section_path_is_list(self):
        record = ChunkRecord(
            chunk_id="c" * 64,
            document_id="d" * 64,
            chunk_index=0,
            content="text",
            token_count=1,
            embedding=_make_embedding(),
            section_path=["A", "B", "C"],
            document_title="Doc",
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=4,
            created_at=datetime.now(timezone.utc),
        )
        assert isinstance(record.section_path, list)
        assert record.section_path == ["A", "B", "C"]

    def test_empty_section_path_accepted(self):
        record = ChunkRecord(
            chunk_id="e" * 64,
            document_id="f" * 64,
            chunk_index=0,
            content="content",
            token_count=1,
            embedding=_make_embedding(),
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=7,
            created_at=datetime.now(timezone.utc),
        )
        assert record.section_path == []

    def test_document_title_can_be_none(self):
        record = ChunkRecord(
            chunk_id="g" * 64,
            document_id="h" * 64,
            chunk_index=0,
            content="content",
            token_count=1,
            embedding=_make_embedding(),
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=7,
            created_at=datetime.now(timezone.utc),
        )
        assert record.document_title is None

    def test_page_fields_can_be_none(self):
        record = ChunkRecord(
            chunk_id="i" * 64,
            document_id="j" * 64,
            chunk_index=0,
            content="text",
            token_count=1,
            embedding=_make_embedding(),
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=4,
            created_at=datetime.now(timezone.utc),
        )
        assert record.page is None
        assert record.page_start is None
        assert record.page_end is None

    def test_page_can_be_set(self):
        record = ChunkRecord(
            chunk_id="k" * 64,
            document_id="l" * 64,
            chunk_index=0,
            content="pdf text",
            token_count=2,
            embedding=_make_embedding(),
            section_path=[],
            document_title=None,
            page=3,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=8,
            created_at=datetime.now(timezone.utc),
        )
        assert record.page == 3

    def test_embedding_accepts_list_of_floats(self):
        emb = _make_embedding(384)
        record = ChunkRecord(
            chunk_id="m" * 64,
            document_id="n" * 64,
            chunk_index=0,
            content="text",
            token_count=1,
            embedding=emb,
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=4,
            created_at=datetime.now(timezone.utc),
        )
        assert record.embedding == emb

    def test_repr_contains_chunk_id(self):
        cid = "r" * 64
        record = ChunkRecord(
            chunk_id=cid,
            document_id="s" * 64,
            chunk_index=5,
            content="text",
            token_count=1,
            embedding=_make_embedding(),
            section_path=[],
            document_title=None,
            page=None,
            page_start=None,
            page_end=None,
            start_char=0,
            end_char=4,
            created_at=datetime.now(timezone.utc),
        )
        assert cid in repr(record)
        assert "5" in repr(record)


class TestChunkRecordFromNormalized:
    def test_adapter_copies_chunk_id(self):
        chunk = _make_chunk(chunk_id="a" * 64)
        emb = _make_embedding()
        record = chunk_record_from_normalized(chunk, emb)
        assert record.chunk_id == "a" * 64

    def test_adapter_copies_document_id(self):
        chunk = _make_chunk(document_id="b" * 64)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.document_id == "b" * 64

    def test_adapter_copies_chunk_index(self):
        chunk = _make_chunk(chunk_index=7)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.chunk_index == 7

    def test_adapter_copies_content(self):
        chunk = _make_chunk(content="specific content text here")
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.content == "specific content text here"

    def test_adapter_copies_token_count(self):
        chunk = _make_chunk(token_count=42)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.token_count == 42

    def test_adapter_copies_embedding(self):
        emb = _make_embedding()
        chunk = _make_chunk()
        record = chunk_record_from_normalized(chunk, emb)
        assert record.embedding == emb

    def test_adapter_copies_section_path(self):
        chunk = _make_chunk(section_path=["Auth", "OAuth", "Tokens"])
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.section_path == ["Auth", "OAuth", "Tokens"]

    def test_adapter_section_path_is_copy(self):
        """Modifying the original section_path should not affect the record."""
        path = ["A", "B"]
        chunk = _make_chunk(section_path=path)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        path.append("MUTATED")
        assert record.section_path == ["A", "B"]

    def test_adapter_copies_document_title(self):
        chunk = _make_chunk(document_title="My Document Title")
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.document_title == "My Document Title"

    def test_adapter_copies_none_title(self):
        chunk = _make_chunk(document_title=None)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.document_title is None

    def test_adapter_copies_page(self):
        chunk = _make_chunk(page=5)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.page == 5

    def test_adapter_copies_none_page(self):
        chunk = _make_chunk(page=None)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.page is None

    def test_adapter_copies_start_char(self):
        chunk = _make_chunk(start_char=100, end_char=200)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.start_char == 100

    def test_adapter_copies_end_char(self):
        chunk = _make_chunk(start_char=100, end_char=200)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert record.end_char == 200

    def test_adapter_sets_created_at(self):
        chunk = _make_chunk()
        before = datetime.now(timezone.utc)
        record = chunk_record_from_normalized(chunk, _make_embedding())
        after = datetime.now(timezone.utc)
        assert record.created_at is not None
        assert before <= record.created_at <= after

    def test_adapter_returns_chunk_record_type(self):
        chunk = _make_chunk()
        record = chunk_record_from_normalized(chunk, _make_embedding())
        assert isinstance(record, ChunkRecord)
