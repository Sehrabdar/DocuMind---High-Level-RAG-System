"""
Tests for deterministic chunk ID generation.

Covers:
- Same inputs → same chunk_id
- Different document_id → different chunk_id
- Different chunk_index → different chunk_id
- Different content → different chunk_id
- chunk_id is a 64-character lowercase hex string (SHA-256)
- chunk_document() produces identical IDs across runs (end-to-end determinism)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from documind.ingestion.chunking import _chunk_id, chunk_document
from documind.ingestion.models import NormalizedDocument, DocumentMetadata
from datetime import datetime, timezone


def _make_doc(content: str, file_type: str = "markdown") -> NormalizedDocument:
    return NormalizedDocument(
        id="doc_" + "a" * 60,
        source="/fake/path.md",
        filename="path.md",
        file_type=file_type,
        title="Test",
        content=content,
        metadata=DocumentMetadata(
            source="/fake/path.md",
            filename="path.md",
            file_type=file_type,
            file_extension=".md",
            ingested_at=datetime.now(timezone.utc),
        ),
    )


class TestChunkIdAlgorithm:
    def test_same_inputs_produce_same_id(self):
        id1 = _chunk_id("doc123", 0, "hello world")
        id2 = _chunk_id("doc123", 0, "hello world")
        assert id1 == id2

    def test_is_64_char_hex_string(self):
        cid = _chunk_id("doc123", 0, "some content here")
        assert len(cid) == 64
        assert all(c in "0123456789abcdef" for c in cid)

    def test_different_document_id_produces_different_chunk_id(self):
        id1 = _chunk_id("doc_aaa", 0, "same content")
        id2 = _chunk_id("doc_bbb", 0, "same content")
        assert id1 != id2

    def test_different_chunk_index_produces_different_chunk_id(self):
        id1 = _chunk_id("doc123", 0, "same content")
        id2 = _chunk_id("doc123", 1, "same content")
        assert id1 != id2

    def test_different_content_produces_different_chunk_id(self):
        id1 = _chunk_id("doc123", 0, "content A")
        id2 = _chunk_id("doc123", 0, "content B")
        assert id1 != id2

    def test_call_order_does_not_affect_id(self):
        # Compute IDs in both orders
        a_first = _chunk_id("docA", 0, "text")
        b_first = _chunk_id("docB", 0, "text")
        b_second = _chunk_id("docB", 0, "text")
        a_second = _chunk_id("docA", 0, "text")
        assert a_first == a_second
        assert b_first == b_second

    def test_empty_content_is_handled(self):
        # The algorithm uses content[:64], so empty string is valid
        cid = _chunk_id("doc123", 0, "")
        assert len(cid) == 64


class TestChunkDocumentDeterminism:
    def test_same_markdown_doc_same_ids(self):
        """Chunking the same Markdown document twice produces identical chunk IDs."""
        doc = _make_doc(
            "# Auth\n\nSome text.\n\n## OAuth\n\nOAuth details here.\n",
            file_type="markdown",
        )
        chunks1 = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        chunks2 = chunk_document(doc, chunk_size=512, chunk_overlap=50)

        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_same_pdf_doc_same_ids(self):
        """Chunking the same PDF document twice produces identical chunk IDs."""
        doc = _make_doc("Page one text.\f Page two text.", file_type="pdf")
        chunks1 = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        chunks2 = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]

    def test_all_chunk_ids_unique_within_document(self):
        """No two chunks from the same document should share an ID."""
        doc = _make_doc(
            "# A\n\nContent A.\n\n## B\n\nContent B.\n\n## C\n\nContent C.\n",
            file_type="markdown",
        )
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunk_indexes_are_unique(self):
        doc = _make_doc(
            "# A\n\nContent A.\n\n## B\n\nContent B.\n",
            file_type="markdown",
        )
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        indexes = [c.chunk_index for c in chunks]
        assert len(indexes) == len(set(indexes))

    def test_ids_independent_of_previous_chunk_count(self):
        """IDs don't shift if we chunk a different document before this one."""
        doc = _make_doc("# Header\n\nBody text.", file_type="markdown")

        # Chunk doc alone
        chunks_alone = chunk_document(doc, chunk_size=512, chunk_overlap=50)

        # Chunk another doc first (simulating pipeline order)
        other_doc = _make_doc("# Other\n\nOther content.", file_type="markdown")
        chunk_document(other_doc)

        chunks_after = chunk_document(doc, chunk_size=512, chunk_overlap=50)

        assert [c.chunk_id for c in chunks_alone] == [c.chunk_id for c in chunks_after]
