"""
Tests for chunk metadata — provenance, offsets, page info, config validation.

Covers:
- start_char and end_char are within bounds
- document_id matches source document
- document_title is propagated correctly
- PDF page numbers are 1-indexed and correct
- section_path entries are strings
- token_count is positive and consistent with content
- start_char <= end_char always
- Character offsets are deterministic
- Settings-level config validation (chunk_overlap >= chunk_size)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from documind.ingestion.chunking import chunk_document, _count_tokens
from documind.ingestion.models import NormalizedDocument, DocumentMetadata


def _doc(
    content: str,
    file_type: str = "markdown",
    doc_id: str = "test_" + "a" * 59,
    title: str | None = "Test Document",
) -> NormalizedDocument:
    ext = {"markdown": ".md", "pdf": ".pdf", "html": ".html", "docx": ".docx"}.get(file_type, ".md")
    return NormalizedDocument(
        id=doc_id,
        source=f"/fake/doc{ext}",
        filename=f"doc{ext}",
        file_type=file_type,
        title=title,
        content=content,
        metadata=DocumentMetadata(
            source=f"/fake/doc{ext}",
            filename=f"doc{ext}",
            file_type=file_type,
            file_extension=ext,
            ingested_at=datetime.now(timezone.utc),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Character offset invariants
# ─────────────────────────────────────────────────────────────────────────────

class TestCharacterOffsets:
    def test_start_char_is_non_negative(self):
        doc = _doc("# Section\n\nContent here.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.start_char >= 0, f"Negative start_char on chunk {chunk.chunk_index}"

    def test_end_char_gte_start_char(self):
        doc = _doc("# Section\n\nContent here.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.end_char >= chunk.start_char

    def test_end_char_does_not_exceed_document_length(self):
        content = "# Section\n\nSome content here that should be within bounds."
        doc = _doc(content)
        chunks = chunk_document(doc)
        doc_len = len(content)
        for chunk in chunks:
            assert chunk.end_char <= doc_len, (
                f"end_char {chunk.end_char} exceeds document length {doc_len}"
            )

    def test_offsets_are_deterministic(self):
        """Same document chunked twice produces identical offsets."""
        content = "# Auth\n\nAuth content.\n\n## OAuth\n\nOAuth content."
        doc = _doc(content)
        chunks1 = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        chunks2 = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.start_char == c2.start_char
            assert c1.end_char == c2.end_char

    def test_pdf_offsets_within_bounds(self):
        content = "Page one text.\fPage two text."
        doc = _doc(content, file_type="pdf")
        chunks = chunk_document(doc)
        doc_len = len(content)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char <= doc_len


# ─────────────────────────────────────────────────────────────────────────────
# Document identity in metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestDocumentIdentityInChunks:
    def test_document_id_matches_source(self):
        custom_id = "custom_doc_" + "z" * 53
        doc = _doc("# H\n\nBody.", doc_id=custom_id)
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_id == custom_id

    def test_document_title_propagated(self):
        doc = _doc("# Section\n\nContent.", title="My Special Document")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_title == "My Special Document"

    def test_document_title_none_propagated(self):
        doc = _doc("# Section\n\nContent.", title=None)
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_title is None


# ─────────────────────────────────────────────────────────────────────────────
# Token count metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenCountMetadata:
    def test_token_count_is_positive(self):
        doc = _doc("# A\n\nSome content here.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.token_count > 0

    def test_token_count_matches_content(self):
        doc = _doc("# A\n\nKnown content text.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            expected = _count_tokens(chunk.content)
            assert chunk.token_count == expected

    def test_token_count_consistent_across_runs(self):
        doc = _doc("# Section\n\nContent here.")
        chunks1 = chunk_document(doc)
        chunks2 = chunk_document(doc)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.token_count == c2.token_count


# ─────────────────────────────────────────────────────────────────────────────
# PDF page provenance
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFPageProvenance:
    def test_page_is_one_indexed(self):
        doc = _doc("First page.\fSecond page.", file_type="pdf")
        chunks = chunk_document(doc)
        pages = [c.page for c in chunks if c.page is not None]
        assert min(pages) == 1

    def test_page_numbers_cover_all_non_empty_pages(self):
        doc = _doc("Page 1.\fPage 2.\fPage 3.", file_type="pdf")
        chunks = chunk_document(doc)
        page_numbers = sorted({c.page for c in chunks if c.page is not None})
        assert page_numbers == [1, 2, 3]

    def test_non_pdf_page_is_none(self):
        doc = _doc("# Auth\n\nContent.", file_type="markdown")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.page is None

    def test_html_page_is_none(self):
        doc = _doc("Content.", file_type="html")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.page is None

    def test_docx_page_is_none(self):
        doc = _doc("Content.", file_type="docx")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.page is None


# ─────────────────────────────────────────────────────────────────────────────
# Section path metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestSectionPathMetadata:
    def test_section_path_entries_are_strings(self):
        doc = _doc("# Auth\n\n## OAuth\n\nContent.", file_type="markdown")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert all(isinstance(s, str) for s in chunk.section_path)

    def test_section_path_depth_matches_heading_level(self):
        content = "# L1\n\n## L2\n\n### L3\n\nDeep content."
        doc = _doc(content)
        chunks = chunk_document(doc)
        deep_chunk = next((c for c in chunks if "Deep content" in c.content), None)
        if deep_chunk:
            assert len(deep_chunk.section_path) == 3

    def test_markdown_preamble_has_empty_section_path(self):
        content = "Preamble text before any heading.\n\n# Header\n\nBody."
        doc = _doc(content)
        chunks = chunk_document(doc)
        preamble_chunk = next((c for c in chunks if "Preamble" in c.content), None)
        if preamble_chunk:
            assert preamble_chunk.section_path == []


# ─────────────────────────────────────────────────────────────────────────────
# Settings-level config validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSettingsValidation:
    def test_settings_rejects_zero_chunk_size(self):
        from pydantic import ValidationError
        from config import Settings
        # Pass directly to constructor — this takes precedence over env vars
        with pytest.raises((ValueError, ValidationError)):
            Settings(chunk_size=0, chunk_overlap=0)

    def test_settings_rejects_negative_chunk_size(self):
        from pydantic import ValidationError
        from config import Settings
        with pytest.raises((ValueError, ValidationError)):
            Settings(chunk_size=-10, chunk_overlap=0)

    def test_settings_rejects_overlap_equal_to_chunk_size(self):
        from pydantic import ValidationError
        from config import Settings
        with pytest.raises((ValueError, ValidationError)):
            Settings(chunk_size=50, chunk_overlap=50)

    def test_settings_rejects_overlap_greater_than_chunk_size(self):
        from pydantic import ValidationError
        from config import Settings
        with pytest.raises((ValueError, ValidationError)):
            Settings(chunk_size=50, chunk_overlap=60)

    def test_settings_rejects_negative_overlap(self):
        from pydantic import ValidationError
        from config import Settings
        with pytest.raises((ValueError, ValidationError)):
            Settings(chunk_size=100, chunk_overlap=-1)

    def test_settings_accepts_valid_chunk_config(self):
        from config import Settings
        s = Settings(chunk_size=256, chunk_overlap=25)
        assert s.chunk_size == 256
        assert s.chunk_overlap == 25

    def test_settings_defaults_are_sensible(self):
        from config import Settings
        s = Settings()
        assert s.chunk_size > 0
        assert s.chunk_overlap >= 0
        assert s.chunk_overlap < s.chunk_size
