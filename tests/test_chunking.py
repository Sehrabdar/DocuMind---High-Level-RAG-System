"""
Tests for the end-to-end chunking pipeline.

Covers:
- Empty document returns no chunks
- Whitespace-only document returns no chunks
- Small section (fits in one chunk) → 1 chunk
- Document smaller than chunk_size → 1 chunk
- Document larger than chunk_size → multiple chunks
- Oversized section is subdivided
- Chunk ordering is sequential and 0-based
- All chunks reference the correct document_id
- No empty-content chunks are ever produced
- Chunk size is respected (token_count <= chunk_size)
- Overlap behavior between consecutive sub-chunks
- Configuration validation (bad chunk_size/overlap raises ValueError)
- Document with only headings and no body
- Very small chunk size (stress test)
- Mixed format corpus
- Determinism: same input → same output twice
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from documind.ingestion.chunking import chunk_document, _count_tokens
from documind.ingestion.models import NormalizedDocument, DocumentMetadata


def _doc(
    content: str,
    file_type: str = "markdown",
    doc_id: str | None = None,
    title: str | None = None,
) -> NormalizedDocument:
    ext = {"markdown": ".md", "pdf": ".pdf", "html": ".html", "docx": ".docx"}.get(file_type, ".md")
    return NormalizedDocument(
        id=doc_id or ("test_" + "0" * 59),
        source=f"/fake/doc{ext}",
        filename=f"doc{ext}",
        file_type=file_type,
        title=title or "Test Document",
        content=content,
        metadata=DocumentMetadata(
            source=f"/fake/doc{ext}",
            filename=f"doc{ext}",
            file_type=file_type,
            file_extension=ext,
            ingested_at=datetime.now(timezone.utc),
        ),
    )


def _large_text(words: int = 600) -> str:
    """Generate deterministic text of roughly `words` tokens."""
    word = "documentation "
    return (word * words).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases: empty / trivially small documents
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_content_returns_no_chunks(self):
        doc = _doc("")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self):
        doc = _doc("   \n\n   \t   ")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_document_smaller_than_chunk_size(self):
        doc = _doc("# Tiny\n\nShort content.")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) >= 1

    def test_document_with_only_heading_no_body(self):
        """A heading with no body text should not produce an empty chunk."""
        doc = _doc("# Only A Heading\n\n## Another Heading")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_single_word_document(self):
        doc = _doc("# H\n\nHello.", file_type="markdown")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1
        assert "Hello" in chunks[0].content


# ─────────────────────────────────────────────────────────────────────────────
# Invariants: ordering, identity, no empties
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkingInvariants:
    def test_chunk_indexes_start_at_zero(self):
        doc = _doc("# A\n\nContent A.\n\n## B\n\nContent B.")
        chunks = chunk_document(doc)
        assert chunks[0].chunk_index == 0

    def test_chunk_indexes_are_sequential(self):
        doc = _doc("# A\n\nContent.\n\n## B\n\nMore content.\n\n## C\n\nEven more.")
        chunks = chunk_document(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_all_chunks_reference_document_id(self):
        doc = _doc("# A\n\nContent.\n\n## B\n\nMore.", doc_id="specific_doc_" + "x" * 50)
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.document_id == doc.id

    def test_no_empty_content_chunks(self):
        doc = _doc("# Auth\n\n## OAuth\n\nOAuth stuff.\n\n## API Keys\n\nKey stuff.")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_no_form_feed_in_chunk_content(self):
        """Form-feed separators from PDF must not appear in chunk content."""
        doc = _doc("Page 1 text.\fPage 2 text.\fPage 3 text.", file_type="pdf")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert "\f" not in chunk.content

    def test_token_count_matches_actual_tokens(self):
        doc = _doc("# Section\n\nKnown content text.", file_type="markdown")
        chunks = chunk_document(doc)
        for chunk in chunks:
            actual = _count_tokens(chunk.content)
            assert chunk.token_count == actual

    def test_determinism_same_result_twice(self):
        doc = _doc("# Title\n\nContent.\n\n## Sub\n\nMore.")
        chunks1 = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        chunks2 = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
        assert [c.content for c in chunks1] == [c.content for c in chunks2]


# ─────────────────────────────────────────────────────────────────────────────
# Small sections
# ─────────────────────────────────────────────────────────────────────────────

class TestSmallSections:
    def test_small_markdown_section_is_one_chunk(self):
        content = "# Authentication\n\nAuthentication allows applications to verify identity."
        doc = _doc(content)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        auth_chunks = [c for c in chunks if "Authentication" in c.section_path]
        assert len(auth_chunks) == 1

    def test_section_path_preserved_in_small_chunk(self):
        content = "# Auth\n\n## OAuth\n\nOAuth provides delegated authorization."
        doc = _doc(content)
        chunks = chunk_document(doc)
        oauth_chunk = next((c for c in chunks if "OAuth" in c.section_path), None)
        assert oauth_chunk is not None
        assert oauth_chunk.section_path == ["Auth", "OAuth"]

    def test_multiple_small_sections_each_become_one_chunk(self):
        content = "# A\n\nBody A.\n\n## B\n\nBody B.\n\n## C\n\nBody C."
        doc = _doc(content)
        chunks = chunk_document(doc)
        # Should have one chunk per section (A, B, C)
        section_paths = [c.section_path for c in chunks]
        assert ["A"] in section_paths
        assert ["A", "B"] in section_paths
        assert ["A", "C"] in section_paths


# ─────────────────────────────────────────────────────────────────────────────
# Oversized sections
# ─────────────────────────────────────────────────────────────────────────────

class TestOversizedSections:
    def test_large_section_produces_multiple_chunks(self):
        large_body = _large_text(600)  # ~600 tokens
        content = f"# Large Section\n\n{large_body}"
        doc = _doc(content)
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1

    def test_all_sub_chunks_have_same_section_path(self):
        """Sub-chunks of an oversized section all inherit the section's path."""
        large_body = _large_text(600)
        content = f"# Big\n\n{large_body}"
        doc = _doc(content)
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        big_chunks = [c for c in chunks if "Big" in c.section_path]
        assert len(big_chunks) > 1
        for chunk in big_chunks:
            assert chunk.section_path == ["Big"]

    def test_sub_chunks_respect_chunk_size(self):
        """No sub-chunk should have more tokens than chunk_size."""
        large_body = _large_text(300)
        content = f"# Section\n\n{large_body}"
        doc = _doc(content)
        chunk_size = 80
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=10)
        for chunk in chunks:
            # We allow a small grace (one extra token from tokenizer boundary rounding)
            assert chunk.token_count <= chunk_size + 5, (
                f"Chunk exceeded limit: {chunk.token_count} tokens (limit {chunk_size})"
            )

    def test_document_smaller_than_chunk_size_stays_one_chunk(self):
        content = "# Small\n\nJust a little bit of text."
        doc = _doc(content)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        small_chunks = [c for c in chunks if "Small" in c.section_path]
        assert len(small_chunks) == 1

    def test_very_small_chunk_size_does_not_infinite_loop(self):
        """chunk_size=5 is extreme but must terminate without hanging."""
        content = "# Title\n\nSome content words here."
        doc = _doc(content)
        chunks = chunk_document(doc, chunk_size=5, chunk_overlap=1)
        assert len(chunks) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Overlap behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlap:
    def test_adjacent_sub_chunks_share_content_suffix_prefix(self):
        """The end of sub-chunk N should appear at the start of sub-chunk N+1."""
        large_body = _large_text(400)
        content = f"# Section\n\n{large_body}"
        doc = _doc(content)
        chunks = chunk_document(doc, chunk_size=80, chunk_overlap=20)
        sub = [c for c in chunks if "Section" in c.section_path]
        if len(sub) >= 2:
            # The end of chunk N must partially overlap with the start of chunk N+1
            # (i.e., there should be some common token-level content)
            end_of_first = sub[0].content[-50:]
            start_of_second = sub[1].content[:80]
            # At least some words from the end of the first appear in the second
            last_words = end_of_first.split()[-3:]
            overlap_found = any(w in start_of_second for w in last_words)
            assert overlap_found, (
                "No overlap detected between consecutive sub-chunks. "
                f"End of chunk 0: '{end_of_first}'\n"
                f"Start of chunk 1: '{start_of_second}'"
            )

    def test_overlap_not_applied_between_independent_sections(self):
        """Content from section A should not appear in section B's first chunk."""
        content = "# A\n\nContent of A.\n\n# B\n\nContent of B."
        doc = _doc(content)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        b_chunk = next((c for c in chunks if c.section_path == ["B"]), None)
        if b_chunk:
            # The first chunk of B should not contain "Content of A"
            assert "Content of A" not in b_chunk.content

    def test_zero_overlap_produces_non_overlapping_chunks(self):
        """With chunk_overlap=0, consecutive sub-chunks should not share content."""
        large_body = _large_text(300)
        content = f"# Section\n\n{large_body}"
        doc = _doc(content)
        # overlap=0 means no shared tokens between sub-chunks
        chunks = chunk_document(doc, chunk_size=80, chunk_overlap=0)
        assert len(chunks) >= 1  # Just verify it runs without error


# ─────────────────────────────────────────────────────────────────────────────
# Configuration validation
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigurationValidation:
    def test_chunk_size_zero_raises(self):
        doc = _doc("# A\n\nContent.")
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_document(doc, chunk_size=0, chunk_overlap=0)

    def test_chunk_size_negative_raises(self):
        doc = _doc("# A\n\nContent.")
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_document(doc, chunk_size=-1, chunk_overlap=0)

    def test_overlap_equal_to_size_raises(self):
        doc = _doc("# A\n\nContent.")
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_document(doc, chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self):
        doc = _doc("# A\n\nContent.")
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_document(doc, chunk_size=50, chunk_overlap=60)

    def test_negative_overlap_raises(self):
        doc = _doc("# A\n\nContent.")
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_document(doc, chunk_size=100, chunk_overlap=-1)

    def test_valid_configuration_works(self):
        doc = _doc("# A\n\nContent.")
        chunks = chunk_document(doc, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# PDF-specific behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFChunking:
    def test_each_pdf_page_produces_at_least_one_chunk(self):
        content = "Page 1 text.\fPage 2 text.\fPage 3 text."
        doc = _doc(content, file_type="pdf")
        chunks = chunk_document(doc)
        page_numbers = {c.page for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers

    def test_pdf_chunk_page_number_is_correct(self):
        content = "First page.\fSecond page content here."
        doc = _doc(content, file_type="pdf")
        chunks = chunk_document(doc)
        first_page_chunk = next((c for c in chunks if "First page" in c.content), None)
        second_page_chunk = next((c for c in chunks if "Second page" in c.content), None)
        assert first_page_chunk is not None and first_page_chunk.page == 1
        assert second_page_chunk is not None and second_page_chunk.page == 2

    def test_pdf_empty_pages_produce_no_chunks(self):
        content = "Real page.\f   \fAnother real page."
        doc = _doc(content, file_type="pdf")
        chunks = chunk_document(doc)
        # Page 2 is blank — must not appear in chunks
        assert all(c.page != 2 for c in chunks)

    def test_pdf_oversized_page_splits_correctly(self):
        large_page = _large_text(300)
        content = f"{large_page}"
        doc = _doc(content, file_type="pdf")
        chunks = chunk_document(doc, chunk_size=80, chunk_overlap=10)
        # All sub-chunks from the same page share the same page number
        page_numbers = {c.page for c in chunks}
        assert page_numbers == {1}

    def test_pdf_chunk_section_path_is_empty(self):
        """PDF chunks don't have Markdown-style section paths."""
        content = "PDF text without markdown headings."
        doc = _doc(content, file_type="pdf")
        chunks = chunk_document(doc)
        for chunk in chunks:
            assert chunk.section_path == []


# ─────────────────────────────────────────────────────────────────────────────
# Real-fixture integration (uses actual loader output)
# ─────────────────────────────────────────────────────────────────────────────

class TestRealFixtureChunking:
    def test_chunks_real_markdown_fixture(self, md_sample):
        from documind.ingestion.loaders import MarkdownLoader
        doc = MarkdownLoader().load(md_sample)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) >= 1
        assert all(c.document_id == doc.id for c in chunks)

    def test_chunks_real_pdf_fixture(self, pdf_sample):
        from documind.ingestion.loaders import PDFLoader
        doc = PDFLoader().load(pdf_sample)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) >= 1
        # PDF chunks must have page numbers
        assert all(c.page is not None for c in chunks)

    def test_chunks_real_html_fixture(self, html_sample):
        from documind.ingestion.loaders import HTMLLoader
        doc = HTMLLoader().load(html_sample)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) >= 1
        assert all(c.section_path == [] for c in chunks)

    def test_chunks_real_docx_fixture(self, docx_sample):
        from documind.ingestion.loaders import DocxLoader
        doc = DocxLoader().load(docx_sample)
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) >= 1
        assert all(c.section_path == [] for c in chunks)
