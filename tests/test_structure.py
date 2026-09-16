"""
Tests for structure detection — Markdown heading hierarchy and PDF page splitting.

Covers:
- Markdown heading levels 1–6
- Correct path for nested headings
- Heading stack resets correctly (## D after ### C → ["A", "D"] not ["A","B","C","D"])
- Preamble content before first heading gets empty section_path
- Consecutive headings without body text
- Document with no headings
- Document with only headings
- Deeply nested hierarchy
- PDF page splitting via form-feed separator
- Empty PDF pages are skipped
- Single-page PDF
- Multi-page PDF with correct page numbers
- HTML / DOCX flat fallback
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from documind.ingestion.chunking import (
    _extract_sections_markdown,
    _split_pdf_pages,
    _extract_sections_flat,
    chunk_document,
)
from documind.ingestion.models import NormalizedDocument, DocumentMetadata


def _doc(content: str, file_type: str = "markdown", title: str | None = None) -> NormalizedDocument:
    ext = {"markdown": ".md", "pdf": ".pdf", "html": ".html", "docx": ".docx"}.get(file_type, ".md")
    return NormalizedDocument(
        id="test_" + "0" * 59,
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


# ─────────────────────────────────────────────────────────────────────────────
# Markdown heading hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class TestMarkdownHeadingHierarchy:
    def test_single_h1_section(self):
        content = "# Authentication\n\nSome text here."
        sections = _extract_sections_markdown(content)
        body_sections = [s for s in sections if s.content]
        assert len(body_sections) == 1
        assert body_sections[0].section_path == ["Authentication"]
        assert "Some text here." in body_sections[0].content

    def test_h1_then_h2(self):
        content = "# Authentication\n\n## OAuth\n\nOAuth text."
        sections = _extract_sections_markdown(content)
        paths = [s.section_path for s in sections if s.content]
        assert ["Authentication", "OAuth"] in paths

    def test_h1_h2_h3_nesting(self):
        content = "# A\n\n## B\n\n### C\n\nDeep content."
        sections = _extract_sections_markdown(content)
        paths = [s.section_path for s in sections if s.content]
        assert ["A", "B", "C"] in paths

    def test_heading_level_reset(self):
        """## D after ### C must produce path ["A", "D"] not ["A", "B", "C", "D"]."""
        content = "# A\n\n## B\n\n### C\n\nC text.\n\n## D\n\nD text."
        sections = _extract_sections_markdown(content)
        d_section = next((s for s in sections if "D text." in s.content), None)
        assert d_section is not None
        assert d_section.section_path == ["A", "D"], (
            f"Expected ['A', 'D'] but got {d_section.section_path}"
        )

    def test_preamble_has_empty_section_path(self):
        """Content before the first heading has an empty section_path."""
        content = "This is preamble.\n\n# Introduction\n\nBody."
        sections = _extract_sections_markdown(content)
        preamble = next((s for s in sections if "preamble" in s.content), None)
        assert preamble is not None
        assert preamble.section_path == []

    def test_document_with_no_headings(self):
        content = "Just plain text.\n\nMore text here."
        sections = _extract_sections_markdown(content)
        # All content in one section with empty path
        assert len(sections) >= 1
        assert all(s.section_path == [] for s in sections)

    def test_document_with_only_h1(self):
        content = "# Title Only\n\nBody content."
        sections = _extract_sections_markdown(content)
        assert any(s.section_path == ["Title Only"] for s in sections)

    def test_consecutive_headings_no_orphan_sections(self):
        """Two adjacent headings without body between them should not emit empty sections."""
        content = "# A\n\n## B\n\nSome body content."
        sections = _extract_sections_markdown(content)
        # No section should have empty stripped content
        for s in sections:
            assert s.content.strip() != ""

    def test_h6_heading_supported(self):
        content = "# A\n\n###### Deep\n\nVery deep content."
        sections = _extract_sections_markdown(content)
        deep = next((s for s in sections if "Very deep" in s.content), None)
        assert deep is not None
        assert "Deep" in deep.section_path

    def test_multiple_h2_under_h1(self):
        """Multiple H2 sections under one H1 should all have ['H1', 'H2n'] paths."""
        content = "# Parent\n\n## Child A\n\nText A.\n\n## Child B\n\nText B."
        sections = _extract_sections_markdown(content)
        paths = [s.section_path for s in sections if s.content]
        assert ["Parent", "Child A"] in paths
        assert ["Parent", "Child B"] in paths

    def test_section_path_is_copy_not_reference(self):
        """Each section's section_path must be an independent list, not a shared reference."""
        content = "# A\n\n## B\n\nText B.\n\n## C\n\nText C."
        sections = _extract_sections_markdown(content)
        paths = [s.section_path for s in sections if s.content]
        # Modifying one path must not affect another
        if len(paths) >= 2:
            original = list(paths[0])
            paths[0].append("MUTATED")
            assert paths[1] != paths[0]


class TestMarkdownSectionStartChar:
    def test_start_char_is_after_heading_line(self):
        """The section body start_char should point past the heading line."""
        content = "# Header\n\nBody content here."
        sections = _extract_sections_markdown(content)
        body = next((s for s in sections if "Body content" in s.content), None)
        assert body is not None
        # start_char should be after "# Header\n" (9 chars)
        assert body.start_char >= len("# Header\n")

    def test_start_char_for_preamble_is_zero(self):
        content = "Preamble text.\n\n# Header\n\nBody."
        sections = _extract_sections_markdown(content)
        preamble = next((s for s in sections if "Preamble" in s.content), None)
        if preamble:
            assert preamble.start_char == 0


# ─────────────────────────────────────────────────────────────────────────────
# PDF page splitting
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFPageSplitting:
    def test_single_page_pdf(self):
        content = "Only one page of content."
        sections = _split_pdf_pages(content)
        assert len(sections) == 1
        assert sections[0].page == 1

    def test_two_page_pdf(self):
        content = "Page one content.\f Page two content."
        sections = _split_pdf_pages(content)
        assert len(sections) == 2
        assert sections[0].page == 1
        assert sections[1].page == 2

    def test_three_page_pdf(self):
        content = "Page 1.\fPage 2.\fPage 3."
        sections = _split_pdf_pages(content)
        assert len(sections) == 3
        pages = [s.page for s in sections]
        assert pages == [1, 2, 3]

    def test_empty_page_is_skipped(self):
        """A blank page (whitespace-only after stripping) should not produce a section."""
        content = "Page 1.\f   \f Page 3."
        sections = _split_pdf_pages(content)
        page_numbers = [s.page for s in sections]
        # Page 2 is empty — should be skipped
        assert 2 not in page_numbers
        assert 1 in page_numbers
        assert 3 in page_numbers

    def test_page_content_strips_form_feeds(self):
        """The \f character itself should not appear in chunk content."""
        content = "Page A.\fPage B."
        sections = _split_pdf_pages(content)
        for s in sections:
            assert "\f" not in s.content

    def test_page_numbers_are_one_indexed(self):
        content = "Text.\fMore text."
        sections = _split_pdf_pages(content)
        assert min(s.page for s in sections) == 1

    def test_single_page_section_has_no_page_start_end(self):
        """Single-page sections use page=N, not page_start/page_end."""
        content = "One page."
        sections = _split_pdf_pages(content)
        assert sections[0].page_start is None
        assert sections[0].page_end is None


# ─────────────────────────────────────────────────────────────────────────────
# HTML / DOCX flat fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestFlatSectionFallback:
    def test_html_produces_single_section(self):
        content = "Introduction\n\nThis document covers authentication.\n\nMore details."
        sections = _extract_sections_flat(content, document_title="API Docs")
        assert len(sections) == 1

    def test_html_section_path_is_empty(self):
        """HTML/DOCX flat sections have empty section_path (no heading hierarchy)."""
        content = "Some HTML content."
        sections = _extract_sections_flat(content, document_title="Doc")
        assert sections[0].section_path == []

    def test_empty_content_returns_no_sections(self):
        sections = _extract_sections_flat("", document_title=None)
        assert sections == []

    def test_whitespace_only_returns_no_sections(self):
        sections = _extract_sections_flat("   \n\n   ", document_title=None)
        assert sections == []

    def test_chunk_document_html_returns_empty_section_paths(self):
        doc = _doc("Header text\n\nBody content.", file_type="html", title="Doc")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.section_path == []

    def test_chunk_document_docx_returns_empty_section_paths(self):
        doc = _doc("Paragraph one.\n\nParagraph two.", file_type="docx", title="Doc")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.section_path == []
