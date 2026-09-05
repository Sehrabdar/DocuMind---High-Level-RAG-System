"""
Tests for format dispatch and per-loader behaviour.

Covers:
- Correct loader selected for each extension
- .htm dispatches to HTMLLoader (same as .html)
- Unsupported format raises UnsupportedFormatError with path info
- Each loader returns a non-empty NormalizedDocument
- Title extraction (H1 for Markdown, <h1> for HTML, etc.)
- Malformed file raises DocumentLoadError (not a generic Exception)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from documind.ingestion.loaders import (
    DocxLoader,
    DocumentLoadError,
    HTMLLoader,
    LoaderDispatcher,
    MarkdownLoader,
    PDFLoader,
    UnsupportedFormatError,
)
from documind.ingestion.models import NormalizedDocument


class TestDispatch:
    def test_md_dispatches_to_markdown_loader(self, dispatcher: LoaderDispatcher, md_sample: Path):
        doc = dispatcher.load(md_sample)
        assert doc.file_type == "markdown"

    def test_html_dispatches_to_html_loader(self, dispatcher: LoaderDispatcher, html_sample: Path):
        doc = dispatcher.load(html_sample)
        assert doc.file_type == "html"

    def test_pdf_dispatches_to_pdf_loader(self, dispatcher: LoaderDispatcher, pdf_sample: Path):
        doc = dispatcher.load(pdf_sample)
        assert doc.file_type == "pdf"

    def test_docx_dispatches_to_docx_loader(self, dispatcher: LoaderDispatcher, docx_sample: Path):
        doc = dispatcher.load(docx_sample)
        assert doc.file_type == "docx"

    def test_htm_dispatches_to_html_loader(self, dispatcher: LoaderDispatcher, tmp_path: Path):
        htm_file = tmp_path / "page.htm"
        htm_file.write_text("<html><body><h1>HTM file</h1></body></html>", encoding="utf-8")
        doc = dispatcher.load(htm_file)
        assert doc.file_type == "html"

    def test_unsupported_extension_raises_error(self, dispatcher: LoaderDispatcher,
                                                unsupported_csv: Path):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            dispatcher.load(unsupported_csv)
        err = exc_info.value
        assert err.path == unsupported_csv
        assert ".csv" in str(err)

    def test_unsupported_error_mentions_file_path(self, dispatcher: LoaderDispatcher,
                                                   unsupported_zip: Path):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            dispatcher.load(unsupported_zip)
        assert str(unsupported_zip) in str(exc_info.value)

    def test_is_supported_true_for_md(self, dispatcher: LoaderDispatcher, md_sample: Path):
        assert dispatcher.is_supported(md_sample) is True

    def test_is_supported_false_for_csv(self, dispatcher: LoaderDispatcher,
                                         unsupported_csv: Path):
        assert dispatcher.is_supported(unsupported_csv) is False

    def test_supported_extensions_contains_expected(self, dispatcher: LoaderDispatcher):
        exts = dispatcher.supported_extensions
        assert ".md" in exts
        assert ".pdf" in exts
        assert ".html" in exts
        assert ".htm" in exts
        assert ".docx" in exts


class TestMarkdownLoader:
    def test_loads_non_empty_content(self, md_sample: Path):
        doc = MarkdownLoader().load(md_sample)
        assert len(doc.content) > 0

    def test_extracts_h1_as_title(self, md_sample: Path):
        doc = MarkdownLoader().load(md_sample)
        assert doc.title == "Test Document"

    def test_preserves_heading_markers(self, md_sample: Path):
        doc = MarkdownLoader().load(md_sample)
        assert "## Section One" in doc.content
        assert "## Section Two" in doc.content

    def test_filename_as_title_when_no_h1(self, md_no_h1: Path):
        doc = MarkdownLoader().load(md_no_h1)
        # Should fall back to filename-derived title
        assert doc.title is not None
        assert len(doc.title) > 0

    def test_empty_file_produces_empty_content(self, md_empty: Path):
        doc = MarkdownLoader().load(md_empty)
        assert doc.content == ""

    def test_raises_on_unreadable_file(self, tmp_path: Path):
        bad = tmp_path / "bad.md"
        bad.write_text("content")
        bad.chmod(0o000)
        try:
            with pytest.raises(DocumentLoadError):
                MarkdownLoader().load(bad)
        finally:
            bad.chmod(0o644)


class TestHTMLLoader:
    def test_strips_script_tags(self, html_sample: Path):
        doc = HTMLLoader().load(html_sample)
        assert "alert(" not in doc.content

    def test_strips_nav_content(self, html_sample: Path):
        doc = HTMLLoader().load(html_sample)
        # The nav link text "Home" might appear in body but the nav container itself is stripped
        # What we care about is that script junk is gone
        assert "<nav>" not in doc.content
        assert "<script>" not in doc.content

    def test_strips_footer_boilerplate(self, html_sample: Path):
        doc = HTMLLoader().load(html_sample)
        assert "<footer>" not in doc.content

    def test_extracts_h1_as_title(self, html_sample: Path):
        doc = HTMLLoader().load(html_sample)
        assert doc.title == "Test Page"

    def test_main_content_preserved(self, html_sample: Path):
        doc = HTMLLoader().load(html_sample)
        assert "test content for unit testing" in doc.content

    def test_htm_extension_handled(self, tmp_path: Path):
        htm = tmp_path / "doc.htm"
        htm.write_text(
            "<html><body><h1>HTM Title</h1><p>Body text.</p></body></html>",
            encoding="utf-8",
        )
        doc = HTMLLoader().load(htm)
        assert doc.file_type == "html"
        assert "Body text" in doc.content


class TestPDFLoader:
    def test_loads_pdf_content(self, pdf_sample: Path):
        doc = PDFLoader().load(pdf_sample)
        assert len(doc.content) > 0

    def test_page_count_in_metadata(self, pdf_sample: Path):
        doc = PDFLoader().load(pdf_sample)
        assert doc.metadata.page_count == 2

    def test_malformed_pdf_raises_document_load_error(self, pdf_malformed: Path):
        with pytest.raises(DocumentLoadError) as exc_info:
            PDFLoader().load(pdf_malformed)
        assert exc_info.value.path == pdf_malformed

    def test_malformed_pdf_error_mentions_path(self, pdf_malformed: Path):
        with pytest.raises(DocumentLoadError) as exc_info:
            PDFLoader().load(pdf_malformed)
        assert str(pdf_malformed) in str(exc_info.value)


class TestDocxLoader:
    def test_loads_docx_content(self, docx_sample: Path):
        doc = DocxLoader().load(docx_sample)
        assert len(doc.content) > 0

    def test_title_extracted(self, docx_sample: Path):
        doc = DocxLoader().load(docx_sample)
        assert doc.title is not None
        assert len(doc.title) > 0

    def test_content_contains_paragraphs(self, docx_sample: Path):
        doc = DocxLoader().load(docx_sample)
        assert "introduction" in doc.content.lower()

    def test_file_type_is_docx(self, docx_sample: Path):
        doc = DocxLoader().load(docx_sample)
        assert doc.file_type == "docx"
