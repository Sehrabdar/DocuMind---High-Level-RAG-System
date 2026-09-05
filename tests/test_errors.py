"""
Tests for error handling in the ingestion system.

Covers:
- Malformed/unreadable files raise DocumentLoadError (not generic Exception)
- Unsupported formats raise UnsupportedFormatError (not generic Exception)
- Errors include the file path
- Empty documents don't crash the system
- Errors in one file don't corrupt other documents
"""

from __future__ import annotations

from pathlib import Path

import pytest

from documind.ingestion.loaders import (
    DocumentLoadError,
    HTMLLoader,
    MarkdownLoader,
    PDFLoader,
    UnsupportedFormatError,
    LoaderDispatcher,
)
from documind.ingestion.pipeline import ingest_directory


class TestUnsupportedFormat:
    def test_unsupported_raises_correct_exception_type(self, unsupported_csv: Path,
                                                        dispatcher: LoaderDispatcher):
        with pytest.raises(UnsupportedFormatError):
            dispatcher.load(unsupported_csv)

    def test_unsupported_error_is_loader_error_subclass(self, unsupported_csv: Path,
                                                          dispatcher: LoaderDispatcher):
        from documind.ingestion.loaders import LoaderError
        with pytest.raises(LoaderError):
            dispatcher.load(unsupported_csv)

    def test_unsupported_error_has_path_attribute(self, unsupported_csv: Path,
                                                   dispatcher: LoaderDispatcher):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            dispatcher.load(unsupported_csv)
        assert exc_info.value.path == unsupported_csv

    def test_unsupported_error_message_mentions_extension(self, unsupported_csv: Path,
                                                           dispatcher: LoaderDispatcher):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            dispatcher.load(unsupported_csv)
        assert ".csv" in str(exc_info.value)


class TestMalformedFiles:
    def test_malformed_pdf_raises_document_load_error(self, pdf_malformed: Path):
        with pytest.raises(DocumentLoadError) as exc_info:
            PDFLoader().load(pdf_malformed)
        assert exc_info.value.path == pdf_malformed

    def test_malformed_pdf_error_mentions_path(self, pdf_malformed: Path):
        with pytest.raises(DocumentLoadError) as exc_info:
            PDFLoader().load(pdf_malformed)
        assert str(pdf_malformed) in str(exc_info.value)

    def test_unreadable_markdown_raises_document_load_error(self, tmp_path: Path):
        locked = tmp_path / "locked.md"
        locked.write_text("# Locked", encoding="utf-8")
        locked.chmod(0o000)
        try:
            with pytest.raises(DocumentLoadError):
                MarkdownLoader().load(locked)
        finally:
            locked.chmod(0o644)

    def test_invalid_html_falls_back_gracefully(self, tmp_path: Path):
        """Completely invalid HTML should not raise an unexpected exception."""
        bad_html = tmp_path / "broken.html"
        bad_html.write_text("<<< not html at all >>>", encoding="utf-8")
        # Should either succeed (BS4 is lenient) or raise DocumentLoadError
        try:
            doc = HTMLLoader().load(bad_html)
            assert isinstance(doc.content, str)
        except DocumentLoadError:
            pass  # Acceptable


class TestEmptyDocuments:
    def test_empty_markdown_produces_empty_content(self, md_empty: Path):
        doc = MarkdownLoader().load(md_empty)
        assert doc.content == ""

    def test_empty_markdown_still_has_valid_metadata(self, md_empty: Path):
        doc = MarkdownLoader().load(md_empty)
        assert doc.id
        assert doc.filename
        assert doc.file_type == "markdown"

    def test_empty_html_handled(self, tmp_path: Path):
        empty_html = tmp_path / "empty.html"
        empty_html.write_text("", encoding="utf-8")
        doc = HTMLLoader().load(empty_html)
        assert isinstance(doc.content, str)


class TestBestEffortIngestion:
    def test_failed_doc_does_not_abort_pipeline(self, tmp_path: Path,
                                                  md_sample: Path, pdf_malformed: Path):
        """A malformed PDF should not prevent the Markdown file from being ingested."""
        import shutil
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        shutil.copy(md_sample, corpus / "good.md")
        shutil.copy(pdf_malformed, corpus / "bad.pdf")

        result = ingest_directory(corpus, fail_fast=False)

        # The good document must be ingested
        assert result.stats.total_ingested >= 1
        ingested_names = {doc.filename for doc in result.documents}
        assert "good.md" in ingested_names

        # The bad document must be recorded as a failure
        assert result.stats.total_failed >= 1
        failed_names = {f.path.name for f in result.failed}
        assert "bad.pdf" in failed_names

    def test_fail_fast_aborts_on_first_error(self, tmp_path: Path,
                                               md_sample: Path, pdf_malformed: Path):
        """With fail_fast=True, the pipeline re-raises on the first failure."""
        import shutil
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        shutil.copy(pdf_malformed, corpus / "bad.pdf")

        with pytest.raises(Exception):
            ingest_directory(corpus, fail_fast=True)

    def test_failure_record_contains_error_type(self, tmp_path: Path, pdf_malformed: Path):
        import shutil
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        shutil.copy(pdf_malformed, corpus / "bad.pdf")

        result = ingest_directory(corpus)
        assert result.failed
        failure = result.failed[0]
        assert failure.error_type
        assert failure.error

    def test_failure_record_contains_path(self, tmp_path: Path, pdf_malformed: Path):
        import shutil
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        shutil.copy(pdf_malformed, corpus / "bad.pdf")

        result = ingest_directory(corpus)
        assert result.failed
        assert result.failed[0].path.name == "bad.pdf"
