"""
Tests for normalization contract.

Every supported format must produce a NormalizedDocument with:
- Non-None id (str)
- Non-None source (str, absolute path)
- Non-None filename (str)
- Non-None file_type (str, one of the known types)
- Non-None content (str — may be empty for genuinely empty files)
- A valid DocumentMetadata object
- metadata.filename == doc.filename
- metadata.file_type == doc.file_type
- metadata.ingested_at is a timezone-aware datetime
"""

from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pytest

from documind.ingestion.loaders import LoaderDispatcher
from documind.ingestion.models import NormalizedDocument

KNOWN_FILE_TYPES = {"markdown", "pdf", "html", "docx"}


@pytest.fixture(
    params=["md_sample", "html_sample", "pdf_sample", "docx_sample"],
)
def any_supported_doc(request, md_sample, html_sample, pdf_sample, docx_sample):
    """Parametrized fixture that yields one NormalizedDocument per format."""
    paths = {
        "md_sample": md_sample,
        "html_sample": html_sample,
        "pdf_sample": pdf_sample,
        "docx_sample": docx_sample,
    }
    dispatcher = LoaderDispatcher()
    return dispatcher.load(paths[request.param])


class TestNormalizationContract:
    """All supported formats must satisfy the same NormalizedDocument interface."""

    def test_id_is_non_empty_string(self, any_supported_doc: NormalizedDocument):
        assert isinstance(any_supported_doc.id, str)
        assert len(any_supported_doc.id) > 0

    def test_source_is_non_empty_string(self, any_supported_doc: NormalizedDocument):
        assert isinstance(any_supported_doc.source, str)
        assert len(any_supported_doc.source) > 0

    def test_source_is_absolute_path(self, any_supported_doc: NormalizedDocument):
        assert Path(any_supported_doc.source).is_absolute()

    def test_filename_is_non_empty_string(self, any_supported_doc: NormalizedDocument):
        assert isinstance(any_supported_doc.filename, str)
        assert len(any_supported_doc.filename) > 0

    def test_file_type_is_known(self, any_supported_doc: NormalizedDocument):
        assert any_supported_doc.file_type in KNOWN_FILE_TYPES

    def test_content_is_string(self, any_supported_doc: NormalizedDocument):
        assert isinstance(any_supported_doc.content, str)

    def test_metadata_present(self, any_supported_doc: NormalizedDocument):
        assert any_supported_doc.metadata is not None

    def test_metadata_filename_matches_doc(self, any_supported_doc: NormalizedDocument):
        assert any_supported_doc.metadata.filename == any_supported_doc.filename

    def test_metadata_file_type_matches_doc(self, any_supported_doc: NormalizedDocument):
        assert any_supported_doc.metadata.file_type == any_supported_doc.file_type

    def test_metadata_ingested_at_is_timezone_aware(self, any_supported_doc: NormalizedDocument):
        ts = any_supported_doc.metadata.ingested_at
        assert ts.tzinfo is not None
        assert ts.tzinfo == timezone.utc or ts.utcoffset() is not None

    def test_document_is_serializable_to_dict(self, any_supported_doc: NormalizedDocument):
        data = any_supported_doc.model_dump()
        assert "id" in data
        assert "content" in data
        assert "metadata" in data

    def test_document_is_json_serializable(self, any_supported_doc: NormalizedDocument):
        json_str = any_supported_doc.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_document_source_matches_metadata_source(self, any_supported_doc: NormalizedDocument):
        assert any_supported_doc.source == any_supported_doc.metadata.source
