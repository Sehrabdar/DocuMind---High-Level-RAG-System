"""
Tests for deterministic document ID generation.

The ingestion contract requires that:
- Same source path → same document ID (across multiple calls)
- Different source paths → different document IDs
- ID is a non-empty hex string (SHA-256 → 64 chars)
- Process execution order does not affect IDs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from documind.ingestion.loaders import LoaderDispatcher, _document_id


class TestDeterministicIds:
    def test_same_path_produces_same_id(self, md_sample: Path):
        """Calling _document_id twice with the same path returns identical results."""
        id1 = _document_id(md_sample)
        id2 = _document_id(md_sample)
        assert id1 == id2

    def test_different_paths_produce_different_ids(self, md_sample: Path,
                                                    html_sample: Path):
        assert _document_id(md_sample) != _document_id(html_sample)

    def test_id_is_64_char_hex(self, md_sample: Path):
        """SHA-256 hex digest is always 64 lowercase hex characters."""
        doc_id = _document_id(md_sample)
        assert len(doc_id) == 64
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_loader_returns_deterministic_id(self, md_sample: Path):
        """Calling the loader twice on the same file returns the same document ID."""
        dispatcher = LoaderDispatcher()
        doc1 = dispatcher.load(md_sample)
        doc2 = dispatcher.load(md_sample)
        assert doc1.id == doc2.id

    def test_all_formats_have_deterministic_ids(self, md_sample, html_sample,
                                                 pdf_sample, docx_sample):
        """Each format loader returns a stable ID on repeated calls."""
        dispatcher = LoaderDispatcher()
        for path in [md_sample, html_sample, pdf_sample, docx_sample]:
            id1 = dispatcher.load(path).id
            id2 = dispatcher.load(path).id
            assert id1 == id2, f"Non-deterministic ID for {path}"

    def test_copy_of_file_produces_different_id(self, md_sample: Path, tmp_path: Path):
        """A copy at a different path gets a different ID (path-based, not content-based)."""
        import shutil
        copy = tmp_path / "copy.md"
        shutil.copy(md_sample, copy)
        assert _document_id(md_sample) != _document_id(copy)

    def test_id_does_not_depend_on_call_order(self, md_sample: Path, html_sample: Path):
        """IDs computed in different call orders must match."""
        dispatcher = LoaderDispatcher()

        # Load in order A → B
        id_md_first = dispatcher.load(md_sample).id
        id_html_first = dispatcher.load(html_sample).id

        # Load in order B → A
        id_html_second = dispatcher.load(html_sample).id
        id_md_second = dispatcher.load(md_sample).id

        assert id_md_first == id_md_second
        assert id_html_first == id_html_second
