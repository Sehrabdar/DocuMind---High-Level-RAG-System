"""
Tests for the directory ingestion pipeline.

Covers:
- Mixed corpus ingestion (all formats)
- Stats accuracy (ingested count, format breakdown, character count)
- Unsupported files are skipped and tracked
- Failed documents don't corrupt successful ones
- Empty corpus ingests cleanly
- Ingestion is deterministic (same corpus → same documents on repeat)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from documind.ingestion.pipeline import ingest_directory, IngestionResult


class TestPipelineBasics:
    def test_mixed_corpus_ingests_all_supported_formats(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        file_types = {doc.file_type for doc in result.documents}
        assert "markdown" in file_types
        assert "html" in file_types
        assert "pdf" in file_types
        assert "docx" in file_types

    def test_stats_ingested_count_matches_documents(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert result.stats.total_ingested == len(result.documents)

    def test_stats_failed_count_matches_failed_list(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert result.stats.total_failed == len(result.failed)

    def test_stats_skipped_count_matches_skipped_list(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert result.stats.total_skipped == len(result.skipped)

    def test_unsupported_files_skipped(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert result.stats.total_skipped >= 1
        skipped_names = {p.name for p in result.skipped}
        assert "data.csv" in skipped_names

    def test_character_count_is_positive(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert result.stats.total_characters > 0

    def test_by_format_breakdown_present(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert result.stats.by_format
        # Each format that was ingested should appear
        for doc in result.documents:
            assert doc.file_type in result.stats.by_format

    def test_discovered_equals_supported_plus_unsupported(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert (
            result.stats.total_discovered
            == result.stats.total_supported + result.stats.total_skipped
        )

    def test_empty_corpus_returns_empty_result(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_corpus"
        empty_dir.mkdir()
        result = ingest_directory(empty_dir)
        assert result.documents == []
        assert result.failed == []
        assert result.skipped == []
        assert result.stats.total_ingested == 0

    def test_nonexistent_corpus_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            ingest_directory(tmp_path / "does_not_exist")


class TestPipelineDeterminism:
    def test_same_corpus_produces_same_document_ids(self, mixed_corpus: Path):
        """Running ingestion twice on the same corpus produces identical document IDs."""
        result1 = ingest_directory(mixed_corpus)
        result2 = ingest_directory(mixed_corpus)

        ids1 = sorted(doc.id for doc in result1.documents)
        ids2 = sorted(doc.id for doc in result2.documents)
        assert ids1 == ids2

    def test_same_corpus_produces_same_document_count(self, mixed_corpus: Path):
        result1 = ingest_directory(mixed_corpus)
        result2 = ingest_directory(mixed_corpus)
        assert len(result1.documents) == len(result2.documents)

    def test_document_content_stable_across_runs(self, mixed_corpus: Path):
        """Document content must not change between runs (no random elements)."""
        result1 = ingest_directory(mixed_corpus)
        result2 = ingest_directory(mixed_corpus)

        by_id1 = {doc.id: doc.content for doc in result1.documents}
        by_id2 = {doc.id: doc.content for doc in result2.documents}
        assert by_id1 == by_id2


class TestPipelineIngestionResult:
    def test_result_is_ingestion_result_type(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        assert isinstance(result, IngestionResult)

    def test_all_documents_have_valid_ids(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        for doc in result.documents:
            assert doc.id and len(doc.id) == 64

    def test_all_documents_have_absolute_source_paths(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        for doc in result.documents:
            assert Path(doc.source).is_absolute()

    def test_no_duplicate_document_ids(self, mixed_corpus: Path):
        result = ingest_directory(mixed_corpus)
        ids = [doc.id for doc in result.documents]
        assert len(ids) == len(set(ids)), "Duplicate document IDs found"
