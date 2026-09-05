"""
Tests for file discovery logic.

Covers:
- Recursive traversal
- Supported vs unsupported file partitioning
- Non-existent directory raises FileNotFoundError
- Single-file directory (edge case)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from documind.ingestion.pipeline import discover_files


class TestDiscoverFiles:
    def test_discovers_supported_files_recursively(self, fixtures_dir: Path):
        """Supported files in nested subdirectories are all discovered."""
        supported, _ = discover_files(fixtures_dir)
        extensions = {p.suffix.lower() for p in supported}
        # We expect to find at least .md, .html, .pdf, .docx
        assert ".md" in extensions
        assert ".html" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions

    def test_unsupported_files_are_partitioned_not_silently_ignored(self, fixtures_dir: Path):
        """Unsupported files appear in the unsupported list, never the supported list."""
        supported, unsupported = discover_files(fixtures_dir)
        supported_names = {p.name for p in supported}
        unsupported_names = {p.name for p in unsupported}

        assert "data.csv" in unsupported_names
        assert "data.csv" not in supported_names

    def test_returns_sorted_deterministic_order(self, fixtures_dir: Path):
        """Running discover_files twice produces the same order."""
        run1_supported, run1_unsupported = discover_files(fixtures_dir)
        run2_supported, run2_unsupported = discover_files(fixtures_dir)
        assert run1_supported == run2_supported
        assert run1_unsupported == run2_unsupported

    def test_raises_for_nonexistent_directory(self, tmp_path: Path):
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            discover_files(nonexistent)

    def test_raises_for_file_instead_of_directory(self, md_sample: Path):
        with pytest.raises(NotADirectoryError):
            discover_files(md_sample)

    def test_empty_directory_returns_empty_lists(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        supported, unsupported = discover_files(empty_dir)
        assert supported == []
        assert unsupported == []

    def test_mixed_corpus_counts(self, mixed_corpus: Path):
        """Mixed corpus: exactly 4 supported, 1 unsupported."""
        supported, unsupported = discover_files(mixed_corpus)
        assert len(supported) == 4
        assert len(unsupported) == 1

    def test_only_files_are_returned_not_directories(self, fixtures_dir: Path):
        """Directories should never appear in the returned lists."""
        supported, unsupported = discover_files(fixtures_dir)
        for p in supported + unsupported:
            assert p.is_file(), f"Non-file found in result: {p}"
