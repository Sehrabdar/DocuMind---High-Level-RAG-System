"""
Tests for the NormalizedChunk model.

Covers:
- Required fields are present and correctly typed
- token_count is a positive integer
- section_path is a list of strings
- page fields are mutually consistent
- Model is immutable (frozen=True)
- JSON round-trip serialization
- Empty section_path is valid
- document_title can be None
"""

from __future__ import annotations

import json

import pytest

from documind.ingestion.chunking import NormalizedChunk, _chunk_id


def _make_chunk(**overrides) -> NormalizedChunk:
    """Build a minimal valid NormalizedChunk for testing."""
    defaults = dict(
        chunk_id="a" * 64,
        document_id="b" * 64,
        chunk_index=0,
        content="This is some chunk content.",
        token_count=6,
        section_path=["Authentication", "OAuth"],
        document_title="API Reference",
        start_char=0,
        end_char=27,
        page=None,
        page_start=None,
        page_end=None,
    )
    defaults.update(overrides)
    return NormalizedChunk(**defaults)


class TestNormalizedChunkModel:
    def test_can_construct_minimal_chunk(self):
        chunk = _make_chunk()
        assert chunk.chunk_index == 0
        assert chunk.content == "This is some chunk content."

    def test_chunk_id_is_str(self):
        chunk = _make_chunk()
        assert isinstance(chunk.chunk_id, str)
        assert len(chunk.chunk_id) > 0

    def test_document_id_is_str(self):
        chunk = _make_chunk()
        assert isinstance(chunk.document_id, str)

    def test_chunk_index_is_int(self):
        chunk = _make_chunk()
        assert isinstance(chunk.chunk_index, int)
        assert chunk.chunk_index >= 0

    def test_content_is_str(self):
        chunk = _make_chunk()
        assert isinstance(chunk.content, str)
        assert len(chunk.content) > 0

    def test_token_count_is_positive_int(self):
        chunk = _make_chunk(token_count=42)
        assert isinstance(chunk.token_count, int)
        assert chunk.token_count > 0

    def test_section_path_is_list_of_str(self):
        chunk = _make_chunk(section_path=["A", "B", "C"])
        assert isinstance(chunk.section_path, list)
        assert all(isinstance(s, str) for s in chunk.section_path)

    def test_empty_section_path_is_valid(self):
        chunk = _make_chunk(section_path=[])
        assert chunk.section_path == []

    def test_document_title_can_be_none(self):
        chunk = _make_chunk(document_title=None)
        assert chunk.document_title is None

    def test_document_title_can_be_string(self):
        chunk = _make_chunk(document_title="My Document")
        assert chunk.document_title == "My Document"

    def test_start_char_is_non_negative(self):
        chunk = _make_chunk(start_char=100, end_char=200)
        assert chunk.start_char >= 0

    def test_end_char_is_gte_start_char(self):
        chunk = _make_chunk(start_char=50, end_char=150)
        assert chunk.end_char >= chunk.start_char

    def test_page_fields_default_to_none(self):
        chunk = _make_chunk()
        assert chunk.page is None
        assert chunk.page_start is None
        assert chunk.page_end is None

    def test_single_page_can_be_set(self):
        chunk = _make_chunk(page=3)
        assert chunk.page == 3

    def test_page_range_can_be_set(self):
        chunk = _make_chunk(page_start=2, page_end=3)
        assert chunk.page_start == 2
        assert chunk.page_end == 3


class TestNormalizedChunkImmutability:
    def test_chunk_is_frozen(self):
        chunk = _make_chunk()
        with pytest.raises(Exception):
            chunk.content = "modified"  # type: ignore[misc]

    def test_chunk_index_immutable(self):
        chunk = _make_chunk()
        with pytest.raises(Exception):
            chunk.chunk_index = 99  # type: ignore[misc]


class TestNormalizedChunkSerialization:
    def test_model_dump_returns_dict(self):
        chunk = _make_chunk()
        data = chunk.model_dump()
        assert isinstance(data, dict)

    def test_model_dump_contains_required_keys(self):
        chunk = _make_chunk()
        data = chunk.model_dump()
        for key in [
            "chunk_id", "document_id", "chunk_index", "content",
            "token_count", "section_path", "document_title",
            "start_char", "end_char", "page", "page_start", "page_end",
        ]:
            assert key in data, f"Missing key: {key}"

    def test_model_dump_json_produces_valid_json(self):
        chunk = _make_chunk()
        json_str = chunk.model_dump_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["chunk_index"] == 0
        assert parsed["content"] == "This is some chunk content."

    def test_json_round_trip_preserves_section_path(self):
        path = ["Authentication", "OAuth", "Tokens"]
        chunk = _make_chunk(section_path=path)
        parsed = json.loads(chunk.model_dump_json())
        assert parsed["section_path"] == path

    def test_json_round_trip_preserves_none_fields(self):
        chunk = _make_chunk(page=None, document_title=None)
        parsed = json.loads(chunk.model_dump_json())
        assert parsed["page"] is None
        assert parsed["document_title"] is None
