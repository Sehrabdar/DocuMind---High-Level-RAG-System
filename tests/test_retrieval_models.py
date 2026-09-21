"""
Unit tests for documind/retrieval/models.py — RetrievedChunk.

Covers:
- Construction via RetrievedChunk.from_row()
- Distance/similarity semantics (distance = 1 - similarity)
- Score ranges
- Rank validation (must be >= 1)
- format_summary() output
- JSON serialization
- All provenance fields are preserved
"""

from __future__ import annotations

import json
import math

import pytest

from documind.retrieval.models import RetrievedChunk


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_chunk(**overrides) -> RetrievedChunk:
    """Return a RetrievedChunk with sensible defaults."""
    defaults = dict(
        chunk_id="a" * 64,
        document_id="b" * 64,
        content="The API key can be revoked from the security dashboard.",
        token_count=11,
        chunk_index=0,
        section_path=["API Keys", "Revocation"],
        document_title="API Reference",
        page=None,
        page_start=None,
        page_end=None,
        start_char=0,
        end_char=55,
        distance=0.25,
        rank=1,
    )
    defaults.update(overrides)
    return RetrievedChunk.from_row(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievedChunkConstruction:

    def test_from_row_creates_instance(self):
        chunk = _make_chunk()
        assert isinstance(chunk, RetrievedChunk)

    def test_identity_fields_preserved(self):
        chunk = _make_chunk(chunk_id="c" * 64, document_id="d" * 64)
        assert chunk.chunk_id == "c" * 64
        assert chunk.document_id == "d" * 64

    def test_content_fields_preserved(self):
        chunk = _make_chunk(content="hello world", token_count=2)
        assert chunk.content == "hello world"
        assert chunk.token_count == 2

    def test_structural_provenance_preserved(self):
        path = ["Auth", "OAuth", "Token Refresh"]
        chunk = _make_chunk(
            chunk_index=3,
            section_path=path,
            document_title="Security Guide",
        )
        assert chunk.chunk_index == 3
        assert chunk.section_path == path
        assert chunk.document_title == "Security Guide"

    def test_location_provenance_preserved(self):
        chunk = _make_chunk(page=5, page_start=4, page_end=6, start_char=100, end_char=500)
        assert chunk.page == 5
        assert chunk.page_start == 4
        assert chunk.page_end == 6
        assert chunk.start_char == 100
        assert chunk.end_char == 500

    def test_none_fields_allowed(self):
        chunk = _make_chunk(page=None, page_start=None, page_end=None, document_title=None)
        assert chunk.page is None
        assert chunk.page_start is None
        assert chunk.page_end is None
        assert chunk.document_title is None

    def test_empty_section_path_allowed(self):
        chunk = _make_chunk(section_path=[])
        assert chunk.section_path == []


# ─────────────────────────────────────────────────────────────────────────────
# Score semantics
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievedChunkScoreSemantics:

    def test_similarity_equals_one_minus_distance(self):
        chunk = _make_chunk(distance=0.25)
        assert abs(chunk.similarity - (1.0 - 0.25)) < 1e-9

    def test_similarity_one_when_distance_zero(self):
        """distance=0 means identical vectors → similarity=1."""
        chunk = _make_chunk(distance=0.0)
        assert abs(chunk.similarity - 1.0) < 1e-9

    def test_similarity_zero_when_distance_one(self):
        """distance=1 means orthogonal vectors → similarity=0."""
        chunk = _make_chunk(distance=1.0)
        assert abs(chunk.similarity - 0.0) < 1e-9

    def test_similarity_minus_one_when_distance_two(self):
        """distance=2 means opposite vectors → similarity=-1."""
        chunk = _make_chunk(distance=2.0)
        assert abs(chunk.similarity - (-1.0)) < 1e-9

    def test_distance_is_stored_exactly(self):
        chunk = _make_chunk(distance=0.12345)
        assert abs(chunk.distance - 0.12345) < 1e-9

    def test_distance_zero(self):
        chunk = _make_chunk(distance=0.0)
        assert chunk.distance == 0.0

    def test_distance_near_two(self):
        chunk = _make_chunk(distance=1.9999)
        assert chunk.distance == 1.9999

    def test_multiple_chunks_rank_order_implies_distance_order(self):
        """Lower rank must correspond to lower distance when constructed correctly."""
        chunks = [
            _make_chunk(distance=0.1, rank=1),
            _make_chunk(distance=0.3, rank=2),
            _make_chunk(distance=0.7, rank=3),
        ]
        # Verify assumption: rank 1 has smallest distance
        distances = [c.distance for c in chunks]
        assert distances == sorted(distances)

    def test_similarity_derived_consistently_for_various_distances(self):
        for dist in [0.0, 0.1, 0.5, 1.0, 1.5, 2.0]:
            chunk = _make_chunk(distance=dist)
            expected_sim = 1.0 - dist
            assert abs(chunk.similarity - expected_sim) < 1e-9, (
                f"distance={dist}: expected similarity={expected_sim}, got {chunk.similarity}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Rank validation
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievedChunkRankValidation:

    def test_rank_one_is_valid(self):
        chunk = _make_chunk(rank=1)
        assert chunk.rank == 1

    def test_rank_large_is_valid(self):
        chunk = _make_chunk(rank=100)
        assert chunk.rank == 100

    def test_rank_zero_raises(self):
        with pytest.raises(Exception):
            _make_chunk(rank=0)

    def test_rank_negative_raises(self):
        with pytest.raises(Exception):
            _make_chunk(rank=-1)


# ─────────────────────────────────────────────────────────────────────────────
# format_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievedChunkFormatSummary:

    def test_format_summary_contains_rank(self):
        chunk = _make_chunk(rank=3)
        assert "rank=3" in chunk.format_summary()

    def test_format_summary_contains_distance(self):
        chunk = _make_chunk(distance=0.1818)
        summary = chunk.format_summary()
        assert "dist=0.1818" in summary

    def test_format_summary_contains_similarity(self):
        chunk = _make_chunk(distance=0.2500)
        summary = chunk.format_summary()
        assert "sim=0.7500" in summary

    def test_format_summary_with_section_path(self):
        chunk = _make_chunk(section_path=["Auth", "OAuth"])
        summary = chunk.format_summary()
        assert "Auth" in summary
        assert "OAuth" in summary

    def test_format_summary_without_section_path(self):
        chunk = _make_chunk(section_path=[])
        summary = chunk.format_summary()
        assert "no section" in summary

    def test_format_summary_with_page(self):
        chunk = _make_chunk(page=7)
        assert "page=7" in chunk.format_summary()

    def test_format_summary_without_page(self):
        chunk = _make_chunk(page=None)
        assert "page=—" in chunk.format_summary()

    def test_format_summary_is_string(self):
        summary = _make_chunk().format_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievedChunkSerialization:

    def test_model_dump_returns_dict(self):
        chunk = _make_chunk()
        d = chunk.model_dump()
        assert isinstance(d, dict)

    def test_model_dump_json_serializable(self):
        chunk = _make_chunk()
        json_str = chunk.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["rank"] == chunk.rank
        assert abs(parsed["distance"] - chunk.distance) < 1e-9
        assert abs(parsed["similarity"] - chunk.similarity) < 1e-9

    def test_all_required_fields_present(self):
        chunk = _make_chunk()
        d = chunk.model_dump()
        required = {
            "chunk_id", "document_id", "content", "token_count",
            "chunk_index", "section_path", "document_title",
            "page", "page_start", "page_end", "start_char", "end_char",
            "distance", "similarity", "rank",
        }
        assert required.issubset(d.keys())

    def test_round_trip_preserves_values(self):
        original = _make_chunk(distance=0.333, rank=2)
        restored = RetrievedChunk(**original.model_dump())
        assert restored.chunk_id == original.chunk_id
        assert abs(restored.distance - original.distance) < 1e-9
        assert abs(restored.similarity - original.similarity) < 1e-9
        assert restored.rank == original.rank
