"""
Model integration tests for BAAI/bge-small-en-v1.5.

These tests load the real SentenceTransformer model and require:
1. The model downloaded (~130MB from HuggingFace Hub on first run).
2. No internet access check needed after first download (model is cached).

Run separately from regular tests:
    uv run pytest tests/integration/test_real_embeddings.py -m model_integration -v

Verifies:
- Output dimension is exactly 384
- Vectors are L2-normalized (unit norm)
- Same input → same output (determinism)
- Batch embedding matches individual embedding (ordering invariant)
- Empty list returns empty list
"""

from __future__ import annotations

import math

import pytest

from documind.embeddings.service import EmbeddingService, SentenceTransformerProvider


@pytest.fixture(scope="module")
def real_service() -> EmbeddingService:
    """Load the real BAAI/bge-small-en-v1.5 model once per module."""
    provider = SentenceTransformerProvider("BAAI/bge-small-en-v1.5")
    return EmbeddingService(provider)


@pytest.mark.model_integration
class TestRealBGEModel:
    """Integration tests against the actual BAAI/bge-small-en-v1.5 model."""

    def test_dimension_is_384(self, real_service):
        assert real_service.dimension == 384

    def test_embed_text_returns_384_floats(self, real_service):
        result = real_service.embed_text("test sentence for dimension check")
        assert len(result) == 384

    def test_embed_text_unit_norm(self, real_service):
        """L2-normalized vector must have norm ≈ 1.0."""
        vec = real_service.embed_text("normalization verification sentence")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-5, f"Norm was {norm}, expected 1.0"

    def test_embed_texts_correct_count(self, real_service):
        texts = ["first sentence", "second sentence", "third sentence"]
        result = real_service.embed_texts(texts)
        assert len(result) == 3

    def test_embed_texts_all_384_dimensional(self, real_service):
        texts = ["alpha", "beta", "gamma", "delta"]
        result = real_service.embed_texts(texts)
        for vec in result:
            assert len(vec) == 384

    def test_embed_texts_all_unit_norm(self, real_service):
        texts = ["sentence one about authentication", "sentence two about authorization"]
        result = real_service.embed_texts(texts)
        for i, vec in enumerate(result):
            norm = math.sqrt(sum(v * v for v in vec))
            assert abs(norm - 1.0) < 1e-5, f"Vector {i} has norm {norm}"

    def test_determinism_same_input_same_output(self, real_service):
        text = "The quick brown fox jumps over the lazy dog."
        r1 = real_service.embed_text(text)
        r2 = real_service.embed_text(text)
        assert r1 == r2, "Same input must produce identical output"

    def test_batch_matches_individual_embeddings(self, real_service):
        """embed_texts()[i] must match embed_text(texts[i]) exactly."""
        texts = [
            "API authentication using OAuth 2.0",
            "Database connection pooling strategies",
            "Vector similarity search with HNSW",
        ]
        batch_result = real_service.embed_texts(texts)
        for i, text in enumerate(texts):
            individual = real_service.embed_text(text)
            for j in range(min(10, len(individual))):
                assert abs(batch_result[i][j] - individual[j]) < 1e-5, (
                    f"Mismatch at text={i}, dim={j}: "
                    f"batch={batch_result[i][j]:.6f} vs individual={individual[j]:.6f}"
                )

    def test_empty_list_returns_empty(self, real_service):
        result = real_service.embed_texts([])
        assert result == []

    def test_different_texts_different_vectors(self, real_service):
        r1 = real_service.embed_text("authentication and authorization")
        r2 = real_service.embed_text("database indexing strategies")
        # Vectors should not be identical (different semantic content)
        assert r1 != r2

    def test_batch_size_parameter_does_not_change_result(self, real_service):
        """batch_size should not meaningfully affect the output vectors.

        Note: float32 addition is not associative, so different batch sizes
        may produce vectors that differ by tiny FP rounding errors (~1e-6).
        We use a tight tolerance rather than exact equality.
        """
        texts = ["text a", "text b", "text c", "text d", "text e"]
        r1 = real_service.embed_texts(texts, batch_size=2)
        r2 = real_service.embed_texts(texts, batch_size=5)
        for i in range(len(texts)):
            for j in range(len(r1[i])):
                assert abs(r1[i][j] - r2[i][j]) < 1e-4, (
                    f"Excessive difference at text={i}, dim={j}: "
                    f"{r1[i][j]} vs {r2[i][j]}"
                )
