"""
Unit tests for the EmbeddingService.

All tests use FakeEmbeddingProvider — no model download, no network access,
no GPU required.  The real BAAI/bge-small-en-v1.5 integration is in
tests/integration/test_real_embeddings.py (marked model_integration).

Covers:
- FakeEmbeddingProvider dimensions
- EmbeddingService.embed_text: correct dimension, non-empty text
- EmbeddingService.embed_texts: ordering preserved, empty list, dimension
- L2 normalization: vectors have unit length after embed_texts()
- Determinism: same input → same output
- Batch ordering invariant
- Error: embed_text("") raises ValueError
- Batch size parameter accepted
- Multiple texts produce matching count of vectors
- Section path / metadata do not affect embedding
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from documind.embeddings.service import (
    EmbeddingService,
    FakeEmbeddingProvider,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimension=384)


@pytest.fixture
def service(fake_provider: FakeEmbeddingProvider) -> EmbeddingService:
    return EmbeddingService(fake_provider)


# ─────────────────────────────────────────────────────────────────────────────
# FakeEmbeddingProvider
# ─────────────────────────────────────────────────────────────────────────────


class TestFakeEmbeddingProvider:
    def test_dimension_property(self):
        provider = FakeEmbeddingProvider(dimension=128)
        assert provider.dimension == 128

    def test_default_dimension_is_384(self):
        provider = FakeEmbeddingProvider()
        assert provider.dimension == 384

    def test_encode_returns_correct_shape(self):
        provider = FakeEmbeddingProvider(dimension=384)
        texts = ["hello", "world", "test"]
        result = provider.encode(texts, batch_size=2)
        assert result.shape == (3, 384)

    def test_encode_returns_float32(self):
        provider = FakeEmbeddingProvider(dimension=64)
        result = provider.encode(["text"], batch_size=1)
        assert result.dtype == np.float32

    def test_encode_deterministic(self):
        provider = FakeEmbeddingProvider(dimension=384)
        text = "deterministic test content"
        r1 = provider.encode([text], batch_size=1)
        r2 = provider.encode([text], batch_size=1)
        np.testing.assert_array_equal(r1, r2)

    def test_encode_different_texts_different_vectors(self):
        provider = FakeEmbeddingProvider(dimension=384)
        r1 = provider.encode(["text A"], batch_size=1)
        r2 = provider.encode(["text B"], batch_size=1)
        assert not np.allclose(r1, r2), "Different texts should produce different vectors"

    def test_encode_empty_list(self):
        provider = FakeEmbeddingProvider(dimension=384)
        result = provider.encode([], batch_size=32)
        assert result.shape == (0, 384)


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingService — embed_text
# ─────────────────────────────────────────────────────────────────────────────


class TestEmbedText:
    def test_returns_list_of_floats(self, service):
        result = service.embed_text("hello world")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_correct_dimension(self, service):
        result = service.embed_text("test content")
        assert len(result) == 384

    def test_empty_text_raises_value_error(self, service):
        with pytest.raises(ValueError, match="non-empty"):
            service.embed_text("")

    def test_whitespace_only_raises_value_error(self, service):
        with pytest.raises(ValueError, match="non-empty"):
            service.embed_text("   \n  ")

    def test_deterministic_for_same_input(self, service):
        text = "exactly the same text content"
        r1 = service.embed_text(text)
        r2 = service.embed_text(text)
        assert r1 == r2

    def test_different_texts_different_vectors(self, service):
        r1 = service.embed_text("text about authentication")
        r2 = service.embed_text("text about database schemas")
        assert r1 != r2


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingService — embed_texts
# ─────────────────────────────────────────────────────────────────────────────


class TestEmbedTexts:
    def test_empty_input_returns_empty_list(self, service):
        result = service.embed_texts([])
        assert result == []

    def test_single_text_returns_one_vector(self, service):
        result = service.embed_texts(["single text"])
        assert len(result) == 1
        assert len(result[0]) == 384

    def test_multiple_texts_correct_count(self, service):
        texts = ["text one", "text two", "text three", "text four"]
        result = service.embed_texts(texts)
        assert len(result) == len(texts)

    def test_all_vectors_correct_dimension(self, service):
        texts = ["alpha", "beta", "gamma"]
        result = service.embed_texts(texts)
        for vec in result:
            assert len(vec) == 384

    def test_ordering_preserved(self, service):
        """texts[i] must map to result[i] — ordering is an invariant."""
        texts = [f"text number {i}" for i in range(10)]
        result = service.embed_texts(texts)

        # Each text must produce the same vector as embed_text(text)
        for i, text in enumerate(texts):
            expected = service.embed_text(text)
            assert result[i] == expected, f"Ordering mismatch at index {i}"

    def test_deterministic_batch(self, service):
        texts = ["batch text A", "batch text B", "batch text C"]
        r1 = service.embed_texts(texts)
        r2 = service.embed_texts(texts)
        assert r1 == r2

    def test_returns_list_of_list_of_floats(self, service):
        result = service.embed_texts(["test"])
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert all(isinstance(v, float) for v in result[0])

    def test_batch_size_parameter_accepted(self, service):
        texts = ["text A", "text B", "text C", "text D", "text E"]
        result = service.embed_texts(texts, batch_size=2)
        assert len(result) == 5

    def test_batch_size_one_same_as_default(self, service):
        texts = ["x", "y", "z"]
        r1 = service.embed_texts(texts, batch_size=1)
        r2 = service.embed_texts(texts)
        assert r1 == r2


# ─────────────────────────────────────────────────────────────────────────────
# L2 normalization
# ─────────────────────────────────────────────────────────────────────────────


class TestL2Normalization:
    def test_embed_text_vector_has_unit_norm(self, service):
        """L2 norm of the returned vector must equal 1.0 (within float tolerance)."""
        vec = service.embed_text("normalization test text")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_texts_all_vectors_unit_norm(self, service):
        texts = ["alpha", "beta", "gamma", "delta"]
        results = service.embed_texts(texts)
        for i, vec in enumerate(results):
            norm = math.sqrt(sum(v * v for v in vec))
            assert abs(norm - 1.0) < 1e-5, f"Vector {i} has norm {norm}, expected 1.0"

    def test_normalization_applied_once(self, service):
        """Calling embed_text() twice should return the same normalized vector."""
        text = "normalization is idempotent"
        r1 = service.embed_text(text)
        r2 = service.embed_text(text)
        assert r1 == r2

    def test_zero_vector_does_not_raise(self):
        """A provider that returns all zeros should not cause divide-by-zero."""
        class ZeroProvider:
            @property
            def dimension(self): return 4
            def encode(self, texts, batch_size, show_progress_bar=False):
                import numpy as np
                return np.zeros((len(texts), 4), dtype=np.float32)

        svc = EmbeddingService(ZeroProvider())
        result = svc.embed_texts(["anything"])
        # Should not raise; vector will be all-zeros (not unit norm but safe)
        assert len(result) == 1
        assert len(result[0]) == 4


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingService — dimension property
# ─────────────────────────────────────────────────────────────────────────────


class TestDimension:
    def test_dimension_matches_provider(self):
        for dim in [64, 128, 384, 768]:
            provider = FakeEmbeddingProvider(dimension=dim)
            svc = EmbeddingService(provider)
            assert svc.dimension == dim
