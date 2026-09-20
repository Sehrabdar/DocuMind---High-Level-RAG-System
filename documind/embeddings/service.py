"""
Embedding service for DocuMind Phase 3.

Architecture
------------
The service is built around an ``EmbeddingProvider`` protocol, which allows
the production ``SentenceTransformerProvider`` to be swapped for a
``FakeEmbeddingProvider`` in unit tests — without downloading a model or
reaching the network.

The boundary is:

    EmbeddingService          ← public API (dimension, batching, normalization)
         ↓
    EmbeddingProvider         ← protocol (testable seam)
         ↓
    SentenceTransformerProvider  ← wraps the real BAAI/bge-small-en-v1.5 model

Design decisions
----------------
- **Normalization**: BGE models are trained with cosine similarity as the
  metric.  After L2 normalization, cosine similarity equals dot product, which
  pgvector supports efficiently in the HNSW index with ``vector_cosine_ops``.
  Normalization is applied exactly once in ``EmbeddingService.embed_texts()``,
  never in the persistence or retrieval layers.

- **Single model load**: ``SentenceTransformerProvider`` loads the model once
  at construction time.  Do not call ``SentenceTransformer(...)`` per chunk
  or per batch — the loading overhead would make ingestion unusably slow.

- **Ordered output**: The mapping ``chunk[i] → embedding[i]`` is an invariant
  that must be preserved.  This implementation uses numpy slice indexing to
  guarantee it, and tests verify it explicitly.

- **Empty input**: ``embed_texts([])`` returns ``[]`` without calling the
  model.  This is a documented contract, not an implicit behavior.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Provider protocol — the testable seam
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for an object that can encode a list of strings into vectors.

    Both ``SentenceTransformerProvider`` (production) and
    ``FakeEmbeddingProvider`` (testing) implement this protocol.  Callers
    depend on the protocol, not the concrete implementation.
    """

    @property
    def dimension(self) -> int:
        """Output vector dimension of this provider."""
        ...

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode *texts* into a 2-D float32 array of shape (len(texts), dim).

        Parameters
        ----------
        texts:
            Non-empty list of strings to encode.  The caller guarantees
            non-empty input; providers may assert this.
        batch_size:
            Number of texts per encoding batch.  Providers may ignore this
            if they do not support batching internally.
        show_progress_bar:
            Whether to display a progress indicator.  Defaults to False so
            tests and library usage do not pollute stdout.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(len(texts), self.dimension)``.
            Output ordering matches input ordering exactly.
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Production provider: SentenceTransformer
# ─────────────────────────────────────────────────────────────────────────────


class SentenceTransformerProvider:
    """Wraps ``sentence_transformers.SentenceTransformer`` as an EmbeddingProvider.

    The model is loaded once at construction time and reused for all encode()
    calls.  This is the production implementation used in the ingestion pipeline.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model name (e.g. ``"BAAI/bge-small-en-v1.5"``) or a
        local path to a saved SentenceTransformer model.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        ``None`` lets sentence-transformers choose automatically.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
    ) -> None:
        # Deferred import so that sentence-transformers is only loaded when
        # this class is instantiated (not at module import time).
        from sentence_transformers import SentenceTransformer  # type: ignore[import]

        logger.info("Loading embedding model: %s", model_name_or_path)
        self._model = SentenceTransformer(model_name_or_path, device=device)
        self._dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded. dimension=%d", self._dimension
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        result = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,  # normalization done in EmbeddingService
        )
        return np.asarray(result, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Fake provider: deterministic, no model download — for unit tests
# ─────────────────────────────────────────────────────────────────────────────


class FakeEmbeddingProvider:
    """Deterministic stub provider for unit tests.

    Returns vectors whose values are derived from the input text hash so that:
    - Different texts produce different vectors.
    - Same text always produces the same vector.
    - No model download or network access is required.
    - The output shape is always ``(len(texts), dimension)``.

    This is the test double for ``SentenceTransformerProvider``.  It must
    never be used in the production embedding pipeline.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Return deterministic pseudo-random vectors based on text content."""
        result = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for i, text in enumerate(texts):
            # Seed from the text content hash for determinism
            seed = hash(text) % (2**31)
            rng = np.random.default_rng(seed)
            result[i] = rng.standard_normal(self._dimension).astype(np.float32)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingService — the public API
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingService:
    """Encodes text into L2-normalized vectors using a configured provider.

    Responsibilities
    ----------------
    1. Accept text(s) from callers.
    2. Delegate raw encoding to the provider (which handles batching).
    3. L2-normalize the output vectors exactly once.
    4. Return plain Python ``list[float]`` for JSON-serializable output.

    The service does NOT:
    - Load or manage the embedding model (that is the provider's job).
    - Write to the database (that is the repository's job).
    - Perform any retrieval (that is Phase 4+).

    Parameters
    ----------
    provider:
        Any object implementing the ``EmbeddingProvider`` protocol.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    @property
    def dimension(self) -> int:
        """Embedding output dimension from the underlying provider."""
        return self._provider.dimension

    def embed_text(self, text: str, batch_size: int = 64) -> list[float]:
        """Embed a single text string.

        Parameters
        ----------
        text:
            The text to embed.  Must not be empty.
        batch_size:
            Batch size passed to the provider.  Ignored for single-text calls
            but kept for a consistent call signature.

        Returns
        -------
        list[float]
            L2-normalized vector of length ``self.dimension``.

        Raises
        ------
        ValueError
            If ``text`` is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("embed_text() requires non-empty text")
        vectors = self.embed_texts([text], batch_size=batch_size)
        return vectors[0]

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Embed a batch of texts.

        Parameters
        ----------
        texts:
            List of texts to embed.  Empty list returns ``[]`` immediately.
        batch_size:
            Number of texts per encoding batch forwarded to the provider.
        show_progress_bar:
            Display progress during encoding.  Useful for large corpora;
            disabled by default to keep output clean in library use.

        Returns
        -------
        list[list[float]]
            List of L2-normalized vectors, one per input text.  Ordering
            matches the input ordering exactly: ``texts[i]`` → ``result[i]``.

        Notes
        -----
        Empty texts are replaced with a single space before encoding to avoid
        provider errors.  This is documented behavior, not silent mutation.
        Callers should avoid passing empty strings; the chunking pipeline
        guarantees non-empty content, so this is a defensive measure.
        """
        if not texts:
            return []

        # Defensive: replace empty strings to avoid provider errors
        sanitized = [t if t.strip() else " " for t in texts]

        raw: np.ndarray = self._provider.encode(
            sanitized,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        # L2 normalization — applied exactly once, here.
        # After normalization, cosine similarity ≡ dot product, which is
        # the metric used by pgvector HNSW with vector_cosine_ops.
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        normalized = raw / norms

        return normalized.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────────────────────


def create_embedding_service(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str | None = None,
) -> EmbeddingService:
    """Create an ``EmbeddingService`` backed by the real SentenceTransformer model.

    This is the production factory.  Tests should construct ``EmbeddingService``
    with a ``FakeEmbeddingProvider`` directly rather than calling this function.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.  Defaults to the project-standard
        ``BAAI/bge-small-en-v1.5``.
    device:
        PyTorch device string.  ``None`` auto-selects (CPU, CUDA, or MPS).

    Returns
    -------
    EmbeddingService
        Ready to use.  The model is loaded during this call.
    """
    provider = SentenceTransformerProvider(model_name, device=device)
    return EmbeddingService(provider)
