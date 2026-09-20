"""Embedding layer for DocuMind Phase 3."""

from documind.embeddings.service import (
    EmbeddingProvider,
    EmbeddingService,
    FakeEmbeddingProvider,
    SentenceTransformerProvider,
    create_embedding_service,
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingService",
    "FakeEmbeddingProvider",
    "SentenceTransformerProvider",
    "create_embedding_service",
]
