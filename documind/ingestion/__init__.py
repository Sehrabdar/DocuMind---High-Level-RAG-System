"""Ingestion layer — document loading, normalisation, and chunking."""

from documind.ingestion.chunking import NormalizedChunk, chunk_document
from documind.ingestion.models import DocumentMetadata, NormalizedDocument

__all__ = [
    "NormalizedDocument",
    "DocumentMetadata",
    "NormalizedChunk",
    "chunk_document",
]
