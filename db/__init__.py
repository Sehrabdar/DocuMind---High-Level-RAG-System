"""Database layer for DocuMind — models, session factory, repository (Phase 3), vector search (Phase 4)."""

from db.models import Base, ChunkRecord, chunk_record_from_normalized
from db.repository import ChunkRepository
from db.session import get_async_session, get_engine, get_session_factory
from db.vector_repository import VectorRepository

__all__ = [
    "Base",
    "ChunkRecord",
    "chunk_record_from_normalized",
    "ChunkRepository",
    "VectorRepository",
    "get_engine",
    "get_session_factory",
    "get_async_session",
]
