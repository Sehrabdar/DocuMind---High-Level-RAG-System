"""
DocuMind retrieval package — Phase 4.

Public API
----------
    from documind.retrieval import DenseRetriever, RetrievedChunk, QueryValidationError
"""

from documind.retrieval.models import RetrievedChunk
from documind.retrieval.service import DenseRetriever, QueryValidationError

__all__ = [
    "DenseRetriever",
    "QueryValidationError",
    "RetrievedChunk",
]
