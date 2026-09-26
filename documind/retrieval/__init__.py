"""
DocuMind retrieval package — Phases 4 & 5.

Public API
----------
    from documind.retrieval import DenseRetriever, KeywordRetriever
    from documind.retrieval import RetrievedChunk, QueryValidationError
"""

from documind.retrieval.keyword import KeywordRetriever
from documind.retrieval.models import RetrievedChunk
from documind.retrieval.service import DenseRetriever
from documind.retrieval.validation import QueryValidationError

__all__ = [
    "DenseRetriever",
    "KeywordRetriever",
    "QueryValidationError",
    "RetrievedChunk",
]
