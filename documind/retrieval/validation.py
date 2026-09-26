"""
Shared query validation for DocuMind retrieval services.

Both DenseRetriever (Phase 4) and KeywordRetriever (Phase 5) apply the same
validation rules for query strings and top_k values.  This module provides
a single authoritative implementation that both services import.

Centralising validation here ensures:
- Identical error messages and types across retrieval methods.
- A single place to adjust validation rules in future phases.
- Smaller, focused service classes that delegate validation cleanly.
"""

from __future__ import annotations

from config import settings


class QueryValidationError(ValueError):
    """Raised when a retrieval query string fails validation.

    Inherits from ValueError so callers can catch it without importing
    this class explicitly (though they should prefer the explicit import).
    The str() of this exception is a human-readable message suitable for
    display to end users.
    """


def validate_query(query: str) -> str:
    """Return the stripped query or raise QueryValidationError.

    Parameters
    ----------
    query:
        Raw query string from the caller.

    Returns
    -------
    str
        The stripped query (leading/trailing whitespace removed).

    Raises
    ------
    QueryValidationError
        If the query is empty, whitespace-only, or None-like.
    """
    if not query or not query.strip():
        raise QueryValidationError(
            "Query must be a non-empty, non-whitespace string. "
            f"Got: {query!r}"
        )
    return query.strip()


def validate_top_k(top_k: int | None, max_top_k: int) -> int:
    """Resolve and validate top_k, applying the configured default and ceiling.

    Parameters
    ----------
    top_k:
        Requested number of results.  None → use settings.retrieval_default_top_k.
    max_top_k:
        Hard ceiling enforced by this call (typically settings.retrieval_max_top_k,
        but can be overridden per-retriever for testing).

    Returns
    -------
    int
        Validated top_k value in the range [1, max_top_k].

    Raises
    ------
    ValueError
        If top_k is 0, negative, above max_top_k, or not an integer.
    """
    if top_k is None:
        return settings.retrieval_default_top_k

    # Reject booleans explicitly — Python bool is a subclass of int, so
    # isinstance(True, int) is True, but True/False are not valid top_k values.
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be an integer, got {type(top_k).__name__}")

    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    if top_k > max_top_k:
        raise ValueError(
            f"top_k ({top_k}) exceeds maximum allowed value ({max_top_k}). "
            f"Set DOCUMIND_RETRIEVAL_MAX_TOP_K to increase the limit."
        )
    return top_k
