"""
Retrieval result models for DocuMind Phases 4 & 5.

These are the typed data contracts for the retrieval pipeline.
Pydantic models are used because they are:
- Strongly typed (no arbitrary dicts)
- JSON-serializable (useful for the future FastAPI layer)
- Self-documenting (field descriptions explain semantics)

Score semantics
---------------
DocuMind has two independent retrieval strategies with different scoring systems.
They must NOT be mixed or compared directly without a fusion strategy (Phase 6).

Dense retrieval (Phase 4):
    ``distance``   — pgvector cosine distance, range [0, 2] for L2-normalized vectors
                     lower = better match
    ``similarity`` — 1 - distance, range [-1, 1], higher = better
                     provided for display; always computed from distance

Keyword retrieval (Phase 5):
    ``fts_score``  — PostgreSQL ts_rank_cd() score, range [0, ∞)
                     higher = better match
                     NOT comparable to cosine distance or cosine similarity
                     NOT normalised; absolute value has no fixed meaning

The ``retrieval_method`` field explicitly labels which strategy produced each result.
This prevents accidental score comparison across retrieval methods.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RetrievedChunk(BaseModel):
    """A single chunk returned by any retrieval query, with scores and provenance.

    Supports both dense retrieval (Phase 4) and keyword retrieval (Phase 5).
    Score fields are optional with clear semantics per retrieval method.

    Dense retrieval results
    -----------------------
    - ``retrieval_method == "dense"``
    - ``distance`` is populated (cosine distance, lower = better)
    - ``similarity`` is populated (1 - distance, higher = better)
    - ``fts_score`` is None

    Keyword retrieval results
    -------------------------
    - ``retrieval_method == "keyword"``
    - ``fts_score`` is populated (ts_rank_cd score, higher = better)
    - ``distance`` is None
    - ``similarity`` is None

    Rank
    ----
    ``rank = 1`` is always the best match for the chosen retrieval method:
    - Dense:   rank 1 = smallest distance
    - Keyword: rank 1 = largest fts_score
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id: str = Field(description="Deterministic SHA-256 chunk identifier from Phase 2.")
    document_id: str = Field(description="SHA-256 of the source document path.")

    # ── Content ───────────────────────────────────────────────────────────────
    content: str = Field(description="Chunk text as stored in the database.")
    token_count: int = Field(description="Approximate token count of the content.")

    # ── Structural provenance ─────────────────────────────────────────────────
    chunk_index: int = Field(
        description="Zero-based position of this chunk within its source document."
    )
    section_path: list[str] = Field(
        description=(
            "Heading breadcrumbs from Markdown structure detection "
            "(e.g. ['Authentication', 'OAuth']). Empty for PDF/HTML/DOCX."
        )
    )
    document_title: str | None = Field(
        default=None,
        description="Title of the source document if available.",
    )

    # ── Location provenance ───────────────────────────────────────────────────
    page: int | None = Field(
        default=None,
        description="1-indexed page number (PDF chunks only).",
    )
    page_start: int | None = Field(
        default=None,
        description="First page of a multi-page PDF chunk.",
    )
    page_end: int | None = Field(
        default=None,
        description="Last page of a multi-page PDF chunk.",
    )
    start_char: int = Field(
        description="Character offset of chunk start within the document."
    )
    end_char: int = Field(
        description="Character offset of chunk end within the document."
    )

    # ── Retrieval method ──────────────────────────────────────────────────────
    retrieval_method: Literal["dense", "keyword"] = Field(
        description=(
            "The retrieval strategy that produced this result. "
            "'dense' = pgvector cosine-distance search (Phase 4). "
            "'keyword' = PostgreSQL FTS ts_rank_cd search (Phase 5). "
            "Score fields differ by method — see class docstring."
        )
    )

    # ── Dense retrieval scores (Phase 4) ─────────────────────────────────────
    distance: float | None = Field(
        default=None,
        description=(
            "Cosine distance from the query vector (pgvector <=> operator). "
            "Range [0, 2] for L2-normalized vectors. Lower is better. "
            "Only populated when retrieval_method == 'dense'."
        ),
    )
    similarity: float | None = Field(
        default=None,
        description=(
            "Cosine similarity = 1 - distance. "
            "Range [-1, 1]. Higher is better. "
            "Derived from distance; provided for display convenience. "
            "Only populated when retrieval_method == 'dense'."
        ),
    )

    # ── Keyword retrieval score (Phase 5) ─────────────────────────────────────
    fts_score: float | None = Field(
        default=None,
        description=(
            "PostgreSQL ts_rank_cd() score for keyword retrieval. "
            "Higher is better. Not normalised; not comparable to cosine distance. "
            "Only populated when retrieval_method == 'keyword'."
        ),
    )

    # ── Rank ─────────────────────────────────────────────────────────────────
    rank: int = Field(
        description=(
            "1-indexed retrieval rank. rank=1 is the best result for the "
            "chosen retrieval method. Assigned by the retrieval service."
        )
    )

    @model_validator(mode="after")
    def _validate_scores_and_rank(self) -> RetrievedChunk:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        return self

    # ── Dense factory ─────────────────────────────────────────────────────────

    @classmethod
    def from_row(
        cls,
        *,
        # identity
        chunk_id: str,
        document_id: str,
        # content
        content: str,
        token_count: int,
        # structural provenance
        chunk_index: int,
        section_path: list[str],
        document_title: str | None,
        # location provenance
        page: int | None,
        page_start: int | None,
        page_end: int | None,
        start_char: int,
        end_char: int,
        # dense retrieval scores
        distance: float,
        rank: int,
    ) -> RetrievedChunk:
        """Construct a dense-retrieval result from raw database row data.

        Use this factory in VectorRepository / DenseRetriever to avoid
        duplicating keyword argument mapping at every call site.
        ``similarity`` is always computed as ``1.0 - distance``.
        """
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            token_count=token_count,
            chunk_index=chunk_index,
            section_path=section_path,
            document_title=document_title,
            page=page,
            page_start=page_start,
            page_end=page_end,
            start_char=start_char,
            end_char=end_char,
            retrieval_method="dense",
            distance=distance,
            similarity=1.0 - distance,
            fts_score=None,
            rank=rank,
        )

    # ── Keyword factory ───────────────────────────────────────────────────────

    @classmethod
    def from_fts_row(
        cls,
        *,
        # identity
        chunk_id: str,
        document_id: str,
        # content
        content: str,
        token_count: int,
        # structural provenance
        chunk_index: int,
        section_path: list[str],
        document_title: str | None,
        # location provenance
        page: int | None,
        page_start: int | None,
        page_end: int | None,
        start_char: int,
        end_char: int,
        # keyword retrieval score
        fts_score: float,
        rank: int,
    ) -> RetrievedChunk:
        """Construct a keyword-retrieval result from raw database row data.

        Use this factory in KeywordRepository / KeywordRetriever.
        ``distance`` and ``similarity`` are None for keyword results — they
        carry no meaning here and must not be used for comparison with dense scores.

        ``fts_score`` is the raw ts_rank_cd() value from PostgreSQL:
            higher = better match
            not normalised; absolute value has no fixed semantic meaning
        """
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            token_count=token_count,
            chunk_index=chunk_index,
            section_path=section_path,
            document_title=document_title,
            page=page,
            page_start=page_start,
            page_end=page_end,
            start_char=start_char,
            end_char=end_char,
            retrieval_method="keyword",
            distance=None,
            similarity=None,
            fts_score=fts_score,
            rank=rank,
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def format_summary(self) -> str:
        """Return a compact one-line summary for CLI diagnostic output.

        Examples::

            [dense  rank=1 dist=0.182 sim=0.818] doc=abc123... sec=['API Keys'] page=None
            [keyword rank=1 fts=0.0759]           doc=abc123... sec=['Auth']     page=None
        """
        sec = " > ".join(self.section_path) if self.section_path else "(no section)"
        page_str = str(self.page) if self.page is not None else "—"

        if self.retrieval_method == "dense":
            score_part = (
                f"dist={self.distance:.4f} sim={self.similarity:.4f}"
                if self.distance is not None
                else "dist=? sim=?"
            )
        else:
            score_part = (
                f"fts={self.fts_score:.4f}"
                if self.fts_score is not None
                else "fts=?"
            )

        return (
            f"[{self.retrieval_method} rank={self.rank} {score_part}] "
            f"doc={self.document_id[:12]}… "
            f"sec=[{sec}] "
            f"page={page_str}"
        )
