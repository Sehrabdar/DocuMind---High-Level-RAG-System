"""
Retrieval result models for DocuMind Phase 4.

These are the data contracts for the dense retrieval pipeline.
They are Pydantic models so they are:
- Strongly typed (no arbitrary dicts)
- JSON-serializable (useful for the future FastAPI layer)
- Self-documenting (field descriptions explain semantics)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class RetrievedChunk(BaseModel):
    """A single chunk returned by a retrieval query, with scores and provenance.

    Score semantics
    ---------------
    pgvector's ``<=>`` operator returns **cosine distance**:

        distance = 1 - cosine_similarity

    For L2-normalized vectors (all DocuMind embeddings are normalized):

        distance ∈ [0, 2]
        0.0 → identical direction (perfect semantic match)
        1.0 → orthogonal (unrelated)
        2.0 → exact opposite direction (adversarial)

    ``similarity`` is derived as ``1 - distance`` and lies in [-1, 1]:

        1.0  → perfect match
        0.0  → orthogonal / unrelated
        -1.0 → opposite

    **Always use ``distance`` for sorting** — lower distance = closer match.
    ``similarity`` is provided for human-readable display only.

    Rank
    ----
    ``rank = 1`` is the closest chunk (smallest distance).
    Rank is assigned after sorting by the retrieval service and is
    1-indexed.  Do not rely on database row order for rank.
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

    # ── Retrieval scores ──────────────────────────────────────────────────────
    distance: float = Field(
        description=(
            "Cosine distance from the query vector (pgvector <=> operator). "
            "Range [0, 2] for L2-normalized vectors. Lower is better."
        )
    )
    similarity: float = Field(
        description=(
            "Cosine similarity = 1 - distance. "
            "Range [-1, 1]. Higher is better. "
            "Derived from distance; provided for display convenience."
        )
    )
    rank: int = Field(
        description=(
            "1-indexed retrieval rank. rank=1 is the closest chunk. "
            "Assigned by the retrieval service after sorting by distance."
        )
    )

    @model_validator(mode="after")
    def _check_rank_positive(self) -> "RetrievedChunk":
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        return self

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
        # retrieval scores
        distance: float,
        rank: int,
    ) -> "RetrievedChunk":
        """Construct from raw database row data.

        Use this factory in repository code to avoid duplicating keyword
        argument mapping at every call site.
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
            distance=distance,
            similarity=1.0 - distance,
            rank=rank,
        )

    def format_summary(self) -> str:
        """Return a compact one-line summary for CLI diagnostic output.

        Example::

            [rank=1 dist=0.182 sim=0.818] doc=abc123... sec=['API Keys'] page=None
        """
        sec = " > ".join(self.section_path) if self.section_path else "(no section)"
        page_str = str(self.page) if self.page is not None else "—"
        return (
            f"[rank={self.rank} dist={self.distance:.4f} sim={self.similarity:.4f}] "
            f"doc={self.document_id[:12]}… "
            f"sec=[{sec}] "
            f"page={page_str}"
        )
