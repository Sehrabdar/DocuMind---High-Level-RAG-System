"""
Normalized document representation for DocuMind.

Every loader must produce a ``NormalizedDocument``.  The rest of the pipeline
(chunking, embedding, retrieval, generation) depends *only* on this model —
never on LlamaIndex internals or any parser-specific object.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Structured metadata captured at ingestion time.

    Fields here are intentionally limited to source/document-level provenance.
    Chunk-level metadata (section, page ranges, chunk index) is added in
    Phase 2.
    """

    source: str
    """Absolute or project-relative path to the source file."""

    filename: str
    """Basename of the source file."""

    file_type: str
    """Normalized file type string: 'markdown' | 'pdf' | 'html' | 'docx'."""

    file_extension: str
    """Raw file extension including leading dot, e.g. '.md', '.pdf'."""

    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    """UTC timestamp of when this document was ingested."""

    # Page count for PDFs; None for other formats.
    page_count: int | None = None

    # Any extra metadata preserved from the parser (e.g. PDF author/subject).
    extra: dict[str, Any] = Field(default_factory=dict)


class NormalizedDocument(BaseModel):
    """The canonical document representation produced by the ingestion layer.

    Design invariants:
    - ``id`` is deterministic: same source path → same id across runs.
    - ``content`` is plain text (heading markers preserved for Markdown).
    - This model is independent of LlamaIndex; loaders must translate into it.
    - Serializable to JSON via ``model_dump_json()``.
    """

    id: str
    """Deterministic document identifier (SHA-256 hex of normalized source path)."""

    source: str
    """Absolute path to the source file at ingestion time."""

    filename: str
    """Basename of the source file."""

    file_type: str
    """Normalized file type: 'markdown' | 'pdf' | 'html' | 'docx'."""

    title: str | None = None
    """Best-effort document title extracted from content or filename."""

    content: str
    """Normalized text content.

    For Markdown, heading markers (``#``, ``##``, etc.) are preserved so that
    Phase 2 can perform heading-aware chunking.

    For PDF, pages are separated by a form-feed character (``\\f``) so that
    page information is not lost.

    For HTML, only the meaningful textual content is included (navigation,
    scripts, and styles are stripped).

    For DOCX, paragraph text is joined with newlines.
    """

    metadata: DocumentMetadata

    model_config = ConfigDict(frozen=False)
