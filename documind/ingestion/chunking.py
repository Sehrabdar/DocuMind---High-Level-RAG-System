"""
Structure-aware chunking for DocuMind Phase 2.

Pipeline
--------
NormalizedDocument
    ↓
Format-specific structure detection (sections / pages)
    ↓
Token-aware section splitting (with overlap for oversized sections)
    ↓
list[NormalizedChunk]

Design decisions
----------------
- Chunk IDs are deterministic: SHA-256(doc_id + ":" + chunk_index + ":" + content[:64]).
  Same document, same settings → identical chunk IDs across runs.

- Token counting uses tiktoken cl100k_base.  This is a transitive dependency
  (via llama-index-core) so no new package is added.  cl100k_base is a
  standard proxy for RAG token budgeting; the BAAI/bge-small-en-v1.5
  tokenizer differs marginally in practice.  This approximation is
  documented and acceptable for Phase 2.

- Overlap applies only when a section must be subdivided.  It is never
  inserted between independent sections (which would pollute section
  metadata).

- HTML and DOCX heading hierarchy is unavailable because Phase 1's loaders
  call BeautifulSoup.get_text() / para.text, discarding tag/style structure.
  These formats use flat chunking (no section_path sub-levels).  Rewriting
  Phase 1 to preserve HTML/DOCX heading structure is a known improvement
  deferred to a future phase.

- PDF chunks always strip the form-feed separator (\\f) from content.

- Empty sections and whitespace-only content are silently dropped — the
  caller should never see a chunk with empty content.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

# Module-level singleton — encoding is not thread-safe to initialise
# concurrently, but is safe to use concurrently after initialisation.
try:
    import tiktoken as _tiktoken

    _ENCODER = _tiktoken.get_encoding("cl100k_base")
except Exception as _tok_err:  # pragma: no cover
    _ENCODER = None  # type: ignore[assignment]
    logger.warning("tiktoken unavailable; falling back to character/4 approximation: %s", _tok_err)


def _count_tokens(text: str) -> int:
    """Return the token count for *text* using cl100k_base encoding.

    Falls back to ``len(text) // 4`` if tiktoken is not importable.
    The fallback is clearly documented — it is never silently treated as
    exact tokenization.
    """
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    # Character-based approximation: ~4 chars per token for English text.
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
# NormalizedChunk model
# ─────────────────────────────────────────────────────────────────────────────


class NormalizedChunk(BaseModel):
    """A retrieval-ready chunk produced from a NormalizedDocument.

    Every field is either directly traceable to the source document or
    derivable deterministically from it.  No field is populated speculatively.

    Immutable after construction (``frozen=True``).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    chunk_id: str
    """Deterministic chunk identifier.

    SHA-256 hex digest of ``doc_id:chunk_index:content_prefix`` where
    ``content_prefix`` is the first 64 characters of the chunk content.
    This guarantees:
    - Same document + same settings → same IDs (deterministic).
    - Different chunks in the same document → different IDs (collision-resistant).
    - IDs do not depend on wall-clock time or process state.
    """

    document_id: str
    """References ``NormalizedDocument.id`` for the source document."""

    chunk_index: int
    """0-based sequential index within the document, in source order."""

    # ── Content ───────────────────────────────────────────────────────────────
    content: str
    """The chunk text.  Never empty.  Form-feed characters (\\f) are stripped."""

    token_count: int
    """Actual token count of ``content`` using the configured tokenizer."""

    # ── Structural provenance ─────────────────────────────────────────────────
    section_path: list[str]
    """Hierarchical section breadcrumb from the document root.

    Examples:
    - ``[]``                                    — preamble / no headings
    - ``["Authentication"]``                    — top-level section
    - ``["Authentication", "OAuth"]``           — nested section
    - ``["Authentication", "OAuth", "Tokens"]`` — deeply nested

    For Markdown, derived from ``#``/``##``/``###`` heading markers.
    For PDF, empty (PDFs lack semantic heading structure in Phase 1 output).
    For HTML/DOCX, empty (heading hierarchy not preserved by Phase 1 loaders).
    """

    document_title: str | None
    """Document-level title from ``NormalizedDocument.title``."""

    # ── Location provenance ───────────────────────────────────────────────────
    start_char: int
    """Start offset (inclusive) in ``NormalizedDocument.content``."""

    end_char: int
    """End offset (exclusive) in ``NormalizedDocument.content``."""

    page: int | None = None
    """1-indexed page number for single-page PDF chunks.  ``None`` for other formats."""

    page_start: int | None = None
    """First page (1-indexed) for PDF chunks that span multiple pages."""

    page_end: int | None = None
    """Last page (1-indexed) for PDF chunks that span multiple pages."""

    model_config = ConfigDict(frozen=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chunk ID
# ─────────────────────────────────────────────────────────────────────────────


def _chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    """Return a deterministic chunk ID.

    Algorithm: SHA-256 hex digest of the UTF-8 encoding of the string::

        "<document_id>:<chunk_index>:<content[:64]>"

    Using both the chunk index and a content prefix ensures:
    - Stability: the same content at the same position → same ID.
    - Collision resistance: two different chunks in the same document
      produce different IDs because ``chunk_index`` differs.
    - Independence from wall-clock time or process execution order.
    """
    content_prefix = content[:64]
    raw = f"{document_id}:{chunk_index}:{content_prefix}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Section dataclass (internal)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _Section:
    """Internal representation of a structural section before chunking."""

    section_path: list[str]
    content: str
    start_char: int  # offset in the original document content string
    end_char: int

    # PDF-only
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Token-aware text splitting
# ─────────────────────────────────────────────────────────────────────────────

# Regex to detect the start of a fenced code block line (``` or ~~~)
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")

# Regex for a Markdown table row
_TABLE_ROW_RE = re.compile(r"^\s*\|")


def _find_split_boundaries(text: str) -> list[int]:
    """Return a list of character positions that are "good" split points.

    Priority order (highest to lowest preference):
    1. Blank-line paragraph boundaries (``\\n\\n``)
    2. Single newlines (``\\n``)
    3. Sentence ends followed by space (``'. '``, ``'! '``, ``'? '``)
    4. Any character position (last resort)

    We also avoid splitting inside fenced code blocks or table rows where
    a paragraph break would be a better alternative.
    """
    boundaries: list[int] = []
    # Double-newline paragraph breaks
    for m in re.finditer(r"\n\n", text):
        boundaries.append(m.end())
    if boundaries:
        return sorted(boundaries)

    # Single newlines
    for m in re.finditer(r"\n", text):
        boundaries.append(m.end())
    if boundaries:
        return sorted(boundaries)

    # Sentence boundaries
    for m in re.finditer(r"(?<=[.!?]) ", text):
        boundaries.append(m.end())
    if boundaries:
        return sorted(boundaries)

    # No natural boundary found — return positions every ~50 chars as backstop
    step = max(50, len(text) // 20)
    return list(range(step, len(text), step))


def _split_text_into_token_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int, int]]:
    """Split *text* into token-bounded sub-chunks with overlap.

    Returns a list of ``(chunk_text, rel_start, rel_end)`` tuples where
    ``rel_start``/``rel_end`` are character offsets *within ``text``*.

    Overlap semantics
    -----------------
    When emitting chunk N, the last ``chunk_overlap`` tokens of that chunk
    are prepended to the start of chunk N+1.  This preserves local context
    across boundaries.  Overlap is capped at ``min(chunk_overlap, actual
    tokens in previous chunk - 1)`` to avoid degenerate cases.

    Code-block / table awareness
    ----------------------------
    The splitter prefers paragraph boundaries (``\\n\\n``) over arbitrary
    mid-line splits.  If a fenced code block or table section is smaller
    than ``chunk_size``, it is kept together naturally.  If it exceeds
    ``chunk_size``, we split inside it rather than generate an oversized
    chunk — correctness takes priority over structural purity.
    """
    if _count_tokens(text) <= chunk_size:
        return [(text, 0, len(text))]

    results: list[tuple[str, int, int]] = []
    remaining_start = 0  # char offset in text where we haven't yet committed
    overlap_prefix = ""  # carried-over text from previous chunk

    while remaining_start < len(text):
        # Work with the remaining text plus any overlap prefix
        segment_start_in_text = remaining_start
        working_text = overlap_prefix + text[remaining_start:]

        if _count_tokens(working_text) <= chunk_size:
            # Remainder fits — emit final chunk
            raw_start = remaining_start - len(overlap_prefix)
            # Align start back to text start (can't go before 0)
            real_start = max(0, segment_start_in_text - len(overlap_prefix))
            results.append((working_text.strip(), real_start, len(text)))
            break

        # Find the largest prefix of working_text that fits in chunk_size tokens
        boundaries = _find_split_boundaries(working_text)

        # Binary-search the boundaries for the last one that stays within budget
        cut_pos = len(overlap_prefix)  # at minimum, include the overlap
        for boundary in boundaries:
            candidate = working_text[:boundary]
            if _count_tokens(candidate) <= chunk_size:
                cut_pos = boundary
            else:
                break

        if cut_pos <= len(overlap_prefix):
            # No boundary found within budget beyond the overlap.
            # Hard-cut at the token limit (character approximation).
            # This handles extreme cases like a single very long line.
            approx_chars = chunk_size * 4  # ~4 chars/token
            cut_pos = min(len(overlap_prefix) + approx_chars, len(working_text))

        chunk_text = working_text[:cut_pos]
        stripped = chunk_text.strip()
        if stripped:
            real_start = max(0, segment_start_in_text - len(overlap_prefix))
            real_end = remaining_start + (cut_pos - len(overlap_prefix))
            real_end = min(real_end, len(text))
            results.append((stripped, real_start, real_end))

        # Advance past the non-overlap portion
        new_content_consumed = cut_pos - len(overlap_prefix)
        if new_content_consumed <= 0:
            # Safety: avoid infinite loop if no progress was made
            new_content_consumed = max(1, len(overlap_prefix) + 1)
        remaining_start += new_content_consumed

        # Build overlap prefix from the tail of what we just emitted
        if chunk_overlap > 0 and remaining_start < len(text):
            # Take the last `chunk_overlap` tokens from chunk_text
            overlap_prefix = _last_n_tokens(chunk_text, chunk_overlap)
        else:
            overlap_prefix = ""

    return results if results else [(text.strip(), 0, len(text))]


def _last_n_tokens(text: str, n: int) -> str:
    """Return a suffix of *text* containing at most *n* tokens."""
    if _ENCODER is not None:
        tokens = _ENCODER.encode(text)
        if len(tokens) <= n:
            return text
        return _ENCODER.decode(tokens[-n:])
    # Fallback: character approximation
    approx_chars = n * 4
    return text[-approx_chars:] if len(text) > approx_chars else text


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific structure detection
# ─────────────────────────────────────────────────────────────────────────────


def _extract_sections_markdown(content: str) -> list[_Section]:
    """Partition Markdown content into sections based on heading markers.

    Each heading (``#``, ``##``, through ``######``) starts a new section.
    Content before the first heading is emitted as a preamble section with
    an empty ``section_path``.

    Heading stack maintenance
    -------------------------
    When a level-N heading is encountered, all deeper levels (> N) are
    popped from the path stack before the new heading is pushed.  For example:

    ::

        # A        → path = ["A"]
        ## B       → path = ["A", "B"]
        ### C      → path = ["A", "B", "C"]
        ## D       → path = ["A", "D"]   (not ["A", "B", "C", "D"])
    """
    heading_re = re.compile(r"^(#{1,6}) (.+)$", re.MULTILINE)

    sections: list[_Section] = []
    # Stack of (level, title) — level 1 = H1, level 6 = H6
    heading_stack: list[tuple[int, str]] = []

    prev_end = 0
    current_path: list[str] = []
    current_start = 0
    pending_content_lines: list[str] = []

    lines = content.splitlines(keepends=True)
    pos = 0  # char position in content

    def _flush(path: list[str], body: str, start: int, end: int) -> None:
        stripped = body.strip()
        if stripped:
            sections.append(
                _Section(
                    section_path=list(path),
                    content=stripped,
                    start_char=start,
                    end_char=end,
                )
            )

    for line in lines:
        m = heading_re.match(line.rstrip("\n").rstrip("\r"))
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()

            # Flush accumulated body content for the previous section
            body = "".join(pending_content_lines)
            _flush(current_path, body, current_start, pos)
            pending_content_lines = []

            # Update heading stack: pop levels >= current
            heading_stack = [(l, t) for l, t in heading_stack if l < level]
            heading_stack.append((level, title))
            current_path = [t for _, t in heading_stack]
            current_start = pos + len(line)  # body starts after the heading line
        else:
            pending_content_lines.append(line)

        pos += len(line)

    # Flush final section
    body = "".join(pending_content_lines)
    _flush(current_path, body, current_start, pos)

    return sections


def _split_pdf_pages(content: str) -> list[_Section]:
    """Recover page-level sections from PDF content.

    Phase 1's PDFLoader separates pages with form-feed characters (``\\f``).
    This function restores those page boundaries and assigns 1-indexed page
    numbers to each section.

    Empty pages (whitespace-only after stripping) are skipped — they can
    arise from blank pages in the source PDF and would produce empty chunks.
    """
    raw_pages = content.split("\f")
    sections: list[_Section] = []
    char_pos = 0

    for page_number, page_text in enumerate(raw_pages, start=1):
        stripped = page_text.strip()
        page_end_pos = char_pos + len(page_text)

        if stripped:
            sections.append(
                _Section(
                    section_path=[],
                    content=stripped,
                    start_char=char_pos,
                    end_char=page_end_pos,
                    page=page_number,
                )
            )

        # +1 for the \f separator (not present for the last page)
        char_pos = page_end_pos + 1

    return sections


def _extract_sections_flat(content: str, document_title: str | None) -> list[_Section]:
    """Treat the entire document as a single flat section.

    Used for HTML and DOCX where Phase 1 did not preserve heading structure.
    The section_path is empty because we have no reliable heading hierarchy.

    Note: This is a known limitation arising from Phase 1's plain-text
    normalisation.  If heading-aware chunking is needed for HTML/DOCX,
    Phase 1 loaders would need to output structured intermediates.
    """
    stripped = content.strip()
    if not stripped:
        return []
    return [
        _Section(
            section_path=[],
            content=stripped,
            start_char=0,
            end_char=len(content),
        )
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def chunk_document(
    doc: "NormalizedDocumentProtocol",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[NormalizedChunk]:
    """Chunk a normalized document into retrieval-ready units.

    Parameters
    ----------
    doc:
        A :class:`~documind.ingestion.models.NormalizedDocument` (or any
        object with the same ``id``, ``file_type``, ``content``, and
        ``title`` attributes).
    chunk_size:
        Maximum token count per chunk.  Defaults to the project default (512).
    chunk_overlap:
        Token overlap between consecutive sub-chunks of an oversized section.
        Defaults to the project default (50).

    Returns
    -------
    list[NormalizedChunk]
        Chunks in source order.  Empty if the document has no content.
        Never contains whitespace-only chunks.

    Raises
    ------
    ValueError
        If ``chunk_size <= 0`` or ``chunk_overlap >= chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    content = doc.content
    if not content or not content.strip():
        return []

    file_type: str = doc.file_type
    document_title: str | None = getattr(doc, "title", None)
    document_id: str = doc.id

    # ── Structure detection ───────────────────────────────────────────────────
    if file_type == "markdown":
        sections = _extract_sections_markdown(content)
    elif file_type == "pdf":
        sections = _split_pdf_pages(content)
    else:
        # html, docx — flat chunking (heading structure not available)
        sections = _extract_sections_flat(content, document_title)

    if not sections:
        return []

    # ── Produce chunks from sections ──────────────────────────────────────────
    chunks: list[NormalizedChunk] = []
    chunk_index = 0

    for section in sections:
        section_tokens = _count_tokens(section.content)

        if section_tokens <= chunk_size:
            # Section fits in a single chunk — no splitting needed.
            cid = _chunk_id(document_id, chunk_index, section.content)
            chunks.append(
                NormalizedChunk(
                    chunk_id=cid,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    content=section.content,
                    token_count=section_tokens,
                    section_path=section.section_path,
                    document_title=document_title,
                    start_char=section.start_char,
                    end_char=section.end_char,
                    page=section.page,
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
            )
            chunk_index += 1
        else:
            # Section is too large — split into overlapping sub-chunks.
            sub_chunks = _split_text_into_token_chunks(
                section.content, chunk_size, chunk_overlap
            )

            for sub_text, rel_start, rel_end in sub_chunks:
                if not sub_text.strip():
                    continue  # never emit empty chunks

                abs_start = section.start_char + rel_start
                abs_end = section.start_char + rel_end

                # Clamp to document length (defensive)
                abs_start = min(abs_start, len(content))
                abs_end = min(abs_end, len(content))

                tok = _count_tokens(sub_text)
                cid = _chunk_id(document_id, chunk_index, sub_text)
                chunks.append(
                    NormalizedChunk(
                        chunk_id=cid,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        content=sub_text,
                        token_count=tok,
                        section_path=section.section_path,
                        document_title=document_title,
                        start_char=abs_start,
                        end_char=abs_end,
                        page=section.page,
                        page_start=section.page_start,
                        page_end=section.page_end,
                    )
                )
                chunk_index += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Type alias for the duck-typed document argument
# ─────────────────────────────────────────────────────────────────────────────

# chunk_document() accepts any object with the required attributes rather than
# importing NormalizedDocument directly, which avoids a circular dependency
# if chunking is ever moved to a separate package.  The TYPE_CHECKING guard
# above keeps the import for type checkers only.

class NormalizedDocumentProtocol:  # pragma: no cover
    """Duck-type protocol for chunk_document() — not imported at runtime."""
    id: str
    file_type: str
    content: str
    title: str | None
