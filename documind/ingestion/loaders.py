"""
Document loaders for DocuMind Phase 1.

Architecture
------------
Each format has a dedicated loader class that inherits from ``BaseLoader``.
A ``LoaderDispatcher`` selects the correct loader based on file extension and
returns a ``NormalizedDocument``.

LlamaIndex is intentionally NOT used in this phase. We use format-specific
libraries directly (pypdf, beautifulsoup4, python-docx) which gives us:
  - Finer control over text extraction and normalization
  - Lighter import footprint during ingestion
  - No leakage of LlamaIndex types outside this module

Format-specific notes
---------------------
- **Markdown**: read directly; heading markers preserved for Phase 2 chunking.
- **PDF**: pypdf; pages joined with \\f so page info survives normalization.
- **HTML**: BeautifulSoup4 strips nav/scripts/styles before text extraction.
- **DOCX**: python-docx; paragraph order and table rows preserved.
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported extensions (lowercase, with leading dot)
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".pdf", ".html", ".htm", ".docx"})

# Extension → canonical file_type string
EXTENSION_TO_FILE_TYPE: dict[str, str] = {
    ".md": "markdown",
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".docx": "docx",
}


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────


class LoaderError(Exception):
    """Base class for loader failures."""

    def __init__(self, message: str, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


class UnsupportedFormatError(LoaderError):
    """Raised when a file's extension is not supported by any registered loader."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Unsupported file format '{path.suffix}' for file: {path}",
            path=path,
        )


class DocumentLoadError(LoaderError):
    """Raised when a supported file fails to load or parse."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

from documind.ingestion.models import DocumentMetadata, NormalizedDocument  # noqa: E402


def _document_id(source: Path) -> str:
    """Return a deterministic document ID from the resolved absolute source path.

    Uses the SHA-256 hex digest of the resolved path string.  This guarantees:
    - Same file on the same machine → same ID across re-ingestion runs.
    - Different files → different IDs.
    - ID does not depend on process execution order.
    """
    normalized = str(source.resolve())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _extract_title_from_markdown(content: str) -> str | None:
    """Return the text of the first H1 heading found in Markdown content."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _extract_title_from_filename(path: Path) -> str:
    """Derive a human-readable title from the filename stem."""
    return path.stem.replace("_", " ").replace("-", " ").title()


def _clean_whitespace(text: str) -> str:
    """Normalise whitespace in extracted text.

    - Strips trailing whitespace from each line.
    - Collapses runs of more than two consecutive blank lines to exactly two.
    - Strips leading and trailing whitespace from the whole string.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    result: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                result.append(line)
        else:
            blank_run = 0
            result.append(line)
    return "\n".join(result).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base loader
# ─────────────────────────────────────────────────────────────────────────────


class BaseLoader(ABC):
    """Abstract base class for all format-specific document loaders."""

    #: Canonical file_type string this loader handles.
    file_type: str = ""

    @abstractmethod
    def load(self, path: Path) -> NormalizedDocument:
        """Parse *path* and return a :class:`~documind.ingestion.models.NormalizedDocument`.

        Parameters
        ----------
        path:
            Absolute or relative path to the document to load.

        Raises
        ------
        DocumentLoadError
            If the file cannot be read or parsed.
        """

    def _make_metadata(self, path: Path, **extra: Any) -> DocumentMetadata:
        """Create a :class:`DocumentMetadata` for *path* with optional extras."""
        return DocumentMetadata(
            source=str(path.resolve()),
            filename=path.name,
            file_type=self.file_type,
            file_extension=path.suffix.lower(),
            ingested_at=datetime.now(timezone.utc),
            extra=extra,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Markdown loader
# ─────────────────────────────────────────────────────────────────────────────


class MarkdownLoader(BaseLoader):
    """Load ``.md`` files.

    Heading markers (``#``, ``##``, etc.) are preserved in the output so that
    Phase 2 can perform heading-aware chunking without re-parsing the source.
    """

    file_type = "markdown"

    def load(self, path: Path) -> NormalizedDocument:
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentLoadError(
                f"Failed to read Markdown file '{path}': {exc}", path=path
            ) from exc

        content = _clean_whitespace(raw)
        title = _extract_title_from_markdown(content) or _extract_title_from_filename(path)

        return NormalizedDocument(
            id=_document_id(path),
            source=str(path.resolve()),
            filename=path.name,
            file_type=self.file_type,
            title=title,
            content=content,
            metadata=self._make_metadata(path),
        )


# ─────────────────────────────────────────────────────────────────────────────
# PDF loader
# ─────────────────────────────────────────────────────────────────────────────


class PDFLoader(BaseLoader):
    """Load ``.pdf`` files using pypdf.

    Pages are separated by the form-feed character ``\\f`` so page information
    survives normalization and is available to Phase 2 for page-aware citation.

    Extraction errors on individual pages are logged and recorded in metadata
    but do not abort the entire load (best-effort policy).
    """

    file_type = "pdf"

    def load(self, path: Path) -> NormalizedDocument:
        try:
            import pypdf  # noqa: PLC0415
        except ImportError as exc:
            raise DocumentLoadError(
                "pypdf is not installed. Run: uv add pypdf", path=path
            ) from exc

        try:
            reader = pypdf.PdfReader(str(path))
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to open PDF '{path}': {exc}", path=path
            ) from exc

        pages: list[str] = []
        extraction_errors: list[str] = []

        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                pages.append(text)
            except Exception as exc:
                extraction_errors.append(f"page {i + 1}: {exc}")
                pages.append("")  # keep placeholder so page indices stay correct
                logger.warning("PDF '%s' page %d extraction error: %s", path, i + 1, exc)

        page_count = len(reader.pages)
        content = _clean_whitespace("\f".join(pages))

        # Title: PDF document info → filename fallback
        title: str | None = None
        try:
            pdf_info = reader.metadata
            if pdf_info and getattr(pdf_info, "title", None):
                candidate = str(pdf_info.title).strip()
                title = candidate or None
        except Exception:
            pass
        title = title or _extract_title_from_filename(path)

        extra: dict[str, Any] = {}
        if extraction_errors:
            extra["extraction_errors"] = extraction_errors

        metadata = self._make_metadata(path, **extra)
        metadata.page_count = page_count

        return NormalizedDocument(
            id=_document_id(path),
            source=str(path.resolve()),
            filename=path.name,
            file_type=self.file_type,
            title=title,
            content=content,
            metadata=metadata,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HTML loader
# ─────────────────────────────────────────────────────────────────────────────


class HTMLLoader(BaseLoader):
    """Load ``.html`` / ``.htm`` files using BeautifulSoup4.

    The following elements are removed before text extraction to eliminate
    boilerplate: ``script``, ``style``, ``nav``, ``header``, ``footer``,
    ``aside``, ``noscript``, ``svg``, ``form``, ``button``, ``iframe``.

    After stripping junk, the loader preferentially extracts content from
    semantic container elements (``<main>``, ``<article>``, elements with
    content-related IDs/classes) before falling back to ``<body>``.
    """

    file_type = "html"

    _JUNK_TAGS: tuple[str, ...] = (
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "noscript",
        "svg",
        "form",
        "button",
        "iframe",
    )

    def load(self, path: Path) -> NormalizedDocument:
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError as exc:
            raise DocumentLoadError(
                "beautifulsoup4 is not installed. Run: uv add beautifulsoup4", path=path
            ) from exc

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentLoadError(
                f"Failed to read HTML file '{path}': {exc}", path=path
            ) from exc

        # Parse — prefer lxml for speed, fall back to html.parser
        try:
            soup = BeautifulSoup(raw, "lxml")
        except Exception:
            try:
                soup = BeautifulSoup(raw, "html.parser")
            except Exception as exc2:
                raise DocumentLoadError(
                    f"Failed to parse HTML '{path}': {exc2}", path=path
                ) from exc2

        # Capture title from <h1> (higher priority) or <title> tag
        page_title: str | None = None
        title_tag = soup.find("title")
        if title_tag:
            candidate = title_tag.get_text(strip=True)
            if candidate:
                page_title = candidate

        h1_tag = soup.find("h1")
        if h1_tag:
            candidate = h1_tag.get_text(strip=True)
            if candidate:
                page_title = candidate  # h1 overrides <title>

        # Strip junk
        for tag_name in self._JUNK_TAGS:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Prefer semantic content containers
        content_re = re.compile(r"\b(content|main|body|article|doc)\b", re.I)
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id=content_re)
            or soup.find(class_=content_re)
            or soup.find("body")
            or soup
        )

        raw_text = main_content.get_text(separator="\n")
        content = _clean_whitespace(raw_text)

        title = page_title or _extract_title_from_filename(path)

        return NormalizedDocument(
            id=_document_id(path),
            source=str(path.resolve()),
            filename=path.name,
            file_type=self.file_type,
            title=title,
            content=content,
            metadata=self._make_metadata(path),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DOCX loader
# ─────────────────────────────────────────────────────────────────────────────


class DocxLoader(BaseLoader):
    """Load ``.docx`` files using python-docx.

    Paragraph order is preserved.  Table rows are extracted as pipe-delimited
    text lines appended after body paragraphs.  Sophisticated table extraction
    (merged cells, nested tables) is explicitly out of scope for Phase 1.
    """

    file_type = "docx"

    def load(self, path: Path) -> NormalizedDocument:
        try:
            import docx  # noqa: PLC0415
        except ImportError as exc:
            raise DocumentLoadError(
                "python-docx is not installed. Run: uv add python-docx", path=path
            ) from exc

        try:
            doc = docx.Document(str(path))
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to open DOCX '{path}': {exc}", path=path
            ) from exc

        parts: list[str] = []

        # Body paragraphs (preserves order including blanks for structure)
        for para in doc.paragraphs:
            parts.append(para.text)

        # Tables — append after body paragraphs
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    parts.append(row_text)

        content = _clean_whitespace("\n".join(parts))

        # Title: first Heading-styled paragraph or first non-empty paragraph
        title: str | None = None
        for para in doc.paragraphs:
            if para.style.name.startswith("Heading") and para.text.strip():
                title = para.text.strip()
                break
        if not title:
            for para in doc.paragraphs:
                if para.text.strip():
                    title = para.text.strip()[:120]
                    break
        title = title or _extract_title_from_filename(path)

        return NormalizedDocument(
            id=_document_id(path),
            source=str(path.resolve()),
            filename=path.name,
            file_type=self.file_type,
            title=title,
            content=content,
            metadata=self._make_metadata(path),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────


class LoaderDispatcher:
    """Select and invoke the appropriate loader for a given file path.

    Usage::

        dispatcher = LoaderDispatcher()
        doc = dispatcher.load(Path("docs/api_reference.md"))
    """

    def __init__(self) -> None:
        self._loaders: dict[str, BaseLoader] = {
            ".md": MarkdownLoader(),
            ".pdf": PDFLoader(),
            ".html": HTMLLoader(),
            ".htm": HTMLLoader(),
            ".docx": DocxLoader(),
        }

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Return the set of file extensions this dispatcher can handle."""
        return frozenset(self._loaders.keys())

    def is_supported(self, path: Path) -> bool:
        """Return ``True`` if *path*'s extension is supported."""
        return path.suffix.lower() in self._loaders

    def load(self, path: Path) -> NormalizedDocument:
        """Load *path* using the appropriate format loader.

        Raises
        ------
        UnsupportedFormatError
            If the file extension has no registered loader.
        DocumentLoadError
            If the file is supported but fails to parse.
        """
        ext = path.suffix.lower()
        loader = self._loaders.get(ext)
        if loader is None:
            raise UnsupportedFormatError(path)
        logger.debug("Loading '%s' with %s", path.name, type(loader).__name__)
        return loader.load(path)
