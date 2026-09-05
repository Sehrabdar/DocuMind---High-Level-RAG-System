"""
Ingestion pipeline for DocuMind Phase 1.

Provides:
- ``discover_files``  — recursive discovery of supported/unsupported files.
- ``IngestionResult`` — typed result container with statistics.
- ``ingest_directory`` — top-level directory ingestion entry point.

Design
------
The pipeline is best-effort: a failure on one document is recorded in
``IngestionResult.failed`` and logged, but does not abort the remaining
documents.  This mirrors a realistic document pipeline where malformed files
should not prevent the rest of the corpus from being indexed.

Callers that want fail-fast behaviour can pass ``fail_fast=True``, in which
case the first ``DocumentLoadError`` is re-raised immediately.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from documind.ingestion.loaders import (
    DocumentLoadError,
    LoaderDispatcher,
    UnsupportedFormatError,
)
from documind.ingestion.models import NormalizedDocument

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# File discovery
# ─────────────────────────────────────────────────────────────────────────────


def discover_files(
    root: Path,
    dispatcher: LoaderDispatcher | None = None,
) -> tuple[list[Path], list[Path]]:
    """Recursively discover files under *root*, partitioning them by support.

    Parameters
    ----------
    root:
        Directory to search (recursively).
    dispatcher:
        Optional :class:`~documind.ingestion.loaders.LoaderDispatcher` to use
        for support-checking.  A default instance is created if not provided.

    Returns
    -------
    supported, unsupported:
        Two sorted lists of :class:`~pathlib.Path` objects.  Sorting ensures
        deterministic ordering across different OS directory listing orders.

    Raises
    ------
    FileNotFoundError
        If *root* does not exist.
    NotADirectoryError
        If *root* exists but is not a directory.
    """
    if not root.exists():
        raise FileNotFoundError(f"Corpus root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus root is not a directory: {root}")

    if dispatcher is None:
        dispatcher = LoaderDispatcher()

    supported: list[Path] = []
    unsupported: list[Path] = []

    for path in sorted(root.rglob("*")):
        if path.is_file():
            if dispatcher.is_supported(path):
                supported.append(path)
            else:
                unsupported.append(path)
                logger.debug("Skipping unsupported file: %s", path)

    return supported, unsupported


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class IngestionFailure:
    """Records a single document ingestion failure."""

    path: Path
    error: str
    error_type: str  # class name of the exception


@dataclass
class IngestionStats:
    """Per-format and aggregate statistics for an ingestion run."""

    total_discovered: int = 0
    total_supported: int = 0
    total_ingested: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    total_characters: int = 0

    by_format: dict[str, int] = field(default_factory=dict)
    """Count of successfully ingested documents per file_type."""

    def record_success(self, doc: NormalizedDocument) -> None:
        self.total_ingested += 1
        self.total_characters += len(doc.content)
        self.by_format[doc.file_type] = self.by_format.get(doc.file_type, 0) + 1


@dataclass
class IngestionResult:
    """The complete result of a directory ingestion run.

    Attributes
    ----------
    documents:
        Successfully ingested and normalized documents.
    failed:
        Documents that could not be loaded, with error details.
    skipped:
        Files that were skipped because their format is unsupported.
    stats:
        Aggregate ingestion statistics.
    """

    documents: list[NormalizedDocument] = field(default_factory=list)
    failed: list[IngestionFailure] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    stats: IngestionStats = field(default_factory=IngestionStats)


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion entry point
# ─────────────────────────────────────────────────────────────────────────────


def ingest_directory(
    root: Path,
    *,
    fail_fast: bool = False,
    dispatcher: LoaderDispatcher | None = None,
) -> IngestionResult:
    """Ingest all supported documents under *root*.

    Parameters
    ----------
    root:
        Directory to ingest recursively.
    fail_fast:
        If ``True``, re-raise the first :class:`~documind.ingestion.loaders.DocumentLoadError`
        encountered instead of recording it and continuing.
        Default is ``False`` (best-effort, record failures).
    dispatcher:
        Optional custom :class:`~documind.ingestion.loaders.LoaderDispatcher`.
        A default instance is created if not provided.

    Returns
    -------
    IngestionResult
        Contains successfully loaded documents, failure records, skip list,
        and aggregate statistics.
    """
    if dispatcher is None:
        dispatcher = LoaderDispatcher()

    result = IngestionResult()

    logger.info("Discovering files in: %s", root)
    supported, unsupported = discover_files(root, dispatcher=dispatcher)

    result.skipped = unsupported
    result.stats.total_discovered = len(supported) + len(unsupported)
    result.stats.total_supported = len(supported)
    result.stats.total_skipped = len(unsupported)

    if unsupported:
        logger.info(
            "Skipping %d unsupported file(s): %s",
            len(unsupported),
            [p.name for p in unsupported],
        )

    logger.info("Loading %d supported file(s)…", len(supported))

    for path in supported:
        try:
            doc = dispatcher.load(path)
            result.documents.append(doc)
            result.stats.record_success(doc)
            logger.debug("Ingested: %s (%d chars)", path.name, len(doc.content))

        except (DocumentLoadError, UnsupportedFormatError) as exc:
            failure = IngestionFailure(
                path=path,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            result.failed.append(failure)
            result.stats.total_failed += 1
            logger.warning("Failed to ingest '%s': %s", path, exc)

            if fail_fast:
                raise

        except Exception as exc:
            # Catch unexpected errors so one bad document never aborts the run.
            failure = IngestionFailure(
                path=path,
                error=f"Unexpected error: {exc}",
                error_type=type(exc).__name__,
            )
            result.failed.append(failure)
            result.stats.total_failed += 1
            logger.exception("Unexpected error ingesting '%s'", path)

            if fail_fast:
                raise

    logger.info(
        "Ingestion complete — ingested: %d, failed: %d, skipped: %d",
        result.stats.total_ingested,
        result.stats.total_failed,
        result.stats.total_skipped,
    )
    return result
