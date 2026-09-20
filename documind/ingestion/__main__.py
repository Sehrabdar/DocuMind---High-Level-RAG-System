"""
DocuMind ingestion CLI entry point.

Usage
-----
    uv run python -m documind.ingestion --input data/corpus
    uv run python -m documind.ingestion --input data/corpus --log-level DEBUG
    uv run python -m documind.ingestion --help

The command ingests all supported documents found recursively under --input,
then prints a summary table to stdout.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import settings
from documind.ingestion.chunking import NormalizedChunk, chunk_document
from documind.ingestion.pipeline import ingest_directory

console = Console()


def _configure_logging(level: str) -> None:
    """Set up root logger with the requested level."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_summary(result, input_path: Path) -> None:
    """Render a Rich ingestion summary to the console."""
    stats = result.stats

    # ── Header ────────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]DocuMind Ingestion[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print(f"  [dim]Input:[/dim]  {input_path.resolve()}\n")

    # ── Counts table ──────────────────────────────────────────────────────────
    counts = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    counts.add_column("Label", style="dim")
    counts.add_column("Value", justify="right", style="bold")

    counts.add_row("Discovered", str(stats.total_discovered))
    counts.add_row("Supported", str(stats.total_supported))
    counts.add_row("Skipped (unsupported)", str(stats.total_skipped))
    counts.add_row("", "")
    counts.add_row(
        "[green]Ingested[/green]",
        f"[green]{stats.total_ingested}[/green]",
    )
    if stats.total_failed:
        counts.add_row(
            "[red]Failed[/red]",
            f"[red]{stats.total_failed}[/red]",
        )
    else:
        counts.add_row("[dim]Failed[/dim]", "[dim]0[/dim]")

    console.print(counts)

    # ── Per-format breakdown ──────────────────────────────────────────────────
    if stats.by_format:
        fmt_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        fmt_table.add_column("Format", style="cyan")
        fmt_table.add_column("Documents", justify="right")

        for fmt, count in sorted(stats.by_format.items()):
            fmt_table.add_row(fmt.capitalize(), str(count))

        console.print(fmt_table)

    # ── Character count ───────────────────────────────────────────────────────
    console.print(f"  [dim]Total characters:[/dim]  {stats.total_characters:,}\n")

    # ── Failures ─────────────────────────────────────────────────────────────
    if result.failed:
        console.print("[bold red]Failures[/bold red]")
        for failure in result.failed:
            console.print(f"  [red]✗[/red]  {failure.path.name}")
            console.print(f"       [dim]{failure.error_type}:[/dim] {failure.error}")
        console.print()

    # ── Skipped files ─────────────────────────────────────────────────────────
    if result.skipped:
        console.print(
            f"[dim]Skipped {len(result.skipped)} unsupported file(s): "
            + ", ".join(p.name for p in result.skipped[:5])
            + ("[dim]…[/dim]" if len(result.skipped) > 5 else "")
            + "[/dim]"
        )
        console.print()

    # ── Final status ──────────────────────────────────────────────────────────
    if stats.total_failed == 0:
        console.print("[bold green]✓ Ingestion complete[/bold green]\n")
    else:
        console.print(
            f"[bold yellow]⚠ Ingestion complete with {stats.total_failed} failure(s)[/bold yellow]\n"
        )


def _print_chunk_summary(all_chunks: list[NormalizedChunk]) -> None:
    """Render a Rich chunking summary to the console."""
    if not all_chunks:
        console.print("[dim]No chunks produced.[/dim]\n")
        return

    console.print()
    console.print(
        Panel.fit(
            "[bold magenta]DocuMind Chunking[/bold magenta]",
            border_style="magenta",
        )
    )

    token_counts = [c.token_count for c in all_chunks]
    docs_with_chunks = len({c.document_id for c in all_chunks})
    chunks_with_section = sum(1 for c in all_chunks if c.section_path)

    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column("Label", style="dim")
    summary.add_column("Value", justify="right", style="bold")

    summary.add_row("Documents chunked", str(docs_with_chunks))
    summary.add_row("Total chunks", str(len(all_chunks)))
    summary.add_row("Chunks with section path", str(chunks_with_section))
    summary.add_row("", "")
    summary.add_row("Min tokens / chunk", str(min(token_counts)))
    summary.add_row("Max tokens / chunk", str(max(token_counts)))
    avg = sum(token_counts) / len(token_counts)
    summary.add_row("Avg tokens / chunk", f"{avg:.0f}")
    summary.add_row("Total tokens", str(sum(token_counts)))

    console.print(summary)

    # Per-document breakdown
    doc_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    doc_table.add_column("Document", style="cyan")
    doc_table.add_column("Chunks", justify="right")
    doc_table.add_column("Tokens", justify="right")
    doc_table.add_column("Sections", justify="right")

    from itertools import groupby
    by_doc: dict[str, list[NormalizedChunk]] = {}
    for chunk in all_chunks:
        by_doc.setdefault(chunk.document_id, []).append(chunk)

    for doc_id, doc_chunks in sorted(by_doc.items(), key=lambda kv: kv[1][0].chunk_index):
        # Use document_title or truncated doc_id
        label = (doc_chunks[0].document_title or doc_id[:12] + "…")
        n_chunks = len(doc_chunks)
        n_tokens = sum(c.token_count for c in doc_chunks)
        n_sections = len({tuple(c.section_path) for c in doc_chunks if c.section_path})
        doc_table.add_row(label, str(n_chunks), str(n_tokens), str(n_sections))

    console.print(doc_table)
    console.print("[bold magenta]✓ Chunking complete[/bold magenta]\n")


def _print_embed_summary(
    total_chunks: int,
    upserted: int,
    failed: int,
    elapsed_s: float,
) -> None:
    """Render a Rich embedding + persistence summary."""
    console.print()
    console.print(
        Panel.fit(
            "[bold blue]DocuMind Embedding[/bold blue]",
            border_style="blue",
        )
    )
    summary = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    summary.add_column("Label", style="dim")
    summary.add_column("Value", justify="right", style="bold")
    summary.add_row("Chunks processed", str(total_chunks))
    summary.add_row("Records upserted", str(upserted))
    if failed:
        summary.add_row("[red]Failed[/red]", f"[red]{failed}[/red]")
    summary.add_row("Elapsed", f"{elapsed_s:.1f}s")
    if total_chunks > 0 and elapsed_s > 0:
        cps = total_chunks / elapsed_s
        summary.add_row("Throughput", f"{cps:.0f} chunks/s")
    console.print(summary)
    if failed == 0:
        console.print("[bold blue]✓ Embedding complete[/bold blue]\n")
    else:
        console.print(
            f"[bold yellow]⚠ Embedding complete with {failed} failure(s)[/bold yellow]\n"
        )


@click.command(name="documind-ingest")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Root directory of the corpus to ingest. Defaults to DOCUMIND_CORPUS_DIR or data/corpus.",
)
@click.option(
    "--log-level",
    "log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Logging verbosity. Defaults to DOCUMIND_LOG_LEVEL or INFO.",
)
@click.option(
    "--fail-fast",
    "fail_fast",
    is_flag=True,
    default=False,
    help="Abort on the first document that fails to load.",
)
@click.option(
    "--chunk",
    "do_chunk",
    is_flag=True,
    default=False,
    help="Run structure-aware chunking after ingestion and print chunk statistics.",
)
@click.option(
    "--chunk-size",
    "chunk_size",
    type=int,
    default=None,
    help="Max tokens per chunk (default: DOCUMIND_CHUNK_SIZE or 512).",
)
@click.option(
    "--chunk-overlap",
    "chunk_overlap",
    type=int,
    default=None,
    help="Overlap tokens between sub-chunks (default: DOCUMIND_CHUNK_OVERLAP or 50).",
)
@click.option(
    "--embed",
    "do_embed",
    is_flag=True,
    default=False,
    help="Embed chunks and persist them to PostgreSQL (requires --chunk).",
)
@click.option(
    "--database-url",
    "database_url",
    default=None,
    help="PostgreSQL URL (default: DOCUMIND_DATABASE_URL or docker-compose default).",
)
def main(
    input_path: Path | None,
    log_level: str | None,
    fail_fast: bool,
    do_chunk: bool,
    chunk_size: int | None,
    chunk_overlap: int | None,
    do_embed: bool,
    database_url: str | None,
) -> None:
    """Ingest a corpus directory, optionally chunk and embed to PostgreSQL.

    All supported documents (.md, .pdf, .html, .htm, .docx) found recursively
    under INPUT are loaded and normalized.  A summary is printed to stdout.

    Pass --chunk to also run structure-aware chunking and see chunk statistics.
    Pass --embed (with --chunk) to embed chunks and persist to PostgreSQL.
    """
    effective_log_level = log_level or settings.log_level
    _configure_logging(effective_log_level)

    effective_input = input_path or settings.corpus_dir
    effective_input = Path(effective_input)

    if not effective_input.exists():
        console.print(f"[bold red]Error:[/bold red] Input directory not found: {effective_input}")
        sys.exit(1)

    try:
        result = ingest_directory(effective_input, fail_fast=fail_fast)
    except (FileNotFoundError, NotADirectoryError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise

    _print_summary(result, effective_input)

    all_chunks: list[NormalizedChunk] = []

    if do_chunk and result.documents:
        effective_chunk_size = chunk_size or settings.chunk_size
        effective_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        chunk_failures = 0
        for doc in result.documents:
            try:
                chunks = chunk_document(
                    doc,
                    chunk_size=effective_chunk_size,
                    chunk_overlap=effective_overlap,
                )
                all_chunks.extend(chunks)
            except Exception as exc:
                chunk_failures += 1
                console.print(
                    f"[red]Chunking failed for '{doc.filename}': {exc}[/red]"
                )

        _print_chunk_summary(all_chunks)
        if chunk_failures:
            console.print(
                f"[bold yellow]⚠ {chunk_failures} document(s) failed to chunk[/bold yellow]\n"
            )

    if do_embed:
        if not do_chunk:
            console.print(
                "[bold red]Error:[/bold red] --embed requires --chunk. "
                "Run with: --chunk --embed"
            )
            sys.exit(1)

        if not all_chunks:
            console.print("[dim]No chunks to embed.[/dim]\n")
        else:
            import asyncio
            import time

            from documind.embeddings.service import create_embedding_service
            from db.models import chunk_record_from_normalized
            from db.repository import ChunkRepository
            from db.session import get_async_session, get_engine, get_session_factory

            effective_db_url = database_url or settings.database_url
            effective_batch = settings.embedding_batch_size

            console.print(
                f"[dim]Loading embedding model: {settings.embedding_model}…[/dim]"
            )
            try:
                embedding_svc = create_embedding_service(settings.embedding_model)
            except Exception as exc:
                console.print(f"[bold red]Failed to load embedding model:[/bold red] {exc}")
                sys.exit(1)

            total_upserted = 0
            total_failed = 0
            t0 = time.monotonic()

            async def _embed_and_persist() -> None:
                nonlocal total_upserted, total_failed
                engine = get_engine(effective_db_url)
                session_factory = get_session_factory(engine)
                repo = ChunkRepository()

                for batch_start in range(0, len(all_chunks), effective_batch):
                    batch = all_chunks[batch_start : batch_start + effective_batch]
                    texts = [c.content for c in batch]
                    try:
                        embeddings = embedding_svc.embed_texts(
                            texts,
                            batch_size=effective_batch,
                            show_progress_bar=False,
                        )
                    except Exception as exc:
                        total_failed += len(batch)
                        console.print(f"[red]Embedding batch failed: {exc}[/red]")
                        continue

                    records = [
                        chunk_record_from_normalized(chunk, emb)
                        for chunk, emb in zip(batch, embeddings)
                    ]

                    try:
                        async with get_async_session(session_factory) as session:
                            upserted = await repo.upsert_chunks(records, session)
                            total_upserted += upserted
                    except Exception as exc:
                        total_failed += len(batch)
                        console.print(f"[red]Persistence batch failed: {exc}[/red]")

                await engine.dispose()

            asyncio.run(_embed_and_persist())
            elapsed = time.monotonic() - t0
            _print_embed_summary(len(all_chunks), total_upserted, total_failed, elapsed)

    # Exit with non-zero code if there were any failures
    if result.failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
