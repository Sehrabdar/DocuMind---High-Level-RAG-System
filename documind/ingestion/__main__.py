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
def main(
    input_path: Path | None,
    log_level: str | None,
    fail_fast: bool,
) -> None:
    """Ingest a corpus directory and print a summary.

    All supported documents (.md, .pdf, .html, .htm, .docx) found recursively
    under INPUT are loaded and normalized.  A summary is printed to stdout.
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

    # Exit with non-zero code if there were any failures
    if result.failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
