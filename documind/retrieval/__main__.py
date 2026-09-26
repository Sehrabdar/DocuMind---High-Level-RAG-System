"""
DocuMind Retrieval CLI — Phase 4 (dense) + Phase 5 (keyword) diagnostic tool.

Usage
-----
    # Dense retrieval (default):
    uv run python -m documind.retrieval --query "How do I create an API key?" --top-k 5

    # Dense retrieval (explicit):
    uv run python -m documind.retrieval --query "..." --top-k 5 --method dense

    # Keyword retrieval (no model needed):
    uv run python -m documind.retrieval --query "PostgreSQL connection pooling" --top-k 5 --method keyword

    # Keyword with document scope:
    uv run python -m documind.retrieval --query "JWT authentication" --method keyword --document-id <id>

Output
------
The command prints a ranked table of retrieved chunks.  For each result:
    - Rank, score (Dist+Sim for dense, FTS for keyword), retrieval method
    - Source document ID (first 16 chars)
    - Section path (heading breadcrumbs for Markdown documents)
    - Page number (for PDF documents)
    - Full chunk content

Score semantics
---------------
Dense:   Dist = cosine distance (lower = better), Sim = 1 - Dist (higher = better)
Keyword: FTS  = ts_rank_cd score (higher = better), not comparable to cosine distance

Exit codes
----------
0 — success (results returned or empty database)
1 — validation error (bad query, bad top-k)
2 — database or embedding error
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import textwrap

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m documind.retrieval",
        description="DocuMind retrieval diagnostic — dense (Phase 4) or keyword (Phase 5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        "-q",
        required=True,
        help="Natural-language query string.",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=None,
        metavar="K",
        help="Number of chunks to retrieve (default: DOCUMIND_RETRIEVAL_DEFAULT_TOP_K).",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["dense", "keyword"],
        default="dense",
        help=(
            "Retrieval strategy: 'dense' (pgvector HNSW, default) or "
            "'keyword' (PostgreSQL FTS, no embedding model needed)."
        ),
    )
    parser.add_argument(
        "--document-id",
        default=None,
        metavar="DOC_ID",
        help="Optional: restrict search to chunks from a specific document.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        metavar="URL",
        help=(
            "PostgreSQL async URL.  "
            "Defaults to DOCUMIND_DATABASE_URL env var or the Docker Compose default."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help=(
            "HuggingFace embedding model name (dense only).  "
            "Defaults to DOCUMIND_EMBEDDING_MODEL (BAAI/bge-small-en-v1.5)."
        ),
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        default=False,
        help="Suppress chunk content in output (show scores and metadata only).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser


def _print_results(
    query: str,
    results: list,
    method: str,
    show_content: bool = True,
) -> None:
    """Print retrieval results as a human-readable table."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()

    method_label = "Dense (cosine distance)" if method == "dense" else "Keyword (PostgreSQL FTS)"
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
    console.print(f"[dim]Method:[/dim] {method_label}")
    console.print(f"[dim]{len(results)} result(s) retrieved[/dim]\n")

    if not results:
        if method == "keyword":
            console.print(
                "[yellow]No keyword matches found.  Is the database populated and "
                "migration 0002 applied?  "
                "Run: uv run alembic upgrade head[/yellow]"
            )
        else:
            console.print(
                "[yellow]No chunks found.  Is the database populated?  "
                "Run the ingestion pipeline first.[/yellow]"
            )
        return

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold white",
        expand=False,
    )
    table.add_column("Rank", style="bold cyan", width=6, no_wrap=True)

    if method == "dense":
        table.add_column("Dist", width=7, no_wrap=True)
        table.add_column("Sim", width=7, no_wrap=True)
    else:
        table.add_column("FTS Score", width=9, no_wrap=True)

    table.add_column("Document", width=18, no_wrap=True)
    table.add_column("Section", width=28)
    table.add_column("Page", width=5, no_wrap=True)
    if show_content:
        table.add_column("Content preview", width=60)

    for r in results:
        section = " › ".join(r.section_path) if r.section_path else "[dim]—[/dim]"
        page = str(r.page) if r.page is not None else "[dim]—[/dim]"
        doc = r.document_id[:16] + "…"

        if method == "dense":
            score_cols = [
                f"{r.distance:.4f}" if r.distance is not None else "—",
                f"{r.similarity:.4f}" if r.similarity is not None else "—",
            ]
        else:
            score_cols = [
                f"{r.fts_score:.4f}" if r.fts_score is not None else "—",
            ]

        row = [str(r.rank), *score_cols, doc, section, page]
        if show_content:
            preview = textwrap.shorten(r.content, width=200, placeholder="…")
            row.append(preview)

        table.add_row(*row)

    console.print(table)

    if show_content:
        console.print()
        for r in results:
            section = " › ".join(r.section_path) if r.section_path else "(no section)"
            if method == "dense":
                score_str = (
                    f"dist={r.distance:.4f}  sim={r.similarity:.4f}"
                    if r.distance is not None
                    else "dist=—"
                )
            else:
                score_str = (
                    f"fts={r.fts_score:.4f}"
                    if r.fts_score is not None
                    else "fts=—"
                )
            console.rule(
                f"[bold]Rank {r.rank}[/bold]  "
                f"{score_str}  "
                f"chunk_id={r.chunk_id[:16]}…"
            )
            console.print(f"[dim]Document:[/dim] {r.document_id}")
            console.print(f"[dim]Section:[/dim]  {section}")
            if r.document_title:
                console.print(f"[dim]Title:[/dim]    {r.document_title}")
            if r.page is not None:
                console.print(f"[dim]Page:[/dim]     {r.page}")
            console.print()
            console.print(r.content)
            console.print()


async def _run(args: argparse.Namespace) -> int:
    """Async main: set up retriever, execute query, print results."""
    from config import settings
    from db.session import get_async_session, get_engine, get_session_factory
    from documind.retrieval.validation import QueryValidationError

    db_url = args.database_url or settings.database_url

    try:
        engine = get_engine(db_url)
        factory = get_session_factory(engine)

        if args.method == "dense":
            from documind.embeddings.service import create_embedding_service
            from documind.retrieval.service import DenseRetriever

            model_name = args.model or settings.embedding_model
            logger.info("Loading embedding model: %s", model_name)
            embedding_service = create_embedding_service(model_name)
            retriever = DenseRetriever(embedding_service)

        else:  # keyword
            from documind.retrieval.keyword import KeywordRetriever

            retriever = KeywordRetriever()  # type: ignore[assignment]

        async with get_async_session(factory) as session:
            results = await retriever.retrieve(
                query=args.query,
                top_k=args.top_k,
                session=session,
                document_id=args.document_id,
            )

        _print_results(
            query=args.query,
            results=results,
            method=args.method,
            show_content=not args.no_content,
        )
        return 0

    except QueryValidationError as exc:
        print(f"[ERROR] Invalid query: {exc}", file=sys.stderr)
        return 1

    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    except Exception as exc:
        logger.exception("Retrieval failed")
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
