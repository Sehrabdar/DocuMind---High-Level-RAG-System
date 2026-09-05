"""
Fixture generator for DocuMind.

Creates a representative mixed-format corpus under data/corpus/ and small test
fixtures under tests/fixtures/ — entirely locally, without any internet access.

Run:
    uv run python scripts/create_fixtures.py

Idempotent: re-running overwrites existing files with identical content.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Markdown fixtures
# ─────────────────────────────────────────────────────────────────────────────

MARKDOWN_DOCS: dict[str, str] = {
    "authentication.md": """\
# Authentication

DocuMind supports multiple authentication strategies for securing API access.

## OAuth2 with Password Flow

The recommended approach for browser-based applications is the OAuth2 Password
flow, which exchanges user credentials for a short-lived access token.

### Obtaining a Token

Send a `POST` request to `/auth/token` with your credentials:

```http
POST /auth/token HTTP/1.1
Content-Type: application/x-www-form-urlencoded

username=alice&password=secret
```

A successful response returns:

```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### OAuth2PasswordBearer

Use the `Authorization` header for subsequent requests:

```http
GET /api/documents HTTP/1.1
Authorization: Bearer eyJ...
```

## API Key Authentication

For server-to-server integrations, API keys are preferable.

### Generating an API Key

API keys are scoped to a single project and can be revoked independently.

```bash
curl -X POST https://api.documind.io/keys \\
  -H "Authorization: Bearer $ADMIN_TOKEN" \\
  -d '{"name": "ci-pipeline", "scopes": ["read:docs"]}'
```

## Rate Limiting

All endpoints enforce rate limits:

| Tier    | Requests/minute |
|---------|-----------------|
| Free    | 60              |
| Pro     | 600             |
| Enterprise | Unlimited    |

## Security Considerations

- Tokens are signed with RS256.
- API keys are stored as bcrypt hashes; the raw key is shown only once.
- All traffic must use TLS 1.2 or later.
""",
    "vector_search.md": """\
# Vector Search

DocuMind uses dense vector embeddings to enable semantic search over ingested
technical documentation.

## Overview

Unlike keyword search, vector search retrieves documents based on *semantic
similarity* rather than exact token overlap.  This allows queries like
"how do I authenticate?" to match documents discussing "OAuth2 token flow"
even when the exact words differ.

## Embedding Model

| Property  | Value                  |
|-----------|------------------------|
| Model     | BAAI/bge-small-en-v1.5 |
| Dimension | 384                    |
| Runtime   | sentence-transformers  |

## pgvector Storage

Embeddings are stored in PostgreSQL using the pgvector extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id),
    content     TEXT NOT NULL,
    embedding   vector(384),
    metadata    JSONB
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

## Query Flow

1. The user query is embedded using the same model.
2. An ANN search is performed over the HNSW index.
3. Top-k candidates are returned for downstream reranking.

## Limitations

- Dense retrieval alone misses exact-match queries (e.g. error codes).
- Hybrid retrieval (dense + BM25) is planned for Phase 3.
""",
    "chunking_strategy.md": """\
# Chunking Strategy

Chunking transforms a normalized document into a sequence of overlapping
text segments suitable for embedding and retrieval.

## Goals

- Preserve semantic coherence within each chunk.
- Respect document structure (headings, paragraphs).
- Maintain provenance metadata for citation generation.

## Heading-Aware Chunking

Documents are split at heading boundaries first, then recursively split
by paragraph if a section exceeds the maximum chunk size.

```
Document
  └── Section (H1)
        └── Subsection (H2)
              └── Chunk 1  ← max 512 tokens
              └── Chunk 2  ← overlap with Chunk 1
```

## Parameters

| Parameter       | Default | Description                          |
|-----------------|---------|--------------------------------------|
| max_tokens      | 512     | Hard upper bound per chunk           |
| overlap_tokens  | 64      | Token overlap between adjacent chunks|
| min_tokens      | 50      | Minimum chunk size (avoids tiny chunks)|

## Metadata Preserved

Each chunk carries:
- `document_id` — parent document reference
- `chunk_index` — position within document
- `section` — enclosing heading text
- `page` — page number (PDF only)
- `char_start`, `char_end` — character offsets in original content

> **Note**: Chunking is implemented in Phase 2. This document describes the
> planned strategy.
""",
}

# ─────────────────────────────────────────────────────────────────────────────
# HTML fixtures
# ─────────────────────────────────────────────────────────────────────────────

HTML_DOCS: dict[str, str] = {
    "quickstart.html": """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DocuMind Quickstart Guide</title>
  <style>
    body { font-family: sans-serif; margin: 0; }
    nav { background: #333; color: white; padding: 1em; }
    .content { max-width: 800px; margin: auto; padding: 2em; }
    footer { background: #eee; padding: 1em; text-align: center; }
  </style>
</head>
<body>
  <nav>
    <a href="/">Home</a> | <a href="/docs">Docs</a> | <a href="/api">API</a>
  </nav>

  <main class="content">
    <h1>Quickstart Guide</h1>

    <p>Welcome to DocuMind. This guide walks you through ingesting your first
    corpus and running a query in under five minutes.</p>

    <h2>Prerequisites</h2>
    <ul>
      <li>Python 3.12+</li>
      <li>Docker (for PostgreSQL + pgvector)</li>
      <li><code>uv</code> package manager</li>
    </ul>

    <h2>Installation</h2>
    <p>Clone the repository and install dependencies:</p>
    <pre><code>git clone https://github.com/example/documind.git
cd documind
uv sync</code></pre>

    <h2>Ingest a Corpus</h2>
    <p>Place your documents in <code>data/corpus/</code> and run:</p>
    <pre><code>uv run python -m documind.ingestion --input data/corpus</code></pre>

    <h2>Supported Formats</h2>
    <ul>
      <li>Markdown (<code>.md</code>)</li>
      <li>PDF (<code>.pdf</code>)</li>
      <li>HTML (<code>.html</code>)</li>
      <li>DOCX (<code>.docx</code>)</li>
    </ul>

    <h2>Next Steps</h2>
    <p>See the <a href="/docs/architecture">Architecture Guide</a> for a deep
    dive into the retrieval pipeline.</p>
  </main>

  <footer>
    <script>console.log("analytics");</script>
    &copy; 2024 DocuMind. All rights reserved.
  </footer>
</body>
</html>
""",
    "api_reference.html": """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>API Reference — DocuMind</title>
  <style>body { font-family: monospace; }</style>
  <script src="/static/analytics.js"></script>
</head>
<body>
  <nav id="sidebar">
    <ul>
      <li><a href="#ingest">POST /ingest</a></li>
      <li><a href="#query">POST /query</a></li>
      <li><a href="#documents">GET /documents</a></li>
    </ul>
  </nav>

  <article id="content">
    <h1>API Reference</h1>
    <p>All endpoints accept and return JSON. Authentication is via Bearer token.</p>

    <h2 id="ingest">POST /ingest</h2>
    <p>Trigger corpus ingestion for a given directory.</p>

    <h3>Request</h3>
    <pre><code>{
  "corpus_path": "data/corpus",
  "fail_fast": false
}</code></pre>

    <h3>Response</h3>
    <pre><code>{
  "ingested": 9,
  "failed": 1,
  "skipped": 2,
  "total_characters": 48320
}</code></pre>

    <h2 id="query">POST /query</h2>
    <p>Run a RAG query against the ingested corpus.</p>

    <h3>Request</h3>
    <pre><code>{
  "query": "How does OAuth2 authentication work?",
  "top_k": 5
}</code></pre>

    <h3>Response</h3>
    <pre><code>{
  "answer": "OAuth2 authentication works by...",
  "citations": [
    {"document": "authentication.md", "section": "OAuth2 with Password Flow"}
  ]
}</code></pre>

    <h2 id="documents">GET /documents</h2>
    <p>List all ingested documents.</p>
  </article>

  <footer>
    DocuMind API v0.1 — <a href="/changelog">Changelog</a>
  </footer>
</body>
</html>
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# Minimal valid PDF builder (no external library needed)
# ─────────────────────────────────────────────────────────────────────────────

def _build_minimal_pdf(title: str, pages: list[str]) -> bytes:
    """Build a minimal but valid PDF binary with plain text pages.

    This produces a spec-compliant PDF 1.4 that pypdf can parse.  It is used
    only for generating test fixtures — it does not attempt to replicate a
    full-featured PDF authoring library.
    """
    objects: list[bytes] = []
    offsets: list[int] = []
    buf = bytearray()

    def add_obj(content: bytes) -> int:
        idx = len(objects) + 1
        objects.append(content)
        return idx

    def write(data: bytes) -> None:
        buf.extend(data)

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    write(header)

    # Build page streams
    page_stream_ids: list[int] = []
    page_ids: list[int] = []

    for page_text in pages:
        # Content stream
        stream_body = (
            f"BT\n/F1 12 Tf\n50 750 Td\n"
            f"({page_text.replace('(', r'\\(').replace(')', r'\\)').replace('\\n', ' ')})"
            f" Tj\nET\n"
        ).encode()
        stream_obj = (
            b"<< /Length " + str(len(stream_body)).encode() + b" >>\n"
            b"stream\n" + stream_body + b"\nendstream"
        )
        sid = add_obj(stream_obj)
        page_stream_ids.append(sid)

        page_obj = (
            b"<< /Type /Page /Parent 3 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Contents " + str(sid).encode() + b" 0 R "
            b"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>"
        )
        pid = add_obj(page_obj)
        page_ids.append(pid)

    # Pages dict (obj 3 — placeholder)
    kids = b" ".join(str(pid).encode() + b" 0 R" for pid in page_ids)
    pages_obj = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count "
        + str(len(page_ids)).encode() + b" >>"
    )
    pages_id = add_obj(pages_obj)  # this becomes len(objects) after add

    # Info dict
    info_obj = (
        b"<< /Title (" + title.encode() + b") /Producer (DocuMind fixture generator) >>"
    )
    info_id = add_obj(info_obj)

    # Catalog
    catalog_obj = b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>"
    catalog_id = add_obj(catalog_obj)

    # Now write objects and record offsets
    buf.clear()
    write(header)

    offsets = []
    for i, obj_content in enumerate(objects, start=1):
        offsets.append(len(buf))
        write(str(i).encode() + b" 0 obj\n" + obj_content + b"\nendobj\n")

    # Cross-reference table
    xref_offset = len(buf)
    write(b"xref\n")
    write(b"0 " + str(len(objects) + 1).encode() + b"\n")
    write(b"0000000000 65535 f \n")
    for off in offsets:
        write(f"{off:010d} 00000 n \n".encode())

    write(b"trailer\n")
    write(
        b"<< /Size " + str(len(objects) + 1).encode()
        + b" /Root " + str(catalog_id).encode() + b" 0 R"
        + b" /Info " + str(info_id).encode() + b" 0 R >>\n"
    )
    write(b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF\n")

    return bytes(buf)


PDF_DOCS: dict[str, tuple[str, list[str]]] = {
    "rag_architecture.pdf": (
        "RAG Architecture Overview",
        [
            (
                "RAG Architecture Overview\n\n"
                "Retrieval-Augmented Generation (RAG) combines a retrieval system with\n"
                "a generative language model to produce grounded, citation-backed answers.\n\n"
                "Pipeline Stages:\n"
                "1. Document Ingestion\n"
                "2. Chunking and Embedding\n"
                "3. Vector Storage (pgvector)\n"
                "4. Retrieval (Dense + Keyword)\n"
                "5. Reranking (Cross-Encoder)\n"
                "6. Generation (Claude API)\n"
                "7. Citation Insertion"
            ),
            (
                "Page 2: Retrieval Details\n\n"
                "Dense Retrieval uses HNSW approximate nearest-neighbor search over\n"
                "384-dimensional embeddings produced by BAAI/bge-small-en-v1.5.\n\n"
                "Keyword Retrieval uses PostgreSQL full-text search (tsvector/tsquery)\n"
                "to complement dense retrieval for exact-match queries.\n\n"
                "Reciprocal Rank Fusion (RRF) merges the two ranked lists:\n"
                "  score(d) = sum(1 / (k + rank_i(d)))\n"
                "where k=60 is the standard RRF constant."
            ),
        ],
    ),
    "deployment_guide.pdf": (
        "Deployment Guide",
        [
            (
                "DocuMind Deployment Guide\n\n"
                "DocuMind is designed for deployment on Google Cloud Run.\n\n"
                "Prerequisites:\n"
                "- Docker\n"
                "- Google Cloud SDK\n"
                "- A Cloud SQL PostgreSQL 16 instance with pgvector\n\n"
                "Local Development:\n"
                "  docker-compose up -d\n"
                "  uv run python -m documind.ingestion --input data/corpus"
            ),
            (
                "Page 2: Environment Variables\n\n"
                "DOCUMIND_CORPUS_DIR   Path to corpus directory\n"
                "DOCUMIND_LOG_LEVEL    Logging verbosity (INFO)\n"
                "DATABASE_URL          PostgreSQL connection string\n"
                "ANTHROPIC_API_KEY     Claude API key (Phase 5+)\n\n"
                "Health Check:\n"
                "  GET /health -> {status: ok, version: 0.1.0}"
            ),
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# DOCX builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_docx(title: str, sections: list[tuple[str, str]]) -> None:
    """Write a DOCX file using python-docx."""
    import docx  # noqa: PLC0415
    doc = docx.Document()
    doc.add_heading(title, level=1)
    for heading, body in sections:
        doc.add_heading(heading, level=2)
        for paragraph in body.strip().split("\n\n"):
            doc.add_paragraph(paragraph.strip())
    return doc


DOCX_DOCS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "data_model.docx": (
        "DocuMind Data Model",
        [
            (
                "Document",
                "The Document table stores one row per ingested source file.\n\n"
                "Columns: id (UUID), source (TEXT), filename (TEXT), file_type (TEXT),\n"
                "title (TEXT), content (TEXT), ingested_at (TIMESTAMPTZ), metadata (JSONB).",
            ),
            (
                "Chunk",
                "The Chunk table stores text segments produced by the chunking pipeline.\n\n"
                "Columns: id (UUID), document_id (UUID FK), chunk_index (INT),\n"
                "content (TEXT), embedding (vector(384)), section (TEXT),\n"
                "page (INT), char_start (INT), char_end (INT), metadata (JSONB).",
            ),
            (
                "EvaluationResult",
                "Stores RAGAS evaluation scores for each query-answer pair.\n\n"
                "Columns: id (UUID), query (TEXT), answer (TEXT),\n"
                "faithfulness (FLOAT), answer_relevancy (FLOAT),\n"
                "context_recall (FLOAT), evaluated_at (TIMESTAMPTZ).",
            ),
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures (small, deterministic)
# ─────────────────────────────────────────────────────────────────────────────

TEST_MARKDOWN = """\
# Test Document

This is a test document for unit testing.

## Section One

Content of section one.

## Section Two

Content of section two.
"""

TEST_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
  <nav><a href="/">Home</a></nav>
  <main>
    <h1>Test Page</h1>
    <p>This is test content for unit testing.</p>
    <p>It has multiple paragraphs.</p>
  </main>
  <footer>Footer text</footer>
  <script>alert("junk");</script>
</body>
</html>
"""

TEST_MARKDOWN_EMPTY = ""

TEST_MARKDOWN_NO_H1 = """\
This document has no H1 heading.

Some content here.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

def create_corpus(root: Path) -> None:
    """Create the development corpus under *root*/data/corpus."""
    corpus = root / "data" / "corpus"

    # Markdown
    md_dir = corpus / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)
    for name, content in MARKDOWN_DOCS.items():
        (md_dir / name).write_text(content, encoding="utf-8")
        print(f"  Created: data/corpus/markdown/{name}")

    # HTML
    html_dir = corpus / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    for name, content in HTML_DOCS.items():
        (html_dir / name).write_text(content, encoding="utf-8")
        print(f"  Created: data/corpus/html/{name}")

    # PDF
    pdf_dir = corpus / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for name, (title, pages) in PDF_DOCS.items():
        pdf_bytes = _build_minimal_pdf(title, pages)
        (pdf_dir / name).write_bytes(pdf_bytes)
        print(f"  Created: data/corpus/pdf/{name}")

    # DOCX
    docx_dir = corpus / "docx"
    docx_dir.mkdir(parents=True, exist_ok=True)
    for name, (title, sections) in DOCX_DOCS.items():
        doc = _build_docx(title, sections)
        doc.save(str(docx_dir / name))
        print(f"  Created: data/corpus/docx/{name}")

    # Unsupported file (to verify skipping behaviour)
    (corpus / "markdown" / "notes.txt").write_text(
        "These are raw notes — not a supported format.\n", encoding="utf-8"
    )
    print("  Created: data/corpus/markdown/notes.txt  [unsupported — for skip testing]")


def create_test_fixtures(root: Path) -> None:
    """Create small test fixtures under *root*/tests/fixtures/."""
    fixtures = root / "tests" / "fixtures"

    (fixtures / "markdown").mkdir(parents=True, exist_ok=True)
    (fixtures / "html").mkdir(parents=True, exist_ok=True)
    (fixtures / "pdf").mkdir(parents=True, exist_ok=True)
    (fixtures / "docx").mkdir(parents=True, exist_ok=True)
    (fixtures / "unsupported").mkdir(parents=True, exist_ok=True)

    (fixtures / "markdown" / "sample.md").write_text(TEST_MARKDOWN, encoding="utf-8")
    (fixtures / "markdown" / "empty.md").write_text(TEST_MARKDOWN_EMPTY, encoding="utf-8")
    (fixtures / "markdown" / "no_h1.md").write_text(TEST_MARKDOWN_NO_H1, encoding="utf-8")
    (fixtures / "html" / "sample.html").write_text(TEST_HTML, encoding="utf-8")
    (fixtures / "unsupported" / "data.csv").write_text(
        "col1,col2\nval1,val2\n", encoding="utf-8"
    )
    (fixtures / "unsupported" / "archive.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)

    # PDF fixture
    pdf_bytes = _build_minimal_pdf(
        "Sample PDF",
        ["Page one content.\nThis is a test PDF document.", "Page two content.\nMore test text."],
    )
    (fixtures / "pdf" / "sample.pdf").write_bytes(pdf_bytes)

    # DOCX fixture
    doc = _build_docx(
        "Sample DOCX",
        [("Introduction", "This is the introduction.\n\nIt has two paragraphs.")],
    )
    doc.save(str(fixtures / "docx" / "sample.docx"))

    # Malformed PDF (not a valid PDF)
    (fixtures / "pdf" / "malformed.pdf").write_bytes(b"This is not a PDF file at all.")

    print("  Created: tests/fixtures/  (markdown, html, pdf, docx, unsupported)")


if __name__ == "__main__":
    print("\nDocuMind Fixture Generator")
    print("─" * 40)
    print("\nCreating corpus...")
    create_corpus(ROOT)
    print("\nCreating test fixtures...")
    create_test_fixtures(ROOT)
    print("\n✓ Done.\n")
