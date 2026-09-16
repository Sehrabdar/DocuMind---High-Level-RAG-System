# DocuMind

> A production-oriented RAG system for technical documentation.

DocuMind is designed to demonstrate serious Applied AI / RAG engineering.
Rather than a basic "PDF → embeddings → chatbot", it implements a full
production-quality pipeline with structure-aware chunking, hybrid retrieval
(dense + keyword), cross-encoder reranking, grounded LLM generation with
inline citations, and RAGAS-based evaluation.

---

## Project Status

| Phase | Title                        | Status         |
|-------|------------------------------|----------------|
| 0     | Architecture Decisions       | ✅ Complete    |
| 1     | Corpus & Document Ingestion  | ✅ Complete    |
| 2     | Structure-Aware Chunking     | ✅ Complete    |
| 3     | Embeddings + pgvector        | ⏳ Planned     |
| 4     | Hybrid Retrieval + RRF       | ⏳ Planned     |
| 5     | Cross-Encoder Reranking      | ⏳ Planned     |
| 6     | Grounded Generation + Citations | ⏳ Planned  |
| 7     | RAGAS Evaluation             | ⏳ Planned     |
| 8     | FastAPI + Cloud Run          | ⏳ Planned     |

---

## Architecture

```
Documents
    ↓
Parsing / Normalization          ← Phase 1 (this phase)
    ↓
Structure-Aware Chunking         ← Phase 2
    ↓
Embeddings (BAAI/bge-small-en-v1.5, dim=384)  ← Phase 3
    ↓
PostgreSQL + pgvector (HNSW)     ← Phase 3
    ↓
Dense + Keyword Retrieval        ← Phase 4
    ↓
RRF Fusion                       ← Phase 4
    ↓
Cross-Encoder Reranking          ← Phase 5
    ↓
Grounded LLM Generation          ← Phase 6
    ↓
Inline Citations                  ← Phase 6
    ↓
Evaluation / Observability        ← Phase 7
```

---

## Phase 0 — Technology Decisions

| Decision             | Choice                          |
|----------------------|---------------------------------|
| Language             | Python 3.12                     |
| Package manager      | `uv`                            |
| Project format       | `pyproject.toml` + `uv.lock`    |
| Database             | PostgreSQL 16 + pgvector        |
| Embedding model      | `BAAI/bge-small-en-v1.5` (384d) |
| Embedding runtime    | `sentence-transformers`         |
| Reranker             | `BAAI/bge-reranker-base`        |
| LLM provider         | Anthropic Claude API            |
| LLM framework        | Custom (not LangChain/LlamaIndex for pipeline) |
| Document loading     | `pypdf`, `beautifulsoup4`, `python-docx` |
| Config               | `pydantic-settings`             |
| CLI                  | `click` + `rich`                |
| Testing              | `pytest`                        |
| Containerisation     | Docker Compose (local dev)      |

---

## Phase 1 — Ingestion Architecture

The ingestion layer is intentionally minimal and focused.

```
Raw File
    ↓
LoaderDispatcher          ← detects format by extension
    ↓
Format-specific Loader    ← MarkdownLoader / PDFLoader / HTMLLoader / DocxLoader
    ↓
NormalizedDocument        ← canonical internal representation
```

### Separation of Concerns

| Responsibility     | Module                         |
|--------------------|--------------------------------|
| Format dispatch    | `documind/ingestion/loaders.py` |
| Normalization      | Part of each Loader            |
| Data model         | `documind/ingestion/models.py` |
| Directory pipeline | `documind/ingestion/pipeline.py` |
| CLI                | `documind/ingestion/__main__.py` |
| Configuration      | `config.py`                    |

### `NormalizedDocument` model

```python
class NormalizedDocument(BaseModel):
    id: str              # SHA-256 of resolved source path (deterministic)
    source: str          # Absolute path to source file
    filename: str        # Basename
    file_type: str       # 'markdown' | 'pdf' | 'html' | 'docx'
    title: str | None    # Best-effort title extraction
    content: str         # Normalized text content
    metadata: DocumentMetadata
```

LlamaIndex is **not** used in Phase 1. Document loading uses `pypdf`,
`beautifulsoup4`, and `python-docx` directly, providing full control and
no framework leakage.

---

## Supported Formats

| Extension        | Parser         | Notes                                         |
|------------------|----------------|-----------------------------------------------|
| `.md`            | Built-in       | Heading markers preserved for Phase 2 chunking |
| `.pdf`           | pypdf          | Pages separated by `\f`; page count in metadata |
| `.html`, `.htm`  | BeautifulSoup4 | Nav/scripts/styles stripped; semantic content extracted |
| `.docx`          | python-docx    | Paragraph order preserved; tables as pipe-delimited rows |

---

## Project Structure

```
documind/
├── ingestion/
│   ├── __init__.py
│   ├── __main__.py       ← CLI entry point
│   ├── loaders.py        ← Format-specific loaders + dispatcher
│   ├── models.py         ← NormalizedDocument + DocumentMetadata
│   └── pipeline.py       ← Directory ingestion + IngestionResult
│
├── retrieval/            ← Phase 3 (stub)
├── reranking/            ← Phase 5 (stub)
├── generation/           ← Phase 6 (stub)
├── evaluation/           ← Phase 7 (stub)
├── observability/        ← Phase 7 (stub)
└── api/                  ← Phase 8 (stub)

data/
└── corpus/
    ├── markdown/         ← .md documents
    ├── pdf/              ← .pdf documents
    ├── html/             ← .html documents
    └── docx/             ← .docx documents

tests/
├── conftest.py
├── fixtures/             ← Small deterministic test files
├── test_discovery.py
├── test_loaders.py
├── test_normalization.py
├── test_ids.py
├── test_errors.py
└── test_pipeline.py

scripts/
└── create_fixtures.py    ← Generates corpus + test fixtures (no internet)

config.py
pyproject.toml
docker-compose.yml
```

---

## Getting Started

### Prerequisites

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager

### Install

```bash
uv sync
```

### Generate Corpus and Test Fixtures

```bash
uv run python scripts/create_fixtures.py
```

This creates:
- `data/corpus/` — representative mixed-format technical documentation
- `tests/fixtures/` — small deterministic files for unit tests

No internet access required. The script is fully deterministic.

---

## Running Ingestion

```bash
uv run python -m documind.ingestion --input data/corpus
```

### Options

```
--input PATH          Corpus directory (default: DOCUMIND_CORPUS_DIR or data/corpus)
--log-level LEVEL     DEBUG | INFO | WARNING | ERROR (default: INFO)
--fail-fast           Abort on first document failure (default: best-effort)
--help                Show help
```

### Example Output

```
╭─────────────────────╮
│  DocuMind Ingestion  │
╰─────────────────────╯
  Input: /home/user/documind/data/corpus

  Discovered           10
  Supported             9
  Skipped (unsupported) 1

  Ingested              9
  Failed                0

  Format     Documents
  ──────────────────
  Docx               1
  Html               2
  Markdown           3
  Pdf                2  (wait — we have 2 PDFs)

  Total characters:  18,432

✓ Ingestion complete
```

---

## Running Tests

```bash
# Generate test fixtures first (required once)
uv run python scripts/create_fixtures.py

# Run the full test suite
uv run pytest

# With coverage
uv run pytest --cov=documind --cov=config
```

---

## Configuration

Settings are read from environment variables (prefix: `DOCUMIND_`) or a `.env` file.

```bash
cp .env.example .env
```

| Variable              | Default       | Description                          |
|-----------------------|---------------|--------------------------------------|
| `DOCUMIND_CORPUS_DIR` | `data/corpus` | Default corpus root for CLI          |
| `DOCUMIND_LOG_LEVEL`  | `INFO`        | Logging verbosity                    |
| `DOCUMIND_DATA_DIR`   | `data`        | Root data directory                  |

---

## Error Handling Philosophy

The ingestion pipeline uses **best-effort** mode by default:

- A bad document is **recorded** in `IngestionResult.failed` and logged.
- Other documents continue to be ingested normally.
- The CLI exits with code `2` if any failures occurred.

Use `--fail-fast` to abort immediately on the first failure.

This reflects real-world document pipelines where malformed files should not
prevent an entire corpus from being indexed.

---

## Known Limitations

- **PDF text quality**: pypdf performs best-effort text extraction. Scanned/image-only PDFs will produce empty content. OCR is out of scope.
- **HTML boilerplate**: The heuristic for stripping navigation/junk works well for standard documentation sites but may be imperfect for unusual HTML structures.
- **DOCX tables**: Simple tables are extracted as pipe-delimited rows. Merged cells and nested tables are not handled.
- **Document IDs**: IDs are based on the resolved absolute path, so moving a file changes its ID. This is intentional — a moved file is treated as a new document for database persistence purposes.

---

## Explicitly Deferred to Phase 2

The following are **intentionally not implemented** in Phase 1:

- Semantic / structure-aware chunking
- Chunk-level metadata (section, page range, chunk index, char offsets)
- Citation provenance tracking
- Token counting / embedding preparation
- Any database operations
- Any retrieval, reranking, or generation logic

---

## Phase 2 — Structure-Aware Chunking

### Why structure-aware chunking?

Retrieval quality depends on the units being retrieved.  A chunk that contains
exactly one coherent concept retrieves better than an arbitrary 500-character
window that might begin mid-sentence and end mid-table.

Technical documentation has natural semantic boundaries — headings, procedures,
API sections, configuration options.  Phase 2 exploits those boundaries.

### Why not simple fixed-size splitting?

Fixed-size splitting produces retrieval units with no semantic coherence.  A
chunk may begin at character 1023 of a long section, contain half an API
description and half of an unrelated example, and carry no information about
where in the document it came from.

Structure-aware chunking keeps naturally sized sections intact and only
subdivides when a section is genuinely too large.

### Why overlap?

When a large section must be subdivided, the split boundary creates an
artificial break in the middle of connected prose.  A 50-token overlap ensures
that a sentence or procedure step that happens to fall at a boundary is still
intelligible in both the preceding and following chunk.

Overlap is applied **only within a section** — never between independent
sections.  Inserting section A content into the start of section B's chunks
would corrupt section path metadata.

### Why chunk metadata?

Later stages need traceable retrieval and citations.  When a user asks a
question and receives an answer grounded in chunk 14, the system must be able
to answer: _Which document?  Which section?  Which page?_

`NormalizedChunk` carries:
- `document_id` and `document_title` — document-level identity
- `section_path` — hierarchical breadcrumb (Markdown only in Phase 2)
- `page` / `page_start` / `page_end` — page provenance (PDF only)
- `start_char` / `end_char` — character offsets into the normalized content
- `chunk_index` — deterministic source order

### Why token-aware limits?

Model context and retrieval budgets are ultimately token-based, not
character-based.  A 512-character limit produces wildly different chunk sizes
for dense technical prose vs. heavily formatted Markdown.  A 512-token limit
is consistent regardless of formatting density.

The token limit uses `tiktoken cl100k_base`, which is already a transitive
dependency via `llama-index-core`.  This encoding is a standard proxy for RAG
token budgeting; it is not identical to the BAAI/bge-small-en-v1.5 tokenizer
but is accurate enough for boundary decisions.

### Chunking strategy

```
NormalizedDocument
    ↓
Format-specific structure detection
    │
    ├── Markdown: walk lines, detect # / ## / ### headings,
    │   maintain heading stack → sections with section_path
    │
    ├── PDF: split on \f page separators → sections with page number
    │
    └── HTML / DOCX: flat (heading structure not available from Phase 1)
    ↓
For each section:
    if token_count ≤ chunk_size → emit 1 chunk
    else                        → split with overlap → emit N chunks
    ↓
NormalizedChunk (deterministic IDs, offsets, provenance)
```

### Example: Authentication hierarchy

Given this Markdown document:

```
# Authentication

Authentication allows applications to verify user identity.

## API Keys

API keys are used for server-to-server authentication.

### Creating an API Key

To create an API key, send a POST request to /api/keys...

### Revoking an API Key

API keys can be revoked from the dashboard...

## OAuth

OAuth provides delegated authorization...
```

Chunk metadata produced:

```
Document
  └── Authentication
       ├── section_path=["Authentication"]           → chunk 0
       ├── API Keys
       │    ├── section_path=["Authentication","API Keys"]             → chunk 1
       │    ├── Creating an API Key
       │    │    └── section_path=["Authentication","API Keys","Creating an API Key"]  → chunk 2
       │    └── Revoking an API Key
       │         └── section_path=["Authentication","API Keys","Revoking an API Key"] → chunk 3
       └── OAuth
            └── section_path=["Authentication","OAuth"]               → chunk 4
```

Each chunk also carries: `document_id`, `document_title`, `chunk_index`,
`start_char`, `end_char`, and `token_count`.

### Format-specific notes

| Format   | Section detection | Page provenance | Known limitation |
|----------|-------------------|-----------------|------------------|
| Markdown | `#`/`##`/`###` heading stack | N/A | None |
| PDF      | Form-feed (`\f`) page splits | ✅ 1-indexed | No heading hierarchy |
| HTML     | Flat (no hierarchy) | N/A | Phase 1 `get_text()` discards tags |
| DOCX     | Flat (no hierarchy) | N/A | Phase 1 joins `para.text`, discards styles |

HTML and DOCX heading hierarchy requires Phase 1 loaders to emit structured
intermediates rather than plain text.  This is a documented tradeoff, not an
oversight.

### `NormalizedChunk` model

```python
class NormalizedChunk(BaseModel):
    # Identity
    chunk_id: str           # SHA-256(doc_id:chunk_index:content[:64])
    document_id: str
    chunk_index: int        # 0-based, source order

    # Content
    content: str
    token_count: int

    # Structural provenance
    section_path: list[str]  # ["Auth", "OAuth", "Tokens"]
    document_title: str | None

    # Location provenance
    start_char: int
    end_char: int
    page: int | None         # PDF: 1-indexed page number
    page_start: int | None   # PDF: start page for multi-page chunks
    page_end: int | None     # PDF: end page for multi-page chunks
```

---

## Running Chunking

```bash
# Ingest and chunk in one step
uv run python -m documind.ingestion --input data/corpus --chunk

# With custom token limits
uv run python -m documind.ingestion --input data/corpus --chunk --chunk-size 256 --chunk-overlap 25
```

### Chunk CLI options

```
--chunk               Run chunking after ingestion and print chunk stats
--chunk-size INT      Max tokens per chunk (default: 512)
--chunk-overlap INT   Overlap tokens between sub-chunks (default: 50)
```

---

## Configuration

Settings are read from environment variables (prefix: `DOCUMIND_`) or a `.env` file.

```bash
cp .env.example .env
```

| Variable                | Default       | Description                          |
|-------------------------|---------------|--------------------------------------|
| `DOCUMIND_CORPUS_DIR`   | `data/corpus` | Default corpus root for CLI          |
| `DOCUMIND_LOG_LEVEL`    | `INFO`        | Logging verbosity                    |
| `DOCUMIND_DATA_DIR`     | `data`        | Root data directory                  |
| `DOCUMIND_CHUNK_SIZE`   | `512`         | Max tokens per chunk                 |
| `DOCUMIND_CHUNK_OVERLAP`| `50`          | Overlap tokens between sub-chunks    |

Validation rules (enforced at startup):
- `chunk_size > 0`
- `chunk_overlap >= 0`
- `chunk_overlap < chunk_size`

