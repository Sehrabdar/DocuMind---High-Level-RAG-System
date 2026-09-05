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
| 2     | Structure-Aware Chunking     | ⏳ Planned     |
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
