# DocuMind

> A production-oriented RAG system for technical documentation.

DocuMind is designed to demonstrate serious Applied AI / RAG engineering.
Rather than a basic "PDF → embeddings → chatbot", it implements a full
production-quality pipeline with structure-aware chunking, hybrid retrieval
(dense + keyword), cross-encoder reranking, grounded LLM generation with
inline citations, and RAGAS-based evaluation.

---

## Project Status

| Phase | Title                           | Status         |
|-------|---------------------------------|----------------|
| 0     | Architecture Decisions          | ✅ Complete    |
| 1     | Corpus & Document Ingestion     | ✅ Complete    |
| 2     | Structure-Aware Chunking        | ✅ Complete    |
| 3     | Embeddings + pgvector           | ✅ Complete    |
| 4     | Baseline Dense Retrieval        | ✅ Complete    |
| 5     | Hybrid Retrieval + RRF          | ⏳ Planned     |
| 6     | Cross-Encoder Reranking         | ⏳ Planned     |
| 7     | Grounded Generation + Citations | ⏳ Planned     |
| 8     | RAGAS Evaluation                | ⏳ Planned     |
| 9     | FastAPI + Cloud Run             | ⏳ Planned     |

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
- `embedding_dimension > 0`
- `embedding_batch_size > 0`

---

## Phase 3 — Embeddings + pgvector

### Pipeline

```
NormalizedChunk (from Phase 2)
        ↓
EmbeddingService (BAAI/bge-small-en-v1.5)
        ↓
L2-normalized 384-dimensional vector
        ↓
ChunkRepository.upsert_chunks()
        ↓
PostgreSQL + pgvector (chunks table, HNSW index)
```

### Why local BGE embeddings?

`BAAI/bge-small-en-v1.5` runs entirely locally with `sentence-transformers`.

- **No per-document API cost**: Embedding a 10,000-chunk corpus costs zero API
  calls.  Reproducibility does not depend on a vendor API staying stable.
- **Reproducible vectors**: Same text + same model → identical vector.  This
  is essential for deterministic ingestion and idempotency testing.
- **Appropriate dimensionality**: 384 dimensions is a practical balance between
  semantic quality and index size.  A 768-dimensional model (e.g. `bge-base`)
  doubles the index size for a modest recall improvement that may not matter
  for technical documentation.
- **MTEB performance**: `bge-small-en-v1.5` scores competitively on MTEB
  English semantic similarity tasks.  It is not the highest-performing model,
  but it is the correct baseline for a project that prioritizes reproducibility
  and local execution.
- **Cosine similarity**: BGE models are trained with cosine similarity.
  L2-normalizing the output before storage means cosine similarity equals
  dot product, which pgvector's HNSW index handles efficiently.

### Why pgvector?

The project already uses PostgreSQL (Phase 0 decision).  pgvector extends
PostgreSQL with a vector column type and index — no separate vector database
is needed.  This keeps the infrastructure footprint small and provenance
metadata co-located with embeddings in the same transaction.

### Why HNSW?

HNSW (Hierarchical Navigable Small World) is an approximate nearest-neighbor
index that provides O(log N) query time without a training step.

Compared to the other pgvector option (IVFFlat):
- **No training step**: IVFFlat requires running `CREATE INDEX ... WITH (lists=N)`
  on a representative sample.  HNSW works on an empty table and grows
  incrementally.
- **Better recall at lower ef_search**: For a technical documentation corpus
  that does not exceed millions of chunks, HNSW is the more practical choice.

Index parameters:
- `m=16` — maximum connections per layer (pgvector default, reasonable for
  most workloads)
- `ef_construction=64` — candidate list size during build (higher = better
  recall at build time, lower = faster build)

These parameters are starting points.  Tuning against real recall@K metrics
belongs to the evaluation phase.

### Why cosine distance?

BGE embeddings are L2-normalized before storage.  For unit-norm vectors:

```
cosine_similarity(a, b) = dot(a, b)
cosine_distance(a, b)   = 1 - dot(a, b)
```

The `vector_cosine_ops` operator class in pgvector's HNSW index directly
optimizes for this distance function.  Using inner product (`vector_ip_ops`)
would be equivalent after L2 normalization, but cosine is more conventional
and its semantics are explicit.

### Why separate embedding and persistence layers?

`EmbeddingService` and `ChunkRepository` are deliberately separate:

- **Testability**: `EmbeddingService` is tested with `FakeEmbeddingProvider`
  (no DB, no model download).  `ChunkRepository` is tested with a real DB but
  without the embedding model.
- **Future flexibility**: The embedding model can be swapped without touching
  persistence code.  The persistence schema can change without touching
  embedding code.
- **Phase clarity**: Embedding is a transformation step.  Persistence is an
  I/O step.  Mixing them would make both harder to reason about.

### Why deterministic chunk IDs?

`chunk_id = SHA-256(doc_id:chunk_index:content[:64])`

Running the ingestion pipeline twice on the same document produces the same
chunk IDs.  The database has a `UNIQUE(chunk_id)` constraint.  Re-ingesting
the same document does `INSERT ... ON CONFLICT DO UPDATE` — updating the
row in-place rather than creating a duplicate.  This makes the pipeline
idempotent by design.

### Schema design: explicit columns vs JSONB

Fields that retrieval and filtering queries will use (`document_id`,
`chunk_index`, `page`, `start_char`, `end_char`) are explicit columns.
`section_path` is JSONB because it is a variable-length list — explicit
columns would require either delimiter encoding or a join table, neither of
which simplifies retrieval at this stage.

### Running Phase 3

```bash
# 1. Start PostgreSQL + pgvector
docker compose up -d

# 2. Apply migrations
uv run alembic upgrade head

# 3. Full pipeline: ingest → chunk → embed → persist
uv run python -m documind.ingestion \
    --input data/corpus \
    --chunk \
    --embed

# With custom settings
uv run python -m documind.ingestion \
    --input data/corpus \
    --chunk \
    --chunk-size 256 \
    --embed \
    --database-url "postgresql+asyncpg://user:pass@host/db"
```

### New environment variables

| Variable | Default | Description |
|---|---|---|
| `DOCUMIND_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace model name |
| `DOCUMIND_EMBEDDING_DIMENSION` | `384` | Vector dimension (must match model) |
| `DOCUMIND_EMBEDDING_BATCH_SIZE` | `64` | Chunks per encoding batch |
| `DOCUMIND_DATABASE_URL` | Docker Compose default | PostgreSQL async URL |

### Running integration tests

```bash
# Unit tests only (no DB, no model download)
uv run pytest -m "not integration and not model_integration" -v

# Integration tests (requires docker compose up -d && alembic upgrade head)
uv run pytest tests/integration/ -m integration -v

# Real model tests (requires model download ~130MB)
uv run pytest tests/integration/test_real_embeddings.py -m model_integration -v
```

---

## Phase 4 — Baseline Dense Retrieval

### What dense retrieval does

Phase 4 establishes the first working retrieval baseline: given a natural-language
query, find the most semantically similar chunks in the database using vector
similarity search.

```
User Query
    ↓
EmbeddingService.embed_text()       ← same model as document embedding (BGE-small-en-v1.5)
    ↓
384-dimensional L2-normalized vector
    ↓
VectorRepository.search()
    ↓
pgvector <=> operator (cosine distance)
    ↓
HNSW index (from Phase 3 migration)
    ↓
Top-K (chunk_id, distance) pairs
    ↓
DenseRetriever.retrieve()           ← assembles RetrievedChunk results
    ↓
list[RetrievedChunk]                ← typed, with rank + scores + provenance
```

### Why dense retrieval is the baseline

Starting with dense retrieval alone gives us a single, measurable system.
Later phases will add:

- **Phase 5**: BM25 keyword retrieval (Baseline B) + Reciprocal Rank Fusion
- **Phase 6**: Cross-encoder reranking

By building incrementally, we can isolate the contribution of each component:

```
Baseline A   Dense only              ← Phase 4
Baseline B   Keyword only            ← Phase 5
System C     Dense + Keyword + RRF   ← Phase 5
System D     Dense + Keyword + RRF + Reranking  ← Phase 6
```

No claims about retrieval quality are made at this stage.
Formal evaluation (Recall@K, MRR, NDCG) belongs to a later evaluation phase.

### Score semantics

pgvector's `<=>` operator returns **cosine distance** (not similarity):

```
distance = 1 - cosine_similarity
range:  [0, 2]  for L2-normalized unit vectors
  0.0 = identical direction → perfect semantic match
  1.0 = orthogonal         → unrelated
  2.0 = opposite direction
```

`RetrievedChunk` exposes both:
- `distance` — raw pgvector value, lower is better, use for sorting
- `similarity = 1 - distance` — higher is better, for display

Results are always ordered by **ascending distance** (rank 1 = closest).

### Architecture

```
DenseRetriever          ← documind/retrieval/service.py
   ├── EmbeddingService ← reuses Phase 3 (no new model code)
   └── VectorRepository ← db/vector_repository.py

VectorRepository
   └── SELECT … ORDER BY embedding <=> query_vector LIMIT k

RetrievedChunk          ← documind/retrieval/models.py
   ├── chunk_id, document_id, content, section_path
   ├── page, start_char, end_char, chunk_index
   ├── distance (cosine distance, lower=better)
   ├── similarity (1-distance, higher=better)
   └── rank (1-indexed, 1=closest)
```

### Running Phase 4

```bash
# Ensure Phase 3 is set up (DB running, migration applied, corpus ingested):
docker compose up -d
uv run alembic upgrade head
uv run python -m documind.ingestion --input data/corpus --chunk --embed

# Query the index:
uv run python -m documind.retrieval --query "How do I create an API key?" --top-k 5

# Restrict to a specific document:
uv run python -m documind.retrieval --query "OAuth flow" --document-id <doc_id>

# Suppress chunk content (scores + metadata only):
uv run python -m documind.retrieval --query "API keys" --no-content
```

### New environment variables

| Variable | Default | Description |
|---|---|---|
| `DOCUMIND_RETRIEVAL_DEFAULT_TOP_K` | `5` | Default chunks per query |
| `DOCUMIND_RETRIEVAL_MAX_TOP_K` | `100` | Hard ceiling on top_k |

### Running Phase 4 tests

```bash
# Unit tests (no DB, no model):
uv run pytest tests/test_retrieval_models.py tests/test_retrieval_service.py -v

# Integration tests (requires docker compose up -d):
uv run pytest tests/integration/test_retrieval.py -m integration -v

# Real BGE model retrieval test:
uv run pytest tests/integration/test_retrieval.py -m model_integration -v
```

