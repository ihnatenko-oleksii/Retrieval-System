![Retrieval augmented generation (RAG)](retrieval-system.png)

# Retrieval System (Local-First Production RAG)

A modular, local-first RAG application in Python that supports:
- multi-format ingestion
- dense + lexical hybrid retrieval
- optional query rewriting/expansion
- optional reranking
- grounded answer generation with citations
- CLI, FastAPI, and Gradio UI

This project is designed for practical local use with `Ollama` and persistent local storage.

## Features

- Ingest and index:
  - `.txt`, `.md`
  - `.pdf`
  - `.docx`
  - `.pptx`
  - `.csv`
  - `.json`
  - source code files (`.py`, `.java`, `.js`, `.ts`, `.yml`, `.yaml`, `.xml`, `.properties`)
- Recursive folder ingestion with metadata per chunk
- Configurable chunking (`chunk_size`, `chunk_overlap`)
- Dense retrieval (`ChromaDB` + `sentence-transformers`)
- Sparse retrieval (`BM25`) and hybrid score fusion
- Query rewriting and query expansion switches
- Optional reranking (cross-encoder)
- Conversational RAG (chat history aware)
- Evaluation pipeline (`Recall@K`, `Precision@K`, `MRR`, `nDCG`, keyword hit rate)
- Interfaces:
  - CLI
  - FastAPI
  - Gradio web UI

## Project Structure

```text
app/
  api/           # FastAPI endpoints
  chunking/      # Chunk splitting logic
  core/          # Config + shared data models
  embeddings/    # Embedding adapters
  evals/         # Evaluation pipeline
  generation/    # LLM prompting and answer generation
  ingestion/     # File loaders and ingestion pipeline
  retrieval/     # Retriever, rewriter, reranker
  ui/            # Gradio application
  vector_store/  # Chroma + BM25 stores
data/            # Put your source documents here
storage/         # Persistent indexes (Chroma + BM25)
main.py          # CLI entrypoint
```

## Requirements

- Python `3.12+`
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) installed and running locally
- One local Ollama model pulled (default: `llama3.2`)

## Quick Start

### 1) Install dependencies

```bash
uv sync
```

### 2) Start Ollama and pull a model

```bash
ollama pull llama3.2
```

### 3) Ingest your documents
Put your source documents in `./data`.

```bash
uv run main.py ingest "./data"
```

### 4) Ask a question

```bash
uv run main.py ask "What is OOP?"
```

## CLI Usage

Show command help:

```bash
uv run main.py --help
```

### Ingest

```bash
uv run main.py ingest "./data/DatasetAboutProgramming"
```

### Ask (single query)

```bash
uv run main.py ask "What is SOLID?" --top-k 5 --model llama3.2
```

### Chat (multi-turn conversational RAG)

```bash
uv run main.py chat
```

Type `exit` or `quit` to stop.

### Run evals

```bash
uv run main.py evals "./data/evals.jsonl" --top-k 5
```

### Run API

```bash
uv run main.py api --host 0.0.0.0 --port 8000
```

### Run UI

```bash
uv run main.py ui
```

## FastAPI Endpoints

Base URL: `http://localhost:8000`

- `GET /health`
- `POST /ingest`
- `POST /ask`

Interactive docs:
- `http://localhost:8000/docs`

### Example: `POST /ask`

```json
{
  "query": "What is SOW?",
  "top_k": 5,
  "model": "llama3.2"
}
```

## Configuration (`.env`)

Create a `.env` file in project root:

```env
# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama3.2

# Storage
VECTOR_DB_PATH=./storage/chroma

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_TOP_K=5
HYBRID_SEARCH_WEIGHTS_DENSE=0.7
HYBRID_SEARCH_WEIGHTS_SPARSE=0.3

# Rewriting / expansion
QUERY_REWRITING_ON=false
QUERY_EXPANSION_ON=false

# Reranking
RERANKER_ON=false
RERANK_TOP_N=5
```

Config is loaded by `app/core/config.py` via `pydantic-settings`.

## Evaluation File Format (JSONL)

Each line should be a JSON object:

```json
{"question":"What is SOW?","expected_keywords":["System Obsługi Wydziałów"],"expected_source":"Słownik - wszystkie nasze systemy.md"}
{"question":"What is CRD?","expected_keywords":["Centralny Rejestr Dokumentów"],"expected_source":"CRD.md"}
```

Current eval metrics:
- `recall@k`
- `precision@k`
- `mrr`
- `ndcg`
- `keyword_hit_rate`

## How Retrieval Works

1. Load and chunk documents with metadata.
2. Store embeddings in Chroma (dense retrieval).
3. Store tokenized corpus in BM25 (lexical retrieval).
4. Optionally rewrite and expand query.
5. Retrieve dense + sparse candidates.
6. Fuse scores (hybrid ranking), optional rerank.
7. Build grounded prompt and generate with Ollama.
8. Return answer + source citations.

## Troubleshooting

### 1) "No sentence-transformers model found..." / HF proxy errors

If your environment blocks model downloads:
- ensure internet/proxy access to Hugging Face, or
- pre-download/cache the embedding model, then rerun.

### 2) UI chat errors after multiple messages

This project normalizes chat history content before sending to the model.  
If you still hit a UI issue:
- stop UI
- rerun: `uv run main.py ui`
- share the latest traceback from terminal

### 3) Acronym definitions (example: "SOW") not found

Checklist:
- Re-ingest after data/code changes:
  ```bash
  uv run main.py ingest "./data/CCF Obsidian copy"
  ```
- Increase retrieval depth:
  ```bash
  uv run main.py ask "Co to jest SOW?" --top-k 8
  ```
- Keep glossary files in supported text formats (`.md`, `.txt`) with clean `ACRONYM - definition` lines.

### 4) Duplicate or stale retrieval results

Indexes are persisted in `./storage`.  
If needed, clear and rebuild:

```bash
rm -rf ./storage/chroma ./storage/bm25_index.pkl
uv run main.py ingest "./data"
```

## Extending the System

- Add new file loaders in `app/ingestion/loaders.py`
- Add new chunking strategies in `app/chunking/splitter.py`
- Add new retrieval logic in `app/retrieval/retriever.py`
- Add model providers in `app/generation/generator.py`
- Add eval metrics in `app/evals/evaluator.py`

## Notes

- Local-first: all vector/sparse indexes are persisted on disk.
- This codebase is structured to be extended toward stronger production usage (better reranker, richer evals, auth, observability, etc.).
