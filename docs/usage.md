# Usage Reference

## CLI

Show command help:

```bash
uv run main.py --help
```

| Command | Description |
|---|---|
| `ingest <dir>` | Parse documents, chunk them, and build Chroma + BM25 indexes. |
| `ask "<query>"` | One-shot QA from the terminal. |
| `chat` | Interactive terminal chat (type `exit`/`quit` to stop). |
| `evals <file.jsonl>` | Run JSONL evaluation and print metrics. |
| `ui` | Main Gradio chat UI. |
| `evals-ui` | Gradio dashboard for single-config eval runs + charts. |
| `tuning-ui` | Gradio sweep UI to find the best runtime hyperparameters. |
| `embeddings-ui` | 2D t-SNE explorer for stored embeddings. |
| `api` | Start the FastAPI server. |

Examples:

```bash
uv run main.py ingest "./data"
uv run main.py ask "What is SOLID?" --top-k 5 --model llama3.2
uv run main.py evals "./evals-json.jsonl" --top-k 5
uv run main.py api --host 0.0.0.0 --port 8000
```

The `ui` command's sidebar exposes runtime controls: `top_k`, `dense_weight`,
`reranker_on`, `rerank_top_n`, `query_rewriting_on`, `query_expansion_on`,
`llm_model`. The `tuning-ui` sweep ranks configurations by:

```text
composite = 0.6 * keyword_hit_rate + 0.25 * MRR + 0.15 * recall@K
```

## FastAPI

Base URL: `http://localhost:8000` · interactive docs: `http://localhost:8000/docs`

- `GET /health`
- `POST /ingest` — `{"directory": "./data"}`
- `POST /ask` — see below

```json
POST /ask
{
  "query": "What is dependency injection?",
  "top_k": 5,
  "model": "llama3.2"
}
```

`query` must be non-blank and `top_k` must be `>= 1`; invalid input returns
`422`. Ingesting a directory that doesn't exist returns `400`.

## Configuration (`.env`)

Create a `.env` file in the project root. All values are loaded by
`app/core/config.py` via `pydantic-settings`.

```env
# Models
# Default is multilingual E5 base (~280 MB, strong Polish+English).
# Alternatives: intfloat/multilingual-e5-small (fastest) or BAAI/bge-m3 (best, heavier).
# IMPORTANT: after changing this value, re-ingest your corpus.
EMBEDDING_MODEL=intfloat/multilingual-e5-base
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

## Evaluation file format (JSONL)

Each line is one test case:

```json
{"question":"What is dependency injection?","expected_keywords":["dependency injection","inversion of control"],"expected_source":"architecture/dependency-injection.md"}
{"question":"What does CI stand for?","expected_keywords":["continuous integration"],"expected_source":"glossary.md"}
```

Metrics reported: `recall@k`, `precision@k`, `mrr`, `ndcg`, `keyword_hit_rate`.
See [docs/benchmark.md](benchmark.md) for a worked example and
[docs/architecture.md](architecture.md#evaluation) for how each metric is
computed.

## Extending the system

- Add new file loaders in `app/ingestion/loaders.py`
- Add new chunking strategies in `app/chunking/splitter.py`
- Add new retrieval logic in `app/retrieval/retriever.py`
- Add model providers in `app/generation/generator.py`
- Add eval metrics in `app/evals/evaluator.py`
