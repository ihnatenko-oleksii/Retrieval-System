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
| `tuning-ui` | Optional Gradio sweep UI for experimental runtime comparisons. |
| `embeddings-ui` | 2D t-SNE explorer for stored embeddings. |
| `api` | Start the FastAPI server. |

Examples:

```bash
uv run main.py ingest "./data"
uv run main.py ask "What is SOLID?" --top-k 3 --model qwen3.5:4b-mlx
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
  "model": "qwen3.5:4b-mlx"
}
```

`query` must be non-blank and `top_k` must be `>= 1`; invalid input returns
`422`. Ingesting a directory that doesn't exist returns `400`.

The API example explicitly requests five results; omitting `top_k` uses the
configured interactive default of three.

## Configuration (`.env`)

Copy `.env.example` to `.env` in the project root. All values are loaded by
`app/core/config.py` via `pydantic-settings`.

```env
# Models
# Validated retrieval default. Changing the embedding model requires re-ingestion.
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_QUERY_INSTRUCTION=Given a technical support or engineering question, retrieve passages that directly answer the question or contain the necessary implementation details.
# Generation is optional and uses a separate local Ollama-compatible model.
LLM_MODEL=qwen3.5:4b-mlx

# Storage
VECTOR_DB_PATH=./storage/chroma

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_TOP_K=3
HYBRID_SEARCH_WEIGHTS_DENSE=0.7
HYBRID_SEARCH_WEIGHTS_SPARSE=0.3
FUSION_STRATEGY=weighted_linear
ADAPTIVE_ROUTING=false
RETRIEVAL_CANDIDATE_DEPTH=20

# Rewriting / expansion
QUERY_REWRITING_ON=false
QUERY_EXPANSION_ON=false

# Reranking
RERANKER_ON=false
RERANK_TOP_N=3

# The benchmark reports @5; the interactive default returns three chunks.
# Set RETRIEVAL_TOP_K explicitly when a caller needs a different UX value.
```

If an existing `storage/chroma` index was built with another embedding model,
do not query it with the new default. Follow the [safe re-ingestion procedure](troubleshooting.md#embedding-model-or-chunking-changed)
and keep the old production indexes as a backup until the new index is checked.
The compatibility guard raises an explicit error before retrieval if the
persisted vector dimension or recorded model does not match. If you want to
keep an existing E5 index, set `EMBEDDING_MODEL=intfloat/multilingual-e5-base`
instead of switching the index to Qwen3.

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

The final independent validation is benchmark-v4; see
[docs/evaluation.md](evaluation.md) for the benchmark hierarchy and
reproduction commands.

## Extending the system

- Add new file loaders in `app/ingestion/loaders.py`
- Add new chunking strategies in `app/chunking/splitter.py`
- Add new retrieval logic in `app/retrieval/retriever.py`
- Add model providers in `app/generation/generator.py`
- Add eval metrics in `app/evals/evaluator.py`
