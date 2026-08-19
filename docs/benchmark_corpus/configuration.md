# Configuration

All runtime configuration is defined as a single `pydantic-settings` model,
loaded from environment variables or a `.env` file in the project root, so
there is one source of truth for defaults across the CLI, the API, and the
Gradio UIs.

## Retrieval and storage settings

`VECTOR_DB_PATH` controls where the persistent Chroma index is stored on
disk. `EMBEDDING_MODEL` selects the sentence-transformer model used for
dense retrieval; changing it requires re-ingesting the corpus, because
stored vectors are only comparable if they came from the same model.
`RETRIEVAL_TOP_K` sets the default number of chunks returned per query.

## Hybrid, rewriting, and reranking toggles

`HYBRID_SEARCH_WEIGHTS_DENSE` and `HYBRID_SEARCH_WEIGHTS_SPARSE` set the
fusion weights, defaulting to 0.7 and 0.3. `QUERY_REWRITING_ON` and
`QUERY_EXPANSION_ON` are off by default since they add LLM latency.
`RERANKER_ON` and `RERANK_TOP_N` control the optional cross-encoder
reranking stage, also off by default.

## Per-request overrides

A `RetrievalConfig` dataclass mirrors these settings and lets any caller,
such as the Gradio chat UI, the FastAPI `/ask` endpoint, or the tuning
sweep, override the global defaults for a single request or trial without
mutating global state shared across requests.
