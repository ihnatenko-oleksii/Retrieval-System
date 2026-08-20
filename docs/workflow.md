# Recommended workflow

The repository's validated path is intentionally small: prepare documents, ingest them with the Qwen3 + BM25 defaults, retrieve grounded passages, and optionally generate an answer. Benchmark-v4 is already final and is not a tuning set.

## 1. Prepare and inspect source documents

Keep one topic per section, preserve exact identifiers and URLs, remove duplicated boilerplate, and keep important facts close to the heading that describes them. Retrieval quality depends heavily on source quality.

## 2. Configure the validated defaults

```bash
cp .env.example .env
```

The default configuration uses `Qwen/Qwen3-Embedding-0.6B`, the generic query-only instruction, 0.7 dense / 0.3 BM25 weighted-linear fusion, and 1000/200 chunking. Reranking, rewriting, expansion, PRF, adaptive routing, and LTR are disabled.

If `EMBEDDING_MODEL`, `CHUNK_SIZE`, or `CHUNK_OVERLAP` changes, re-ingest the corpus. Existing Chroma vectors are not interchangeable across embedding models. The query-only `EMBEDDING_QUERY_INSTRUCTION` changes future query encoding but does not change stored document vectors.

## 3. Ingest and use the system

```bash
uv sync
uv run main.py ingest "./data"
uv run main.py ask "What is dependency injection?"
```

The embedding model is required for retrieval. Answer generation is separate: install and run an Ollama-compatible local model configured by `LLM_MODEL` if grounded answer generation is needed.

The same core is available through:

```bash
uv run main.py api
uv run main.py ui
```

## 4. Evaluate custom data

Create a reviewed JSONL evaluation set with graded `relevance` labels where possible, then run retrieval-only evaluation without an LLM:

```bash
uv run main.py evals "./evals-json.jsonl" --top-k 5
```

Use the optional `tuning-ui` only for a clearly scoped experiment on your own development data. Its trials are not the validated default and should not be confused with benchmark-v4.

See [docs/evaluation.md](evaluation.md) for the historical/development/final benchmark hierarchy and the exact verification commands.

## Setting up an AI coding agent

```text
Set up this Retrieval System project locally.

1. Run `uv sync`.
2. Copy `.env.example` to `.env` and review the configured embedding and LLM models.
3. Put source documents in `./data`.
4. Run `uv run main.py ingest "./data"`.
5. Run `uv run main.py ask "What is OOP?"` as a sanity check.
6. Launch `uv run main.py ui` only after the previous steps succeed.

Retrieval requires the configured Qwen3 embedding model. Generation requires
the optional local Ollama-compatible model configured by `LLM_MODEL`.
Report each command and its key output; do not run benchmark-v4 as tuning data.
```
