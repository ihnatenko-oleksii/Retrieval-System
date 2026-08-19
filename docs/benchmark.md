# Retrieval Benchmark

This benchmark compares three retrieval configurations on a small, fixed
corpus and eval set:

- **A. Dense-only** — `dense_weight=1.0`, no BM25 contribution.
- **B. Hybrid (dense + BM25)** — the project defaults, `dense_weight=0.7` / `sparse_weight=0.3`.
- **C. Hybrid + reranking** — B plus the cross-encoder reranking stage.

## Why a purpose-built corpus

The repository ships no real document corpus (`data/` is intentionally
empty — users bring their own documents), and the original
`evals-json_example.jsonl` referenced files (`README.md`, `glossary.md`,
`architecture/dependency-injection.md`) that don't all exist in the repo, so
it wasn't runnable as a real benchmark. `docs/benchmark_corpus/` is six
short markdown documents describing this project's own retrieval,
reranking, chunking, and evaluation design, paired with
`docs/benchmark_eval.jsonl` (20 hand-written questions, each with a known
source document and expected keywords). It's small by design — enough to
demonstrate the workflow end-to-end and be read in full by a reviewer, not
a claim of statistically robust retrieval-quality numbers. A larger, more
diverse corpus would be needed for that.

## Running it

```bash
uv sync
ollama pull llama3.2   # only needed for keyword_hit_rate; skip with --skip-generation
uv run scripts/benchmark.py
```

The script ingests `docs/benchmark_corpus/` into an isolated index under
`./storage/benchmark/` (it never touches a real corpus you may already have
indexed), runs all three configurations against
`docs/benchmark_eval.jsonl`, and rewrites the **Results** section below in
place.

Retrieval-only run, with no LLM required:

```bash
uv run scripts/benchmark.py --skip-generation
```

## Metrics reported

- **recall@k** — did a chunk from the expected source appear in the top k?
- **precision@k** — what fraction of the top k chunks came from the expected source?
- **mrr** — reciprocal rank of the first relevant chunk.
- **ndcg** — rank-discounted score against the best achievable ordering of the relevant chunks found.
- **keyword_hit_rate** — fraction of expected keywords present in the generated answer (needs an LLM; omitted with `--skip-generation`).

See `app/evals/evaluator.py` for the exact implementation.

## Results

**Status: not yet run.** This sandbox has no outbound access to Hugging Face
(needed to download `intfloat/multilingual-e5-base` for dense retrieval) and
no local Ollama daemon, so `scripts/benchmark.py` could not be executed
here — running it requires network access for the embedding model and,
for `keyword_hit_rate`, a local Ollama install. What *is* verified in this
environment: the benchmark harness itself (ingestion → three configs →
`Evaluator.evaluate_cases` → markdown report) is exercised by
`tests/test_evaluator.py` with a mocked retriever/generator, so the wiring
is correct even though no real numbers were produced. Run the command above
locally to populate this section — no numbers are fabricated here.

<!-- BENCHMARK_RESULTS_START -->
_No run recorded yet — see "Status" above._
<!-- BENCHMARK_RESULTS_END -->
