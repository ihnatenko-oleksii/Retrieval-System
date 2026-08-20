# Benchmark history and final validation

> **Final result:** `benchmark-v4` is the authoritative independent validation
> for the current retrieval recommendation. Start with the [evaluation
> hierarchy](evaluation.md), then read the [Phase 5 final report](retrieval_results/phase5_final_results.md)
> and [benchmark-v4 manifest](benchmark_v4/manifest.json).

## Evaluation hierarchy

1. **Historical original benchmark** — the Atlas/support corpus and harness
   below document the first retrieval-system comparisons and remain preserved
   for provenance.
2. **Development experiments** — [benchmark-v2](benchmark_v2/),
   [benchmark-v3](benchmark_v3/), and the Phase 3–5 result files record
   held-out development comparisons, selection decisions, and rejected
   approaches. They are not the final independent validation.
3. **Final benchmark-v4** — the sealed 50-document, 125-query evaluation is
   the current evidence for the recommended Qwen3 hybrid configuration. Its
   [manifest](benchmark_v4/manifest.json) and [Phase 5 report](retrieval_results/phase5_final_results.md)
   define the corpus, labels, and recorded metrics.

The historical harness and numbers below are intentionally retained; they are
not rewritten to look like benchmark-v4.

## Historical benchmark harness

This benchmark compares three retrieval configurations on the same indexed
corpus and evaluation cases:

- **A. Dense-only** — dense vector retrieval with dense_weight=1.0 and no
  BM25 query.
- **B. Hybrid (dense + BM25)** — score fusion with
  dense_weight=0.7 / sparse_weight=0.3.
- **C. Hybrid + reranking** — B plus the configured cross-encoder reranker.

The benchmark is an evaluation harness, not a leaderboard. It keeps the
corpus, questions, labels, embedding model, top-k, chunking settings, and
retrieval configuration visible so a result can be reproduced and challenged.

## Historical corpus and evaluation set

The redesigned corpus is a deliberately confusable Atlas support knowledge
base mixed with the original retrieval-system design notes:

- 34 documents and 100 chunks with the default chunk_size=1000 and
  chunk_overlap=200;
- 60 questions, split evenly across five categories: 12 exact-terminology,
  12 paraphrased-semantic, 12 ambiguous, 12 fine-grained, and 12
  multiple-relevant cases;
- repeated concepts across authentication, API behavior, ingestion, search,
  billing, workflows, and operations, including similar retry, timeout,
  identity, expiry, permission, and status-code language.

The corpus is large enough to create real candidate competition while still
being practical to ingest locally. The cases were written to ask normal
support and engineering questions; categories describe the information need,
not a desired winner. If dense-only remains strongest on these cases, that is
the result to report.

## Ground-truth format

New cases use a relevance object whose keys are stable chunk IDs and whose
values are graded gains:

~~~json
{
  "id": "multi_relevant-12",
  "category": "multiple_relevant",
  "question": "How should a webhook receiver verify, deduplicate, and replay an event?",
  "relevance": {
    "atlas/ingestion-webhooks.md::0": 3,
    "atlas/ingestion-webhooks.md::1": 3,
    "atlas/ingestion-webhooks.md::2": 2
  }
}
~~~

The ID is the corpus-relative filename plus ::chunk_index, and is stored in
chunk metadata during ingestion. A gain of 3 is the most directly useful
chunk, while lower positive gains identify related chunks that are still
legitimately relevant. expected_chunk_ids is also supported as a binary
shorthand. Existing evaluation files using expected_source continue to work,
but source-only labels provide only binary, document-level relevance.

For chunk-labeled cases:

- **Recall@K** is the fraction of all positively labeled chunks retrieved in
  the top K.
- **Precision@K** is the number of relevant retrieved results divided by K.
- **MRR** is the reciprocal rank of the first relevant chunk.
- **nDCG** uses the graded gains and normalizes against the ideal ordering of
  all labeled relevant chunks, not only the chunks found by the system.

keyword_hit_rate remains an optional answer-generation metric. It checks
whether expected terms appear in the generated answer and requires the
configured local Ollama model; it is omitted by --skip-generation.

## Running the historical harness

For investigating this historical harness, the retrieval-only comparison is
the practical default:

~~~bash
uv run scripts/benchmark.py --skip-generation
~~~

This ingests docs/benchmark_corpus/ into the isolated
storage/benchmark/ index, evaluates A/B/C with top_k=5, writes the latest
table into this document, and appends a machine-readable record to
docs/benchmark_history.jsonl. The index is rebuilt by default so a run
does not accidentally reuse a different corpus or embedding model.

To choose a different top-k:

~~~bash
uv run scripts/benchmark.py --skip-generation --top-k 10
~~~

The slower end-to-end run also calls the configured LLM:

~~~bash
uv run scripts/benchmark.py
~~~

The configured generation model is qwen3.5:4b-mlx by default in
app/core/config.py and can be overridden with LLM_MODEL; pull the value
configured on the machine before the full run. Retrieval-only evaluation does
not require Ollama.

The report records the embedding model, generation model, top-k, chunk size,
chunk overlap, BM25 weights, rerank candidate setting, reranker model, corpus
size, and question-category counts. Use --no-history for a local trial that
should not append to the history file.

## Preserved first run

The first successful run is intentionally preserved in
docs/benchmark_history.jsonl as the first-small-benchmark record. It used
the original six-document, 12-chunk corpus and 20 source-level questions:

| Configuration | recall@5 | precision@5 | mrr | ndcg | keyword_hit_rate |
|---|---:|---:|---:|---:|---:|
| A. Dense-only | 1.0 | 0.39 | 1.0 | 0.9722 | 0.975 |
| B. Hybrid (dense + BM25) | 1.0 | 0.39 | 1.0 | 0.9778 | 0.95 |
| C. Hybrid + reranking | 1.0 | 0.38 | 1.0 | 0.9677 | 0.95 |

That first small benchmark saturates Recall@5 and MRR for every
configuration. Its primary value is verifying the end-to-end harness
(ingestion, indexing, three retrieval paths, evaluation, optional generation,
and report writing); it is not discriminative evidence that hybrid retrieval
or reranking improves quality.

## Latest historical recorded run

The block below is generated in place by scripts/benchmark.py. It is kept
separate from the preserved first-run history so later runs cannot erase the
original numbers.

<!-- BENCHMARK_RESULTS_START -->
_Last run: 2026-08-19 11:24 UTC_
_Corpus: `docs/benchmark_corpus` (34 documents, 100 chunks, 60 eval cases)._
_Question categories: ambiguous=12, exact_terminology=12, fine_grained=12, multiple_relevant=12, paraphrased_semantic=12._
_Settings: embedding_model=`intfloat/multilingual-e5-base`, top_k=5, chunk_size=1000, chunk_overlap=200, dense_weight=0.7, sparse_weight=0.3, rerank_top_n=5, reranker_model=`cross-encoder/ms-marco-MiniLM-L-6-v2`, llm_model=`qwen3.5:4b-mlx`._
_Configurations: A dense=1.0/sparse=0.0/rerank=off; B dense=0.7/sparse=0.3/rerank=off; C dense=0.7/sparse=0.3/rerank=on._
_keyword_hit_rate omitted (`--skip-generation`): no LLM calls were made; configured generation model is `qwen3.5:4b-mlx`._

| Configuration | recall@5 | precision@5 | mrr | ndcg |
|---|---|---|---|---|
| A. Dense-only | 0.7892 | 0.3067 | 0.8783 | 0.7705 |
| B. Hybrid (dense + BM25) | 0.8169 | 0.3233 | 0.9047 | 0.8109 |
| C. Hybrid + reranking | 0.8175 | 0.3233 | 0.8625 | 0.8021 |
<!-- BENCHMARK_RESULTS_END -->

## Final independent validation

Do not use benchmark-v4 to tune a new configuration. It was sealed before
final scoring and is reported once as independent validation. The development
selection record is in [phase5_development.md](retrieval_results/phase5_development.md);
the final comparison and uncertainty analysis are in
[phase5_final_results.md](retrieval_results/phase5_final_results.md), with the
corpus and query-label contract in [benchmark_v4/manifest.json](benchmark_v4/manifest.json).
