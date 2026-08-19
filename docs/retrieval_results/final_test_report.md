# Frozen TEST Baseline vs Final

Command: `uv run scripts/retrieval_experiments.py --offline-models --include-llm --llm-model qwen2.5:3b`
Frozen TEST cases: 20.

## Configuration

- Baseline: `baseline-dense-e5` — `intfloat/multilingual-e5-base`, dense-only, no reranking, rewriting, or expansion.
- Final: `bge-m3-hybrid-static+query-rewrite` — `BAAI/bge-m3`, chunk `1000/200`, fusion `weighted_linear`, weights `0.7/0.3`, candidate depth `20`, reranking `False`, query rewriting `True`, query expansion `False`.

## TEST metrics

| Configuration | Recall@5 | Precision@5 | MRR | nDCG | Seconds |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.7475 | 0.28 | 0.825 | 0.7617 | 1.756 |
| Final | 0.8058 | 0.31 | 0.875 | 0.7999 | 14.2799 |

## Relative improvement

| Metric | `(new - baseline) / baseline * 100` |
|---|---:|
| recall@5 | 7.7993% |
| precision@5 | 10.7143% |
| mrr | 6.0606% |
| ndcg | 5.0151% |

## Effect of major optimizations (DEV)

Selection and deltas in this section use DEV only; TEST values are never used for tuning.

| Optimization | Configuration | Status | DEV nDCG | Δ nDCG vs dense baseline | DEV MRR |
|---|---|---|---:|---:|---:|
| Baseline | baseline-dense-e5 | ok | 0.7748 | 0.0% | 0.905 |
| Static linear hybrid | linear-static-d0.7-s0.3 | ok | 0.8159 | 5.3046% | 0.9425 |
| Query-adaptive hybrid | hybrid-adaptive-linear-d0.7-s0.3 | ok | 0.8243 | 6.3887% | 0.9425 |
| RRF | hybrid-static-rrf | ok | 0.7903 | 2.0005% | 0.8883 |
| Weighted RRF | hybrid-static-weighted-rrf | ok | 0.8035 | 3.7042% | 0.9175 |
| BGE-M3 dense | bge-m3-dense | ok | 0.8186 | 5.6531% | 0.9258 |
| BGE-M3 hybrid | bge-m3-hybrid-static | ok | 0.8301 | 7.1373% | 0.9238 |
| Retrieval depth 30 | hybrid-static-depth-30 | ok | 0.8184 | 5.6273% | 0.9425 |
| MiniLM rerank pool 10 | hybrid-minilm-reranker-pool-10 | ok | 0.8013 | 3.4202% | 0.8688 |
| MiniLM rerank pool 20 | hybrid-minilm-reranker-pool-20 | ok | 0.7977 | 2.9556% | 0.8625 |
| MiniLM rerank pool 30 | hybrid-minilm-reranker-pool-30 | ok | 0.7949 | 2.5942% | 0.8583 |
| MiniLM rerank pool 50 | hybrid-minilm-reranker-pool-50 | ok | 0.7939 | 2.4652% | 0.8583 |
| BGE reranker pool 10 | hybrid-bge-reranker-pool-10 | error | - | - | - |
| Chunk 400/80 | hybrid-chunks-400-80 | invalid_label_mapping | - | - | - |
| Chunk 600/100 | hybrid-chunks-600-100 | invalid_label_mapping | - | - | - |
| Chunk 800/150 | hybrid-chunks-800-150 | invalid_label_mapping | - | - | - |
| Chunk 1200/200 | hybrid-chunks-1200-200 | invalid_label_mapping | - | - | - |
| LLM query-rewrite | bge-m3-hybrid-static+query-rewrite | ok | 0.8313 | 7.2922% | 0.9313 |
| LLM query-expansion | bge-m3-hybrid-static+query-expansion | ok | 0.8176 | 5.524% | 0.9396 |
| LLM rewrite-and-expansion | bge-m3-hybrid-static+rewrite-and-expansion | ok | 0.813 | 4.9303% | 0.9271 |

## Recommended CV bullet

> Improved retrieval nDCG by 5.0151% (0.7617 to 0.7999) and MRR by 6.0606% (0.825 to 0.875) on a frozen 20-query TEST set, selecting BAAI/bge-m3 with weighted_linear dense/BM25 fusion (0.7/0.3) plus query rewriting from a 40-query DEV split; latency was 0.714s/query versus 0.0878s/query for the dense baseline.

## Category breakdown

| Category | Baseline nDCG | Final nDCG | Baseline MRR | Final MRR | Cases |
|---|---:|---:|---:|---:|---:|
| ambiguous | 0.2243 | 0.5078 | 0.375 | 0.75 | 4 |
| exact_terminology | 1.0 | 1.0 | 1.0 | 1.0 | 4 |
| fine_grained | 0.9793 | 0.9077 | 1.0 | 0.875 | 4 |
| multiple_relevant | 0.8965 | 0.8508 | 1.0 | 1.0 | 4 |
| paraphrased_semantic | 0.7085 | 0.7331 | 0.75 | 0.75 | 4 |

Best final category by nDCG: `exact_terminology` (1.0).
Worst final category by nDCG: `ambiguous` (0.5078).

## Failed or blocked experiments

| Configuration | Status | Reason |
|---|---|---|
| hybrid-bge-reranker-pool-10 | error | RuntimeError: Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-bge-reranker-pool-20 | error | RuntimeError: Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-bge-reranker-pool-30 | error | RuntimeError: Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-bge-reranker-pool-50 | error | RuntimeError: Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-chunks-400-80 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: content changed for exact-01: atlas/api-pagination.md::2, exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/ingestion-dedup.md::1, exact-06: atlas/api-retries.md::0, exact-07: atlas/access-service-accounts.md::1 ... |
| hybrid-chunks-600-100 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: content changed for exact-01: atlas/api-pagination.md::2, exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/ingestion-dedup.md::1, exact-07: atlas/access-service-accounts.md::1, exact-08: atlas/operations-queue.md::1 ... |
| hybrid-chunks-800-150 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: content changed for exact-01: atlas/api-pagination.md::2, exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/ingestion-dedup.md::1, exact-07: atlas/access-service-accounts.md::1, exact-08: atlas/operations-queue.md::1 ... |
| hybrid-chunks-1200-200 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: ambiguous-03: atlas/access-mfa.md::3, fine_grained-01: atlas/access-sessions.md::3, fine_grained-02: atlas/access-mfa.md::3, multi_relevant-03: atlas/search-semantic.md::3; content changed for exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/api-retries.md::0, exact-07: atlas/access-service-accounts.md::1, semantic-01: atlas/access-sessions.md::0, semantic-03: atlas/api-rate-limits.md::0 ... |

## Runtime tradeoff

- Baseline: 1.756s total (0.0878s/case).
- Final: 14.2799s total (0.714s/case).

The final selection was made from DEV results only; this TEST comparison is not fed back into selection.
