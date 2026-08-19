# Frozen TEST Baseline vs Final

Command: `uv run scripts/retrieval_experiments.py --offline-models --include-llm --llm-model qwen2.5:3b`
Frozen TEST cases: 20.

## Configuration

- Baseline: `baseline-dense-e5` — `intfloat/multilingual-e5-base`, dense-only, no reranking, rewriting, or expansion.
- Previous final: `previous-final-bge-m3-static+global-rewrite` — `BAAI/bge-m3`, chunk `1000/200`, fusion `weighted_linear`, weights `0.7/0.3`, global rewrite replacement `True`.
- New final: `bge-m3-hybrid-adaptive` — `BAAI/bge-m3`, chunk `1000/200`, fusion `weighted_linear`, weights `0.7/0.3`, candidate depth `20`, reranking `False`, query rewriting `False` (`never`), query expansion `False` (`never`), original preserved `False`, stage-2 fusion `weighted_rrf`, confidence routing `False`.

## TEST metrics

| Configuration | Recall@5 | Precision@5 | MRR | nDCG | Seconds |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.7475 | 0.28 | 0.825 | 0.7617 | 1.2157 |
| Previous final | 0.8058 | 0.31 | 0.875 | 0.7999 | 10.5795 |
| New final | 0.8058 | 0.31 | 0.9167 | 0.8232 | 1.6765 |

## Relative improvement

| Metric | `(new - baseline) / baseline * 100` |
|---|---:|
| recall@5 | 7.7993% |
| precision@5 | 10.7143% |
| mrr | 11.1152% |
| ndcg | 8.074% |

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
| LLM global-rewrite | previous-final-bge-m3-static+global-rewrite | ok | 0.8313 | 7.2922% | 0.9313 |
| LLM rewrite-all-replace | bge-m3-hybrid-adaptive+rewrite-all-replace | ok | 0.8115 | 4.7367% | 0.8938 |
| LLM rewrite-all-ensemble-linear | bge-m3-hybrid-adaptive+rewrite-all-ensemble-linear | ok | 0.825 | 6.4791% | 0.905 |
| LLM rewrite-all-ensemble-rrf | bge-m3-hybrid-adaptive+rewrite-all-ensemble-rrf | ok | 0.8345 | 7.7052% | 0.9225 |
| LLM rewrite-selective-ensemble-linear | bge-m3-hybrid-adaptive+rewrite-selective-ensemble-linear | ok | 0.8342 | 7.6665% | 0.9225 |
| LLM rewrite-selective-ensemble-rrf | bge-m3-hybrid-adaptive+rewrite-selective-ensemble-rrf | ok | 0.8342 | 7.6665% | 0.9225 |
| LLM expansion-selective-ensemble-rrf | bge-m3-hybrid-adaptive+expansion-selective-ensemble-rrf | ok | 0.8049 | 3.8849% | 0.8863 |
| LLM rewrite-expansion-selective-ensemble-rrf | bge-m3-hybrid-adaptive+rewrite-expansion-selective-ensemble-rrf | ok | 0.8095 | 4.4786% | 0.9037 |
| LLM rewrite-selective-confidence-ensemble-rrf | bge-m3-hybrid-adaptive+rewrite-selective-confidence-ensemble-rrf | ok | 0.8342 | 7.6665% | 0.9225 |

## Query routing and generalized failure modes

The deterministic gate inspects raw-query acronyms, quoted terms, numeric status codes, identifiers, precision markers, pronouns, ambiguity phrases, and question length. Protected lexical/precision signals skip optional LLM variants; only query text can trigger selective rewrite or expansion. Stage 1 fuses dense/BM25 per query, then stage 2 can fuse original, rewrite, and expansion rankings with weighted linear, RRF, or weighted RRF. Adaptive dense/BM25 routing uses lexical signals to favor sparse retrieval, long semantic questions to favor dense retrieval, and the configured mix otherwise.

- New-final TEST rewrite rate: 0.0%; expansion rate: 0.0%; confidence-triggered rewrites: 0.0.
- Previous-final TEST rewrite rate: 100.0%.
- Rewrite replacement drift was measured as a general failure mode: DEV nDCG was 0.8115 for replacement versus 0.8345 when the original query was retained and fused by rank.
- Query-expansion ensemble retrieval introduced measurable noise on this benchmark: the selective expansion ensemble reached DEV nDCG 0.8049 and MRR 0.8863, below the selected adaptive route.
- Chunk-boundary changes were rejected when labeled chunk content no longer matched the canonical 1000/200 mapping; no invalid-label score entered selection.
- BGE cross-encoder reranker candidates were recorded as unavailable because the offline cache lacked model weights; no silent fallback score was used.

## Recommended CV bullet

> Improved retrieval nDCG by 8.074% (0.7617 to 0.8232) and MRR by 11.1152% (0.825 to 0.9167) on a frozen 20-query TEST set, selecting BAAI/bge-m3 with query-adaptive weighted_linear dense/BM25 fusion (0.7/0.3) from a 40-query DEV split; latency was 0.0838s/query versus 0.0608s/query for the dense baseline.

## Category breakdown

| Category | Baseline nDCG | Previous final nDCG | New final nDCG | Baseline MRR | Previous final MRR | New final MRR | Cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| ambiguous | 0.2243 | 0.5078 | 0.5567 | 0.375 | 0.75 | 0.8333 | 4 |
| exact_terminology | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 4 |
| fine_grained | 0.9793 | 0.9077 | 0.9734 | 1.0 | 0.875 | 1.0 | 4 |
| multiple_relevant | 0.8965 | 0.8508 | 0.853 | 1.0 | 1.0 | 1.0 | 4 |
| paraphrased_semantic | 0.7085 | 0.7331 | 0.7331 | 0.75 | 0.75 | 0.75 | 4 |

Best final category by nDCG: `exact_terminology` (1.0).
Worst final category by nDCG: `ambiguous` (0.5567).

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

- Baseline: 1.2157s total (0.0608s/case).
- Previous final: 10.5795s total (0.529s/case), rewrite rate 100.0%.
- New final: 1.6765s total (0.0838s/case), rewrite rate 0.0%, expansion rate 0.0%.

The final selection was made from DEV results only; this TEST comparison is not fed back into selection.
