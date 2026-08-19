# Frozen TEST Baseline vs Final

Command: `uv run scripts/retrieval_experiments.py --offline-models --skip-qwen-reranker`
Frozen TEST cases: 20.

## Configuration

- Baseline: `baseline-dense-e5` — `intfloat/multilingual-e5-base`, dense-only, no reranking, rewriting, or expansion.
- Previous final: `bge-m3-hybrid-adaptive` — `BAAI/bge-m3`, chunk `1000/200`, fusion `weighted_linear`, weights `0.7/0.3`, global rewrite replacement `False`.
- New final: `qwen3-hybrid-generic-instruction` — `Qwen/Qwen3-Embedding-0.6B`, chunk `1000/200`, fusion `weighted_linear`, weights `0.7/0.3`, candidate depth `20`, reranking `False`, query rewriting `False` (`never`), query expansion `False` (`never`), original preserved `False`, stage-2 fusion `weighted_rrf`, confidence routing `False`, Qwen instruction `generic`, PRF `False`, LTR `False`, diversity `False`, lexical boost `0.0`.

## TEST metrics

| Configuration | Recall@5 | Precision@5 | MRR | nDCG | Seconds |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.7475 | 0.28 | 0.825 | 0.7617 | 1.2092 |
| Previous final | 0.8058 | 0.31 | 0.9167 | 0.8232 | 1.5763 |
| New final | 0.8183 | 0.32 | 0.925 | 0.8557 | 6.341 |

## Relative improvement

| Metric | `(new - baseline) / baseline * 100` |
|---|---:|
| recall@5 | 9.4716% |
| precision@5 | 14.2857% |
| mrr | 12.1212% |
| ndcg | 12.3408% |

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
| BGE-M3 native dense | bge-m3-native-dense | ok | 0.8186 | 5.6531% | 0.9258 |
| BGE-M3 native dense+sparse | bge-m3-native-dense-sparse | ok | 0.74 | -4.4915% | 0.8146 |
| BGE-M3 native dense+sparse+ColBERT | bge-m3-native-dense-sparse-colbert | ok | 0.7918 | 2.1941% | 0.8925 |
| Qwen3 dense without instruction | qwen3-dense-no-instruction | ok | 0.7282 | -6.0145% | 0.8029 |
| Qwen3 dense with generic instruction | qwen3-dense-generic-instruction | ok | 0.8459 | 9.1766% | 0.9417 |
| Qwen3 hybrid with generic instruction | qwen3-hybrid-generic-instruction | ok | 0.872 | 12.5452% | 0.9542 |
| Qwen3 hybrid + reranker pool 10 | qwen3-hybrid-generic-instruction-qwen-reranker-pool-10 | blocked | - | - | - |
| Qwen3 hybrid + reranker pool 20 | qwen3-hybrid-generic-instruction-qwen-reranker-pool-20 | blocked | - | - | - |
| Qwen3 hybrid + reranker pool 30 | qwen3-hybrid-generic-instruction-qwen-reranker-pool-30 | blocked | - | - | - |
| Qwen3 hybrid + reranker pool 50 | qwen3-hybrid-generic-instruction-qwen-reranker-pool-50 | blocked | - | - | - |
| Retrieval depth 30 | hybrid-static-depth-30 | ok | 0.8184 | 5.6273% | 0.9425 |
| MiniLM rerank pool 10 | hybrid-minilm-reranker-pool-10 | ok | 0.8013 | 3.4202% | 0.8688 |
| MiniLM rerank pool 20 | hybrid-minilm-reranker-pool-20 | ok | 0.7977 | 2.9556% | 0.8625 |
| MiniLM rerank pool 30 | hybrid-minilm-reranker-pool-30 | ok | 0.7949 | 2.5942% | 0.8583 |
| MiniLM rerank pool 50 | hybrid-minilm-reranker-pool-50 | ok | 0.7939 | 2.4652% | 0.8583 |
| BGE reranker pool 10 | hybrid-bge-reranker-pool-10 | unavailable | - | - | - |
| Chunk 400/80 | hybrid-chunks-400-80 | invalid_label_mapping | - | - | - |
| Chunk 600/100 | hybrid-chunks-600-100 | invalid_label_mapping | - | - | - |
| Chunk 800/150 | hybrid-chunks-800-150 | invalid_label_mapping | - | - | - |
| Chunk 1200/200 | hybrid-chunks-1200-200 | invalid_label_mapping | - | - | - |
| PRF depth 1 | bge-m3-hybrid-adaptive-prf-depth-1 | ok | 0.8265 | 6.6727% | 0.9238 |
| PRF depth 2 | bge-m3-hybrid-adaptive-prf-depth-2 | ok | 0.8346 | 7.7181% | 0.9238 |
| PRF depth 3 | bge-m3-hybrid-adaptive-prf-depth-3 | ok | 0.8344 | 7.6923% | 0.9238 |
| MMR relevance 0.8 | bge-m3-hybrid-adaptive-mmr-0.8 | ok | 0.8372 | 8.0537% | 0.9238 |
| Lexical overlap 0.25 | bge-m3-hybrid-adaptive-lexical-0.25 | ok | 0.8481 | 9.4605% | 0.935 |

## Query routing and generalized failure modes

The deterministic gate inspects raw-query acronyms, quoted terms, numeric status codes, identifiers, precision markers, pronouns, ambiguity phrases, and question length. Protected lexical/precision signals skip optional LLM variants; only query text can trigger selective rewrite or expansion. Stage 1 fuses dense/BM25 per query, then stage 2 can fuse original, rewrite, and expansion rankings with weighted linear, RRF, or weighted RRF. Adaptive dense/BM25 routing uses lexical signals to favor sparse retrieval, long semantic questions to favor dense retrieval, and the configured mix otherwise.

- New-final TEST rewrite rate: 0.0%; expansion rate: 0.0%; confidence-triggered rewrites: 0.0.
- Previous-final TEST rewrite rate: 0.0%.
- Chunk-boundary changes were rejected when labeled chunk content no longer matched the canonical 1000/200 mapping; no invalid-label score entered selection.
- Explicitly requested models and invalid configurations were recorded as unavailable or invalid with their actual reasons; no fallback score entered those result rows.

## Recommended CV bullet

> Improved retrieval nDCG by 12.3408% (0.7617 to 0.8557) and MRR by 12.1212% (0.825 to 0.925) on a frozen 20-query TEST set, selecting Qwen/Qwen3-Embedding-0.6B with fixed-weight weighted_linear dense/BM25 fusion (0.7/0.3) from a 40-query DEV split; latency was 0.317s/query versus 0.0605s/query for the dense baseline.

## Grouped DEV validation

LTR rows below are query-grouped 5-fold DEV validation results; validation queries are disjoint from training queries in every fold.

| Configuration | Backend | Mean nDCG | Mean MRR | Fold nDCG | Fold MRR |
|---|---|---:|---:|---|---|
| qwen3-hybrid-generic-instruction+ltr-grouped-cv | sklearn-random-forest-regression-fallback | 0.7955 | 0.8808 | 0.8658, 0.6944, 0.7413, 0.7862, 0.8896 | 0.9375, 0.775, 0.8167, 0.875, 1.0 |

## Category breakdown

| Category | Baseline nDCG | Previous final nDCG | New final nDCG | Baseline MRR | Previous final MRR | New final MRR | Cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| ambiguous | 0.2243 | 0.5567 | 0.5956 | 0.375 | 0.8333 | 0.875 | 4 |
| exact_terminology | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 4 |
| fine_grained | 0.9793 | 0.9734 | 1.0 | 1.0 | 1.0 | 1.0 | 4 |
| multiple_relevant | 0.8965 | 0.853 | 0.9327 | 1.0 | 1.0 | 1.0 | 4 |
| paraphrased_semantic | 0.7085 | 0.7331 | 0.75 | 0.75 | 0.75 | 0.75 | 4 |

Best final category by nDCG: `exact_terminology` (1.0).
Worst final category by nDCG: `ambiguous` (0.5956).

## Failed or blocked experiments

| Configuration | Status | Reason |
|---|---|---|
| hybrid-bge-reranker-pool-10 | unavailable | Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-bge-reranker-pool-20 | unavailable | Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-bge-reranker-pool-30 | unavailable | Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-bge-reranker-pool-50 | unavailable | Reranker model BAAI/bge-reranker-v2-m3 could not be loaded |
| hybrid-chunks-400-80 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: content changed for exact-01: atlas/api-pagination.md::2, exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/ingestion-dedup.md::1, exact-06: atlas/api-retries.md::0, exact-07: atlas/access-service-accounts.md::1 ... |
| hybrid-chunks-600-100 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: content changed for exact-01: atlas/api-pagination.md::2, exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/ingestion-dedup.md::1, exact-07: atlas/access-service-accounts.md::1, exact-08: atlas/operations-queue.md::1 ... |
| hybrid-chunks-800-150 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: content changed for exact-01: atlas/api-pagination.md::2, exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/ingestion-dedup.md::1, exact-07: atlas/access-service-accounts.md::1, exact-08: atlas/operations-queue.md::1 ... |
| hybrid-chunks-1200-200 | invalid_label_mapping | Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: ambiguous-03: atlas/access-mfa.md::3, fine_grained-01: atlas/access-sessions.md::3, fine_grained-02: atlas/access-mfa.md::3, multi_relevant-03: atlas/search-semantic.md::3; content changed for exact-02: atlas/api-rate-limits.md::1, exact-06: atlas/api-retries.md::0, exact-07: atlas/access-service-accounts.md::1, semantic-01: atlas/access-sessions.md::0, semantic-03: atlas/api-rate-limits.md::0 ... |
| qwen3-hybrid-generic-instruction-qwen-reranker-pool-10 | blocked | Qwen3 reranker loaded and passed the ordered-pair scorer smoke test, but full DEV candidate-pool evaluation was blocked by the local CPU budget (16 real corpus chunks took 51.58s); no ranking metric was used for selection. |
| qwen3-hybrid-generic-instruction-qwen-reranker-pool-20 | blocked | Qwen3 reranker loaded and passed the ordered-pair scorer smoke test, but full DEV candidate-pool evaluation was blocked by the local CPU budget (16 real corpus chunks took 51.58s); no ranking metric was used for selection. |
| qwen3-hybrid-generic-instruction-qwen-reranker-pool-30 | blocked | Qwen3 reranker loaded and passed the ordered-pair scorer smoke test, but full DEV candidate-pool evaluation was blocked by the local CPU budget (16 real corpus chunks took 51.58s); no ranking metric was used for selection. |
| qwen3-hybrid-generic-instruction-qwen-reranker-pool-50 | blocked | Qwen3 reranker loaded and passed the ordered-pair scorer smoke test, but full DEV candidate-pool evaluation was blocked by the local CPU budget (16 real corpus chunks took 51.58s); no ranking metric was used for selection. |

## Runtime tradeoff

- Baseline: 1.2092s total (0.0605s/case).
- Previous final: 1.5763s total (0.0788s/case), rewrite rate 0.0%.
- New final: 6.341s total (0.317s/case), rewrite rate 0.0%, expansion rate 0.0%.

The final selection was made from DEV results only; this TEST comparison is not fed back into selection.

## Independent challenge set (benchmark-v2)

The separate 50-query challenge artifact was created after Phase 3 selection and frozen TEST evaluation. It is not selection-eligible and uses five balanced categories with labels validated against the unchanged corpus.

| Configuration | Recall@5 | Precision@5 | MRR | nDCG | Seconds/query |
|---|---:|---:|---:|---:|---:|
| Dense baseline | 0.7693 | 0.328 | 0.845 | 0.7736 | 0.1196 |
| Phase 3 final | 0.8273 | 0.352 | 0.905 | 0.8372 | 2.6504 |

Relative Phase 3 improvement on benchmark-v2: Recall +7.5393%, Precision +7.3171%, MRR +7.1006%, nDCG +8.2213%. The challenge analysis is diagnostic only: ambiguous nDCG decreased from 0.5598 to 0.3999, while multiple-relevant increased from 0.7161 to 0.8697; no architecture changes were made after observing it.

## DEV failure analysis

The frozen Phase 3 DEV run had three non-rank-1 cases. Stream-level dense and BM25 ranks, plus explicit nulls for unused learned-sparse, ColBERT, and reranker streams, are recorded in `phase3_dev_failure_analysis.json` and `.md`.
