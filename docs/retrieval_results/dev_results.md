# Dev Retrieval Experiments

Command: `uv run scripts/retrieval_experiments.py --offline-models --skip-qwen-reranker`
DEV cases: 40; frozen TEST cases: 20.

| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | qwen3-hybrid-generic-instruction | ok | 0.872 | 0.9542 | 0.8671 | 0.355 | 0.0205 |
| 2 | bge-m3-hybrid-adaptive-lexical-0.25 | ok | 0.8481 | 0.935 | 0.8433 | 0.33 | 0.0889 |
| 3 | qwen3-dense-generic-instruction | ok | 0.8459 | 0.9417 | 0.8704 | 0.355 | 9.9899 |
| 4 | bge-m3-adaptive-lexical0.30-mmr08 | ok | 0.8454 | 0.93 | 0.8371 | 0.325 | 1.2291 |
| 5 | bge-m3-adaptive-lexical0.15-mmr08 | ok | 0.8429 | 0.9363 | 0.8371 | 0.325 | 1.2383 |
| 6 | bge-m3-hybrid-adaptive-lexical-0.15 | ok | 0.8426 | 0.935 | 0.8371 | 0.325 | 0.0882 |
| 7 | bge-m3-hybrid-adaptive-lexical-0.20 | ok | 0.8418 | 0.935 | 0.8371 | 0.325 | 0.0891 |
| 8 | bge-m3-adaptive-lexical0.25-mmr08 | ok | 0.8409 | 0.935 | 0.8371 | 0.325 | 1.236 |
| 9 | bge-m3-adaptive-lexical0.20-mmr08 | ok | 0.8409 | 0.935 | 0.8371 | 0.325 | 1.2475 |
| 10 | bge-m3-adaptive-d0.5-s0.5 | ok | 0.8401 | 0.9238 | 0.8371 | 0.325 | 0.0251 |
| 11 | bge-m3-hybrid-adaptive-lexical-0.05 | ok | 0.8376 | 0.9238 | 0.8433 | 0.33 | 0.0895 |
| 12 | bge-m3-adaptive-lexical10-mmr08 | ok | 0.8374 | 0.9238 | 0.8433 | 0.33 | 1.2424 |
| 13 | bge-m3-hybrid-adaptive-mmr-0.8 | ok | 0.8372 | 0.9238 | 0.8433 | 0.33 | 1.1438 |
| 14 | bge-m3-hybrid-adaptive-lexical-0.10 | ok | 0.8369 | 0.9238 | 0.8371 | 0.325 | 0.0889 |
| 15 | bge-m3-adaptive-d0.6-s0.4 | ok | 0.8367 | 0.9238 | 0.8371 | 0.325 | 0.0247 |
| 16 | bge-m3-adaptive-d0.9-s0.1 | ok | 0.8367 | 0.9238 | 0.8371 | 0.325 | 0.0267 |
| 17 | bge-m3-adaptive-d0.8-s0.2 | ok | 0.8367 | 0.9238 | 0.8371 | 0.325 | 0.0269 |
| 18 | bge-m3-hybrid-adaptive-mmr-0.7 | ok | 0.835 | 0.9238 | 0.835 | 0.325 | 1.1362 |
| 19 | bge-m3-adaptive-prf2-mmr08 | ok | 0.8348 | 0.9238 | 0.835 | 0.325 | 1.8742 |
| 20 | bge-m3-hybrid-adaptive-prf-depth-2 | ok | 0.8346 | 0.9238 | 0.8371 | 0.325 | 0.466 |
| 21 | bge-m3-hybrid-adaptive | ok | 0.8345 | 0.9238 | 0.8371 | 0.325 | 0.0126 |
| 22 | bge-m3-hybrid-adaptive-prf-depth-3 | ok | 0.8344 | 0.9238 | 0.8371 | 0.325 | 0.4361 |
| 23 | bge-m3-hybrid-adaptive-mmr-0.6 | ok | 0.834 | 0.9238 | 0.835 | 0.325 | 1.1415 |
| 24 | bge-m3-hybrid-static | ok | 0.8301 | 0.9238 | 0.8371 | 0.325 | 0.0173 |
| 25 | bge-m3-hybrid-adaptive-mmr-0.5 | ok | 0.8298 | 0.9238 | 0.8287 | 0.32 | 4.2441 |
| 26 | bge-m3-hybrid-adaptive-prf-depth-1 | ok | 0.8265 | 0.9238 | 0.8308 | 0.32 | 0.5588 |
| 27 | hybrid-adaptive-linear-d0.7-s0.3 | ok | 0.8243 | 0.9425 | 0.8308 | 0.335 | 0.0122 |
| 28 | e5-hybrid-adaptive | ok | 0.8243 | 0.9425 | 0.8308 | 0.335 | 0.0122 |
| 29 | linear-static-d0.6-s0.4 | ok | 0.8207 | 0.9425 | 0.8162 | 0.325 | 0.0109 |
| 30 | linear-static-d0.8-s0.2 | ok | 0.8198 | 0.9425 | 0.8246 | 0.33 | 0.011 |
| 31 | bge-m3-dense | ok | 0.8186 | 0.9258 | 0.8017 | 0.305 | 6.9217 |
| 32 | bge-m3-native-dense | ok | 0.8186 | 0.9258 | 0.8017 | 0.305 | 161.0378 |
| 33 | hybrid-static-depth-30 | ok | 0.8184 | 0.9425 | 0.8225 | 0.33 | 0.9905 |
| 34 | hybrid-static-depth-20 | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.0109 |
| 35 | linear-static-d0.7-s0.3 | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.011 |
| 36 | e5-base-hybrid-static | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.011 |
| 37 | hybrid-chunks-1000-200 | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.0113 |
| 38 | hybrid-static-depth-50 | ok | 0.8136 | 0.93 | 0.8162 | 0.325 | 1.0085 |
| 39 | hybrid-static-depth-75 | ok | 0.8121 | 0.9175 | 0.8162 | 0.325 | 1.0366 |
| 40 | hybrid-static-depth-100 | ok | 0.8101 | 0.905 | 0.81 | 0.32 | 1.0628 |
| 41 | hybrid-static-weighted-rrf | ok | 0.8035 | 0.9175 | 0.8108 | 0.32 | 0.0107 |
| 42 | hybrid-minilm-reranker-pool-10 | ok | 0.8013 | 0.8688 | 0.8221 | 0.33 | 2.4593 |
| 43 | linear-static-d0.5-s0.5 | ok | 0.8 | 0.8975 | 0.8233 | 0.33 | 0.0111 |
| 44 | hybrid-minilm-reranker-pool-20 | ok | 0.7977 | 0.8625 | 0.8213 | 0.33 | 5.3894 |
| 45 | qwen3-hybrid-generic-instruction+ltr-grouped-cv | ok | 0.7955 | 0.8808 | 0.8163 | 0.315 | 15.3978 |
| 46 | hybrid-minilm-reranker-pool-30 | ok | 0.7949 | 0.8583 | 0.8192 | 0.33 | 8.5492 |
| 47 | hybrid-minilm-reranker-pool-50 | ok | 0.7939 | 0.8583 | 0.8192 | 0.33 | 8.7417 |
| 48 | bge-m3-native-dense-sparse-colbert | ok | 0.7918 | 0.8925 | 0.7871 | 0.305 | 219.1658 |
| 49 | hybrid-static-rrf | ok | 0.7903 | 0.8883 | 0.8088 | 0.32 | 0.0107 |
| 50 | linear-static-d0.9-s0.1 | ok | 0.7766 | 0.9 | 0.7913 | 0.32 | 0.0163 |
| 51 | linear-static-d1.0-s0.0 | ok | 0.7748 | 0.905 | 0.81 | 0.32 | 0.0078 |
| 52 | e5-base-dense | ok | 0.7748 | 0.905 | 0.81 | 0.32 | 0.0078 |
| 53 | baseline-dense-e5 | ok | 0.7748 | 0.905 | 0.81 | 0.32 | 5.7429 |
| 54 | linear-static-d0.4-s0.6 | ok | 0.7714 | 0.8571 | 0.7921 | 0.31 | 0.0109 |
| 55 | linear-static-d0.3-s0.7 | ok | 0.7584 | 0.835 | 0.7921 | 0.31 | 0.0109 |
| 56 | bge-m3-native-dense-sparse | ok | 0.74 | 0.8146 | 0.7787 | 0.31 | 220.9499 |
| 57 | qwen3-dense-no-instruction | ok | 0.7282 | 0.8029 | 0.8162 | 0.315 | 5.2673 |
| 58 | linear-static-d0.2-s0.8 | ok | 0.7252 | 0.8042 | 0.7525 | 0.295 | 0.0108 |
| 59 | linear-static-d0.1-s0.9 | ok | 0.6963 | 0.75 | 0.7379 | 0.285 | 0.0108 |
| 60 | linear-static-d0.0-s1.0 | ok | 0.6909 | 0.7425 | 0.7379 | 0.285 | 0.0072 |

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
