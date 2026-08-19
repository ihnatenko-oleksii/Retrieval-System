# Dev Retrieval Experiments

Command: `uv run scripts/retrieval_experiments.py --offline-models --include-llm --llm-model qwen2.5:3b`
DEV cases: 40; frozen TEST cases: 20.

| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | bge-m3-hybrid-static | ok | 0.8301 | 0.9238 | 0.8371 | 0.325 | 0.0088 |
| 2 | hybrid-adaptive-linear-d0.7-s0.3 | ok | 0.8243 | 0.9425 | 0.8308 | 0.335 | 0.0091 |
| 3 | linear-static-d0.6-s0.4 | ok | 0.8207 | 0.9425 | 0.8162 | 0.325 | 0.0012 |
| 4 | linear-static-d0.8-s0.2 | ok | 0.8198 | 0.9425 | 0.8246 | 0.33 | 0.0013 |
| 5 | bge-m3-dense | ok | 0.8186 | 0.9258 | 0.8017 | 0.305 | 9.0167 |
| 6 | hybrid-static-depth-30 | ok | 0.8184 | 0.9425 | 0.8225 | 0.33 | 1.3552 |
| 7 | linear-static-d0.7-s0.3 | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.0012 |
| 8 | e5-base-hybrid-static | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.0014 |
| 9 | hybrid-static-depth-20 | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.0017 |
| 10 | hybrid-chunks-1000-200 | ok | 0.8159 | 0.9425 | 0.8162 | 0.325 | 0.0019 |
| 11 | hybrid-static-depth-50 | ok | 0.8136 | 0.93 | 0.8162 | 0.325 | 1.4987 |
| 12 | hybrid-static-depth-75 | ok | 0.8121 | 0.9175 | 0.8162 | 0.325 | 1.2076 |
| 13 | hybrid-static-depth-100 | ok | 0.8101 | 0.905 | 0.81 | 0.32 | 1.318 |
| 14 | hybrid-static-weighted-rrf | ok | 0.8035 | 0.9175 | 0.8108 | 0.32 | 0.0016 |
| 15 | hybrid-minilm-reranker-pool-10 | ok | 0.8013 | 0.8688 | 0.8221 | 0.33 | 4.693 |
| 16 | linear-static-d0.5-s0.5 | ok | 0.8 | 0.8975 | 0.8233 | 0.33 | 0.0016 |
| 17 | hybrid-minilm-reranker-pool-20 | ok | 0.7977 | 0.8625 | 0.8213 | 0.33 | 4.3469 |
| 18 | hybrid-minilm-reranker-pool-30 | ok | 0.7949 | 0.8583 | 0.8192 | 0.33 | 6.5512 |
| 19 | hybrid-minilm-reranker-pool-50 | ok | 0.7939 | 0.8583 | 0.8192 | 0.33 | 5.8831 |
| 20 | hybrid-static-rrf | ok | 0.7903 | 0.8883 | 0.8088 | 0.32 | 0.0025 |
| 21 | linear-static-d0.9-s0.1 | ok | 0.7766 | 0.9 | 0.7913 | 0.32 | 0.009 |
| 22 | e5-base-dense | ok | 0.7748 | 0.905 | 0.81 | 0.32 | 0.0025 |
| 23 | linear-static-d1.0-s0.0 | ok | 0.7748 | 0.905 | 0.81 | 0.32 | 0.0026 |
| 24 | baseline-dense-e5 | ok | 0.7748 | 0.905 | 0.81 | 0.32 | 8.9384 |
| 25 | linear-static-d0.4-s0.6 | ok | 0.7714 | 0.8571 | 0.7921 | 0.31 | 0.0014 |
| 26 | linear-static-d0.3-s0.7 | ok | 0.7584 | 0.835 | 0.7921 | 0.31 | 0.0019 |
| 27 | linear-static-d0.2-s0.8 | ok | 0.7252 | 0.8042 | 0.7525 | 0.295 | 0.0015 |
| 28 | linear-static-d0.1-s0.9 | ok | 0.6963 | 0.75 | 0.7379 | 0.285 | 0.0015 |
| 29 | linear-static-d0.0-s1.0 | ok | 0.6909 | 0.7425 | 0.7379 | 0.285 | 0.0085 |

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
