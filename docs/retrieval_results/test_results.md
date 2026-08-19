# Test Retrieval Experiments

Command: `uv run scripts/retrieval_experiments.py --offline-models --skip-qwen-reranker`
DEV cases: 40; frozen TEST cases: 20.

| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | baseline-dense-e5 | ok | 0.7617 | 0.825 | 0.7475 | 0.28 | 1.2092 |
| 2 | bge-m3-hybrid-adaptive | ok | 0.8232 | 0.9167 | 0.8058 | 0.31 | 1.5763 |
| 3 | qwen3-hybrid-generic-instruction | ok | 0.8557 | 0.925 | 0.8183 | 0.32 | 6.341 |
