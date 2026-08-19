# Test Retrieval Experiments

Command: `uv run scripts/retrieval_experiments.py --offline-models --include-llm --llm-model qwen2.5:3b`
DEV cases: 40; frozen TEST cases: 20.

| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | baseline-dense-e5 | ok | 0.7617 | 0.825 | 0.7475 | 0.28 | 1.2157 |
| 2 | previous-final-bge-m3-static+global-rewrite | ok | 0.7999 | 0.875 | 0.8058 | 0.31 | 10.5795 |
| 3 | bge-m3-hybrid-adaptive | ok | 0.8232 | 0.9167 | 0.8058 | 0.31 | 1.6765 |
