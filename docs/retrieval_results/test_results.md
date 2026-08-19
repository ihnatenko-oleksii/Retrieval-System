# Test Retrieval Experiments

Command: `uv run scripts/retrieval_experiments.py --offline-models --include-llm --llm-model qwen2.5:3b`
DEV cases: 40; frozen TEST cases: 20.

| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | baseline-dense-e5 | ok | 0.7617 | 0.825 | 0.7475 | 0.28 | 1.756 |
| 2 | bge-m3-hybrid-static+query-rewrite | ok | 0.7999 | 0.875 | 0.8058 | 0.31 | 14.2799 |
