# Dev Llm Retrieval Experiments

Command: `uv run scripts/retrieval_experiments.py --offline-models --include-llm --llm-model qwen2.5:3b`
DEV cases: 40; frozen TEST cases: 20.

| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | bge-m3-hybrid-static+query-rewrite | ok | 0.8313 | 0.9313 | 0.8183 | 0.325 | 26.7076 |
| 2 | bge-m3-hybrid-static+query-expansion | ok | 0.8176 | 0.9396 | 0.8233 | 0.325 | 56.6945 |
| 3 | bge-m3-hybrid-static+rewrite-and-expansion | ok | 0.813 | 0.9271 | 0.7975 | 0.315 | 66.3367 |
