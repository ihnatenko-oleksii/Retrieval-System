# Dev Llm Retrieval Experiments

Command: `uv run scripts/retrieval_experiments.py --offline-models --include-llm --llm-model qwen2.5:3b`
DEV cases: 40; frozen TEST cases: 20.

| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | bge-m3-hybrid-adaptive+rewrite-all-ensemble-rrf | ok | 0.8345 | 0.9225 | 0.8371 | 0.325 | 0.0076 |
| 2 | bge-m3-hybrid-adaptive+rewrite-selective-ensemble-linear | ok | 0.8342 | 0.9225 | 0.8371 | 0.325 | 0.0059 |
| 3 | bge-m3-hybrid-adaptive+rewrite-selective-ensemble-rrf | ok | 0.8342 | 0.9225 | 0.8371 | 0.325 | 0.0082 |
| 4 | bge-m3-hybrid-adaptive+rewrite-selective-confidence-ensemble-rrf | ok | 0.8342 | 0.9225 | 0.8371 | 0.325 | 0.0091 |
| 5 | previous-final-bge-m3-static+global-rewrite | ok | 0.8313 | 0.9313 | 0.8183 | 0.325 | 21.2937 |
| 6 | bge-m3-hybrid-adaptive+rewrite-all-ensemble-linear | ok | 0.825 | 0.905 | 0.8121 | 0.32 | 0.0087 |
| 7 | bge-m3-hybrid-adaptive+rewrite-all-replace | ok | 0.8115 | 0.8938 | 0.8037 | 0.315 | 0.0071 |
| 8 | bge-m3-hybrid-adaptive+rewrite-expansion-selective-ensemble-rrf | ok | 0.8095 | 0.9037 | 0.8308 | 0.32 | 0.0116 |
| 9 | bge-m3-hybrid-adaptive+expansion-selective-ensemble-rrf | ok | 0.8049 | 0.8863 | 0.8379 | 0.325 | 22.8339 |
