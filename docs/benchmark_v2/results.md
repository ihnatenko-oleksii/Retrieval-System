# Independent Challenge Set (benchmark-v2)

Command: `uv run scripts/evaluate_benchmark_v2.py --offline-models`
Cases: 50; categories: {'ambiguous': 10, 'fine_grained': 10, 'lexical': 10, 'multiple_relevant': 10, 'semantic': 10}

This artifact was created after Phase 3 DEV selection and frozen TEST evaluation. It is not selection-eligible.
All relevance IDs were validated against the unchanged default corpus chunks before scoring.

## Overall metrics

| Configuration | Recall@5 | Precision@5 | MRR | nDCG | Seconds | Seconds/query |
|---|---:|---:|---:|---:|---:|---:|
| baseline-dense-e5 | 0.7693 | 0.328 | 0.845 | 0.7736 | 5.9785 | 0.1196 |
| qwen3-hybrid-generic-instruction | 0.8273 | 0.352 | 0.905 | 0.8372 | 132.5193 | 2.6504 |

## Phase 3 improvement over dense baseline

| Metric | Relative improvement |
|---|---:|
| recall@5 | 7.5393% |
| precision@5 | 7.3171% |
| mrr | 7.1006% |
| ndcg | 8.2213% |

## Category nDCG

| Category | Dense baseline | Phase 3 final | Cases |
|---|---:|---:|---:|
| ambiguous | 0.5598 | 0.3999 | 10 |
| fine_grained | 0.9402 | 0.9533 | 10 |
| lexical | 0.9386 | 1.0 | 10 |
| multiple_relevant | 0.7161 | 0.8697 | 10 |
| semantic | 0.7131 | 0.9631 | 10 |
