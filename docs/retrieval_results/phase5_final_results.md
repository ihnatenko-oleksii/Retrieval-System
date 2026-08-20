# Phase 5 Final benchmark-v4 comparison

This is the single final comparison on a frozen corpus-separated benchmark. The 210 observed cases were used only for grouped development CV; benchmark-v4 was not used for selection or tuning.

- Benchmark-v4 corpus SHA-256: `868131625355940d17db5967ce44a10229fd7371936d9fefdac8179bbcef0fb4`
- Benchmark-v4 questions: 125
- Benchmark-v4 documents: 50
- Final scoring call count: `1`
- Frozen development winner: `phase3-base`
- Phase 5 winner equals Phase 3: `True`

## Overall retrieval metrics

| System | Recall@5 | Precision@5 | MRR | nDCG | Mean ms/query | p95 ms/query |
|---|---:|---:|---:|---:|---:|---:|
| original-e5-dense | 0.784000 | 0.171200 | 0.629733 | 0.636121 | 10.532 | 11.069 |
| phase3-qwen3-hybrid | 0.936000 | 0.198400 | 0.812133 | 0.808058 | 10.276 | 10.873 |
| phase5-winner | 0.936000 | 0.198400 | 0.812133 | 0.808058 | 14.453 | 21.313 |

## Relative gains vs original E5 dense

| System | Recall@5 | Precision@5 | MRR | nDCG |
|---|---:|---:|---:|---:|
| original-e5-dense | +0.00% | +0.00% | +0.00% | +0.00% |
| phase3-qwen3-hybrid | +19.39% | +15.89% | +28.96% | +27.03% |
| phase5-winner | +19.39% | +15.89% | +28.96% | +27.03% |

## Bootstrap 95% CIs for delta

Delta is candidate minus baseline, paired by query ID.

| Comparison | Metric | Observed delta | 95% CI | Crosses zero |
|---|---|---:|---:|---:|
| phase3-qwen3-hybrid vs original-e5-dense | ndcg | +0.171937 | [+0.107941, +0.237118] | False |
| phase3-qwen3-hybrid vs original-e5-dense | mrr | +0.182400 | [+0.116397, +0.249203] | False |
| phase5-winner vs original-e5-dense | ndcg | +0.171937 | [+0.107941, +0.237118] | False |
| phase5-winner vs original-e5-dense | mrr | +0.182400 | [+0.116397, +0.249203] | False |
| phase5-winner vs phase3-qwen3-hybrid | ndcg | +0.000000 | [+0.000000, +0.000000] | True |
| phase5-winner vs phase3-qwen3-hybrid | mrr | +0.000000 | [+0.000000, +0.000000] | True |

## Category nDCG / MRR

| Category | Original E5 nDCG | Phase 3 nDCG | Phase 5 nDCG | Original E5 MRR | Phase 3 MRR | Phase 5 MRR |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous | 0.568413 | 0.832392 | 0.832392 | 0.518000 | 0.791333 | 0.791333 |
| fine_grained | 0.574857 | 0.708413 | 0.708413 | 0.528000 | 0.664667 | 0.664667 |
| lexical | 0.804929 | 0.892939 | 0.892939 | 0.766667 | 0.870000 | 0.870000 |
| multiple_relevant | 0.695985 | 0.775361 | 0.775361 | 0.848000 | 0.933333 | 0.933333 |
| semantic | 0.536423 | 0.831186 | 0.831186 | 0.488000 | 0.801333 | 0.801333 |

## Frozen Phase 5 architecture

```json
{
  "name": "phase5-winner",
  "chunking": "character",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "instruction_mode": "generic",
  "dense_weight": 0.7,
  "sparse_weight": 0.3,
  "bm25_k1": 1.2,
  "bm25_b": 0.75,
  "technical_tokens": false,
  "fusion": "weighted_linear",
  "rrf_k": 60,
  "top_k": 5,
  "candidate_depth": 50,
  "ltr_on": false
}
```

## Failed or rejected development experiments

- Short 400/80 chunks and heading-neighbor context reduced grouped-CV nDCG; they were rejected.
- Instruction routing and technical BM25 variants did not provide a stable, complexity-adjusted win.
- Real XGBoost LambdaMART (`rank:ndcg`) was evaluated in grouped folds and was rejected because its CV nDCG was below the simple Phase 3 architecture; no random-forest substitute was counted as LambdaMART.
- The combined best component settings were within the 1% tolerance but more complex, so the simple Phase 3 architecture remained the frozen Phase 5 winner.

## Recommended CV bullet

- Evaluated a frozen retrieval architecture on a corpus-separated 125-query benchmark: Phase 5 achieved nDCG 0.808 vs. original E5 dense 0.636 (+27.0% relative) and MRR 0.812, with paired bootstrap uncertainty reported.

Existing production defaults were not changed by this experiment.
