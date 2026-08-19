# Phase 4 Retrieval Results

Observed development cases: 110 (legacy DEV/TEST + benchmark-v2).
Latency is measured per query in the shared process; later candidates may use warm stream-query caches.
Benchmark-v3 was sealed before selection and was not used for tuning.

## Development selection

Selected architecture: `phase4-routed-prf-cascade`.
Grouped-CV nDCG: 0.8386 +/- 0.0359; MRR: 0.9065 +/- 0.0318.

| Candidate | nDCG mean | nDCG std | MRR mean | MRR std | latency ms | escalation % |
|---|---:|---:|---:|---:|---:|---:|
| phase4-multi-static-rrf | 0.8292 | 0.0337 | 0.8958 | 0.0192 | 465.5 | 0.0 |
| phase4-multi-weighted-router | 0.8295 | 0.0345 | 0.8958 | 0.0279 | 0.4 | 0.0 |
| phase4-routed-instruction-field-bm25 | 0.8216 | 0.0462 | 0.8908 | 0.0248 | 384.0 | 0.0 |
| phase4-routed-prf-cascade | 0.8386 | 0.0359 | 0.9065 | 0.0318 | 609.5 | 56.4 |

## One-time benchmark-v3 comparison

| System | nDCG | MRR | Recall@5 | Precision@5 |
|---|---:|---:|---:|---:|
| original-e5-dense | 0.6886 | 0.7898 | 0.7650 | 0.2660 |
| phase3-qwen3-hybrid | 0.7710 | 0.8745 | 0.8342 | 0.3040 |
| phase4-final | 0.7544 | 0.8442 | 0.8333 | 0.3000 |

v3 scoring calls recorded: 3 (required: 3).
Relative nDCG vs original E5: 9.56%.
Relative MRR vs original E5: 6.89%.

Measurement note: An initial pre-correction v3 control pass was discarded because plain BM25 could inherit a field-aware index. This corrected pass uses per-trial sparse variants; selection parameters were unchanged and v3 was not consulted for selection.

## Blocked or unavailable components

{'lambda_mart': 'blocked: xgboost is not installed; grouped pairwise-linear ranking was evaluated', 'qwen_hyde': 'blocked unless a local hypothetical-answer provider is explicitly configured', 'qwen_fast_reranker': 'not selected: the Phase 3 CPU smoke test measured 51.58 seconds for 16 real chunks'}
