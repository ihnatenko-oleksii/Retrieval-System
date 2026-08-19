# Phase 4 observed-data failure analysis

This analysis uses only the 110 observed legacy and benchmark-v2 cases. The
sealed benchmark-v3 cases were not used for diagnosis, selection, or tuning.

## Ambiguity recovery

Ambiguous queries remain the weakest observed category for every Phase 4
candidate. The selected routed-PRF/cascade candidate reached nDCG 0.5566 and
MRR 0.7356 on ambiguous cases, versus nDCG 0.5319 for static multi-stream RRF.
PRF plus escalation therefore recovered some ambiguous misses, but the gain was
not stable enough to claim that ambiguity was solved. The routed-instruction
candidate was worse at nDCG 0.5272, so it was not selected as a global policy.

## Stream and architecture comparison

| Observed configuration | nDCG | MRR | Interpretation |
|---|---:|---:|---|
| Original E5 dense control | 0.7719 | 0.8632 | Strong semantic baseline, weak on broad multi-relevant cases |
| Frozen Phase 3 Qwen3 + plain BM25 | 0.8178 | 0.8768 | Better semantic and exact retrieval, but still ambiguity-sensitive |
| Phase 4 static Qwen3/BGE-M3/E5/BM25 RRF | 0.8292 | 0.8958 | Independent stream agreement improves aggregate quality |
| Phase 4 query/result router | 0.8295 | 0.8958 | Router effect is small without escalation; it is label-free and stable |
| Phase 4 routed PRF/cascade | 0.8386 | 0.9065 | Best observed aggregate; 56.4% of cases escalated |

The traces retain `qwen_rank`, `bge_rank`, `e5_rank`, and `bm25_rank` for every
candidate. Disagreement is most useful as an escalation signal; deterministic
instruction routing alone did not improve the observed aggregate.

## Fine-grained and multi-relevant behavior

The selected candidate reached nDCG 0.9635 on fine-grained cases and 0.8475 on
multiple-relevant cases. The latter improved only modestly over static RRF
(0.8369), indicating that adding independent streams does not guarantee that
all complementary evidence survives top-k selection. Hierarchical section
signals and PRF were retained in the frozen architecture because they improved
the observed development aggregate, with their cost and escalation rate
reported explicitly.

## Known blocked components

- LambdaMART was not available because `xgboost` is not installed; grouped
  pairwise-linear ranking was implemented and cross-validated as the equivalent
  fallback, but it was not selected for the final run.
- Qwen HyDE was preserved as an optional original-preserving hook but was
  blocked without a local hypothetical-answer provider.
- The dedicated Qwen reranker was not selected because the prior CPU smoke test
  measured 51.58 seconds for 16 real corpus chunks.

The one-time benchmark-v3 result is reported in
`docs/retrieval_results/phase4_report.md` without retuning after that score.
