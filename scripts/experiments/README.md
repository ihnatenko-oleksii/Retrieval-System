# Historical experiment utilities

This directory contains the Phase 3, Phase 4, and Phase 5 research utilities
that produced the preserved development and failure-analysis artifacts under
`docs/retrieval_results/`. They are kept for reproducibility and are not part
of the validated runtime path.

The root-level scripts remain the primary entry points:

- `scripts/benchmark.py` — historical benchmark harness.
- `scripts/retrieval_experiments.py` — integrity-preserving DEV/TEST protocol.
- `scripts/create_benchmark_v4.py` — sealed benchmark-v4 artifact creator.
- `scripts/phase5_final_score.py` — one-shot benchmark-v4 scorer; it refuses to
  overwrite an existing final artifact.

Optional research dependencies use semantic extras:

```bash
uv sync --extra bge  # FlagEmbedding / BGE experiments
uv sync --extra ltr  # XGBoost LambdaMART experiments
```

Do not use benchmark-v4 as tuning data or rerun its scorer merely because a
historical utility was moved.
