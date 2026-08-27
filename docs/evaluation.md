# Evaluation map

The repository contains historical development evidence and one final independent validation. Historical results remain unchanged; the final recommendation is based on benchmark-v4.

## Benchmark hierarchy

1. **Historical original benchmark** — `docs/benchmark.md`, `docs/benchmark_eval.jsonl`, and `docs/benchmark_history.jsonl` record the earlier 20/60-case Atlas harness runs. They are useful for checking the evaluation pipeline, not for claiming held-out generalization.
2. **Development experiments** — `docs/benchmark_v2/`, `docs/benchmark_v3/`, and the Phase 3/4/5 result files record development comparisons and rejected approaches. The 210 cases used by Phase 5 are development data only.
3. **Final independent validation** — `docs/benchmark_v4/` contains 50 PSF-licensed Python standard-library documents and 125 new span-labeled questions. Its manifest records that the corpus was separated and sealed before final evaluation.

The authoritative final result is [phase5_final_results.md](retrieval_results/phase5_final_results.md). The frozen development selection and rejected experiments are in [phase5_development.md](retrieval_results/phase5_development.md). The integrity contract is in the [benchmark-v4 manifest](benchmark_v4/manifest.json).

The development record preserves the negative evidence as well as the winner:
MiniLM reranking, the more complex Phase 4 cascade, LambdaMART, the locally
slow Qwen reranker, and alternative chunking strategies did not justify
replacing the simpler validated path.

## Validated configuration

The recommended runtime path is Qwen/Qwen3-Embedding-0.6B with the generic query-only instruction, 0.7 dense / 0.3 BM25 weighted-linear fusion, and 1000/200 character chunking. Reranking, rewriting, expansion, PRF, adaptive routing, and LTR remain optional experiments and are disabled by default.

Changing the embedding model or chunking settings requires re-ingesting the application corpus. The benchmark-v4 artifact itself is already final; do not use it for tuning or rerun it merely to reproduce the published numbers.

## Reproducibility commands

These checks are lightweight and do not download multi-gigabyte models:

```bash
uv run pytest
uv run ruff check .
uv run python -m json.tool docs/benchmark_v4/manifest.json >/dev/null
```

The historical harness can be run independently when its behavior is being investigated:

```bash
uv run scripts/benchmark.py --skip-generation
```

That command evaluates the historical corpus and may append a history record; it is not the final benchmark-v4 comparison. The one-shot v4 scorer is preserved in `scripts/phase5_final_score.py` and refuses to overwrite an existing final artifact.

Historical Phase 3/4/5 ablation utilities are grouped under
`scripts/experiments/`. The root-level scripts remain the easy-to-find
benchmark and reproducibility entry points. Their optional dependencies use
semantic extras: `uv sync --extra bge` enables FlagEmbedding experiments, and
`uv sync --extra ltr` enables XGBoost LambdaMART experiments.

## Metric contract

- Recall@K measures the fraction of all positively labeled relevant evidence retrieved in the top K.
- Precision@K measures the fraction of the top K that is relevant.
- MRR is the reciprocal rank of the first relevant result.
- Graded nDCG compares the retrieved ranking against the ideal ordering constructed from all labeled relevant evidence, then normalizes to 0–1.

Retrieval metrics do not require an LLM. Generation and keyword-hit evaluation are separate, optional checks that require the configured local Ollama-compatible model.
