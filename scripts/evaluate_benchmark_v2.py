"""Evaluate the frozen Phase 3 configuration on the independent challenge set."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=root / "docs" / "benchmark_corpus")
    parser.add_argument("--eval", type=Path, default=root / "docs" / "benchmark_v2" / "eval.jsonl")
    parser.add_argument("--manifest", type=Path, default=root / "docs" / "benchmark_v2" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=root / "docs" / "benchmark_v2")
    parser.add_argument("--index-root", type=Path, default=root / "storage" / "benchmark_v2")
    parser.add_argument("--offline-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline_models:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from app.evals.benchmark_protocol import load_jsonl
    from app.evals.experiment_runner import (
        DEFAULT_QWEN3_EMBEDDING_MODEL,
        ExperimentRunner,
        ExperimentSpec,
        _validate_chunk_labels,
        baseline_spec,
        relative_improvement,
    )

    cases = load_jsonl(args.eval)
    if len(cases) < 40:
        raise ValueError(f"Independent challenge set must contain at least 40 cases, found {len(cases)}")

    runner = ExperimentRunner(corpus_dir=args.corpus, index_root=args.index_root, skip_generation=True)
    _validate_chunk_labels(cases, list(runner._get_canonical_chunks()))

    phase3_spec = ExperimentSpec(
        name="qwen3-hybrid-generic-instruction",
        embedding_model=DEFAULT_QWEN3_EMBEDDING_MODEL,
        dense_weight=0.7,
        sparse_weight=0.3,
        qwen_instruction_mode="generic",
    )
    results = [
        runner.run_one(baseline_spec(), args.eval, "benchmark_v2"),
        runner.run_one(phase3_spec, args.eval, "benchmark_v2"),
    ]
    if any(result.status != "ok" for result in results):
        reasons = "; ".join(f"{result.spec.name}: {result.status}: {result.error}" for result in results)
        raise RuntimeError(f"Independent challenge evaluation did not complete: {reasons}")

    baseline, final = results
    relative = {
        metric: relative_improvement(baseline.metrics, final.metrics, metric)
        for metric in ("recall@5", "precision@5", "mrr", "ndcg")
    }
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    command = shlex.join(["uv", "run", "scripts/evaluate_benchmark_v2.py", *sys.argv[1:]])
    payload = {
        "artifact": "independent_challenge_set",
        "selection_eligible": False,
        "command": command,
        "manifest": manifest,
        "results": [result.as_dict() for result in results],
        "relative_improvement_percent": relative,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with (args.output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["configuration", "status", "recall@5", "precision@5", "mrr", "ndcg", "elapsed_seconds", "seconds_per_case"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "configuration": result.spec.name,
                    "status": result.status,
                    **{metric: result.metrics.get(metric, "") for metric in ("recall@5", "precision@5", "mrr", "ndcg")},
                    "elapsed_seconds": result.elapsed_seconds,
                    "seconds_per_case": result.seconds_per_case,
                }
            )

    lines = [
        "# Independent Challenge Set (benchmark-v2)",
        "",
        f"Command: `{command}`",
        f"Cases: {len(cases)}; categories: {manifest['category_counts']}",
        "",
        "This artifact was created after Phase 3 DEV selection and frozen TEST evaluation. It is not selection-eligible.",
        "All relevance IDs were validated against the unchanged default corpus chunks before scoring.",
        "",
        "## Overall metrics",
        "",
        "| Configuration | Recall@5 | Precision@5 | MRR | nDCG | Seconds | Seconds/query |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.spec.name} | {result.metrics['recall@5']} | {result.metrics['precision@5']} | "
            f"{result.metrics['mrr']} | {result.metrics['ndcg']} | {result.elapsed_seconds} | {result.seconds_per_case} |"
        )
    lines.extend(
        [
            "",
            "## Phase 3 improvement over dense baseline",
            "",
            "| Metric | Relative improvement |",
            "|---|---:|",
            *[f"| {metric} | {value}% |" for metric, value in relative.items()],
            "",
            "## Category nDCG",
            "",
            "| Category | Dense baseline | Phase 3 final | Cases |",
            "|---|---:|---:|---:|",
        ]
    )
    categories = sorted(set(baseline.category_metrics) | set(final.category_metrics))
    for category in categories:
        baseline_category = baseline.category_metrics.get(category, {})
        final_category = final.category_metrics.get(category, {})
        lines.append(
            f"| {category} | {baseline_category.get('ndcg', '-')} | {final_category.get('ndcg', '-')} | "
            f"{final_category.get('cases', baseline_category.get('cases', '-'))} |"
        )
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote independent challenge results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
