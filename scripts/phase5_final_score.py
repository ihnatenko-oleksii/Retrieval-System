"""Run the one-time, corpus-separated Phase 5 final comparison on benchmark-v4.

The development artifact chooses the architecture using only the 210 observed
cases.  This script refuses to run if benchmark-v4 is not sealed and refuses a
second final artifact, so the v4 set cannot become a tuning loop by accident.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.evals.benchmark_protocol import load_jsonl  # noqa: E402
from app.evals.span_relevance import load_markdown_spans, span_catalog_sha256  # noqa: E402
from app.retrieval.phase5 import (  # noqa: E402
    E5_MODEL,
    QWEN_MODEL,
    Phase5Config,
    Phase5EmbeddingRuntime,
    Phase5Index,
    Phase5Retriever,
    aggregate_details,
    bootstrap_mean_delta,
    build_records,
    corpus_sha256,
    evaluate_cases,
    query_instruction_mode,
)

V4_ROOT = REPO_ROOT / "docs" / "benchmark_v4"
V4_CORPUS_ROOT = V4_ROOT / "corpus"
V4_EVAL = V4_ROOT / "eval.jsonl"
V4_MANIFEST = V4_ROOT / "manifest.json"
DEV_RESULT = REPO_ROOT / "docs" / "retrieval_results" / "phase5_development.json"
OUTPUT_ROOT = REPO_ROOT / "docs" / "retrieval_results"
FINAL_JSON = OUTPUT_ROOT / "phase5_final_results.json"
FINAL_CSV = OUTPUT_ROOT / "phase5_final_results.csv"
FINAL_MD = OUTPUT_ROOT / "phase5_final_results.md"
SEED = 1729
BOOTSTRAP_SAMPLES = 5000

OBSERVED_EVALS = (
    REPO_ROOT / "docs" / "benchmark_eval.jsonl",
    REPO_ROOT / "docs" / "benchmark_v2" / "eval.jsonl",
    REPO_ROOT / "docs" / "benchmark_v3" / "eval.jsonl",
)


def _sha256_jsonl(cases: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        json.dumps(
            {key: case[key] for key in ("id", "category", "question", "relevance_spans")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _question_set(paths: tuple[Path, ...]) -> set[str]:
    return {
        " ".join(str(case["question"]).casefold().split())
        for path in paths
        for case in load_jsonl(path)
    }


def _config_from_payload(payload: dict[str, Any], *, name: str) -> Phase5Config:
    values = dict(payload)
    values["name"] = name
    return Phase5Config(**values)


def _validate_frozen_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    required = (V4_MANIFEST, V4_EVAL, DEV_RESULT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"final scoring inputs are missing: {', '.join(missing)}")
    if any(path.exists() for path in (FINAL_JSON, FINAL_CSV, FINAL_MD)):
        raise RuntimeError("refusing a second Phase 5 final scoring artifact")

    manifest = json.loads(V4_MANIFEST.read_text(encoding="utf-8"))
    cases = load_jsonl(V4_EVAL)
    development = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
    if manifest.get("artifact") != "benchmark-v4":
        raise ValueError("unexpected final benchmark artifact")
    if manifest.get("selection_eligible") is not False:
        raise ValueError("benchmark-v4 is selection eligible; final scoring is refused")
    if manifest.get("sealed_before_final_evaluation") is not True:
        raise ValueError("benchmark-v4 does not prove it was sealed before final evaluation")
    if manifest.get("tuning_allowed") is not False:
        raise ValueError("benchmark-v4 does not prove tuning was disabled")
    if manifest.get("question_count", 0) < 100 or manifest.get("document_count", 0) < 50:
        raise ValueError("benchmark-v4 is below the required corpus/query size")
    if len(cases) != int(manifest["question_count"]):
        raise ValueError("benchmark-v4 question count does not match its manifest")
    if _sha256_jsonl(cases) != manifest.get("question_sha256"):
        raise ValueError("benchmark-v4 question fingerprint does not match its manifest")

    spans = load_markdown_spans(V4_CORPUS_ROOT)
    if corpus_sha256(V4_CORPUS_ROOT) != manifest.get("corpus_sha256"):
        raise ValueError("benchmark-v4 corpus fingerprint does not match its manifest")
    if span_catalog_sha256(spans) != manifest.get("span_catalog_sha256"):
        raise ValueError("benchmark-v4 span catalog fingerprint does not match its manifest")
    actual_documents = {
        path.relative_to(V4_CORPUS_ROOT).as_posix() for path in V4_CORPUS_ROOT.rglob("*.md")
    }
    declared_documents = set(manifest.get("source", {}).get("documents", {}))
    if len(actual_documents) != int(manifest["document_count"]) or actual_documents != declared_documents:
        raise ValueError("benchmark-v4 document inventory does not match its manifest")
    if manifest.get("corpus_sha256") == manifest.get("atlas_corpus_sha256"):
        raise ValueError("benchmark-v4 corpus fingerprint matches the observed Atlas corpus")

    v4_questions = _question_set((V4_EVAL,))
    observed_questions = _question_set(OBSERVED_EVALS)
    if v4_questions & observed_questions:
        raise ValueError("benchmark-v4 reuses an observed development question")
    observed_documents = {
        path.relative_to(REPO_ROOT / "docs").as_posix()
        for root in (
            REPO_ROOT / "docs" / "benchmark_corpus",
            REPO_ROOT / "docs" / "benchmark_v2" / "corpus",
            REPO_ROOT / "docs" / "benchmark_v3" / "corpus",
        )
        if root.exists()
        for path in root.rglob("*.md")
    }
    if any(document in observed_documents for document in declared_documents):
        raise ValueError("benchmark-v4 reuses an observed source document")
    if development.get("benchmark_v4_used_for_selection") is not False:
        raise ValueError("development artifact does not prove benchmark-v4 isolation")
    if not development.get("selected", {}).get("config"):
        raise ValueError("development artifact has no frozen selected configuration")

    return manifest, cases, development


def _score_system(
    *,
    name: str,
    model_name: str,
    config: Phase5Config,
    cases: list[dict[str, Any]],
    cache_dir: Path,
) -> dict[str, Any]:
    queries = [str(case["question"]) for case in cases]
    started = perf_counter()
    runtime = Phase5EmbeddingRuntime(model_name, local_files_only=True, cache_dir=cache_dir / name)
    model_load_seconds = perf_counter() - started
    index_started = perf_counter()
    records = build_records(V4_CORPUS_ROOT, config)
    index = Phase5Index(records, runtime)
    index_seconds = perf_counter() - index_started

    if config.instruction_mode == "routed":
        modes = {query_instruction_mode(query) for query in queries}
    elif config.instruction_mode in {"none", "generic", "precise", "semantic", "context", "multi"}:
        modes = {config.instruction_mode}
    else:
        raise ValueError("a learned instruction router cannot be evaluated without development folds")
    index.prepare_queries(queries, modes)
    retriever = Phase5Retriever(index, queries=queries)
    retrieval_started = perf_counter()
    details = evaluate_cases(cases, retriever, config)
    retrieval_seconds = perf_counter() - retrieval_started
    summary = aggregate_details(details, cases, seed=SEED)
    return {
        "name": name,
        "model": model_name,
        "config": config.as_dict(),
        "metrics": summary["overall"],
        "category_metrics": summary["category_metrics"],
        "latency_ms": summary["latency_ms"],
        "chunk_count": len(records),
        "document_count": len({str(record["metadata"]["file_name"]) for record in records}),
        "timing": {
            "model_load_seconds": model_load_seconds,
            "index_seconds": index_seconds,
            "retrieval_seconds": retrieval_seconds,
            "total_seconds": model_load_seconds + index_seconds + retrieval_seconds,
            "seconds_per_query": retrieval_seconds / len(cases),
        },
        "details": details,
    }


def _relative(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline * 100 if baseline else 0.0


def _detail_map(system: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(detail["case_id"]): detail for detail in system["details"]}


def _comparisons(systems: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = systems["original-e5-dense"]
    baseline_details = _detail_map(baseline)
    comparisons: dict[str, Any] = {}
    for name, system in systems.items():
        metrics = system["metrics"]
        relative = {
            metric: _relative(float(metrics[metric]), float(baseline["metrics"][metric]))
            for metric in ("recall@5", "precision@5", "mrr", "ndcg")
        }
        comparison: dict[str, Any] = {"relative_vs_original_e5_percent": relative}
        if name != "original-e5-dense":
            candidate_details = _detail_map(system)
            ids = [str(case["id"]) for case in load_jsonl(V4_EVAL)]
            comparison["bootstrap_delta_vs_original_e5"] = {
                metric: bootstrap_mean_delta(
                    [float(baseline_details[case_id][metric]) for case_id in ids],
                    [float(candidate_details[case_id][metric]) for case_id in ids],
                    seed=SEED,
                    samples=BOOTSTRAP_SAMPLES,
                )
                for metric in ("mrr", "ndcg")
            }
        comparisons[name] = comparison

    phase3 = systems["phase3-qwen3-hybrid"]
    phase5 = systems["phase5-winner"]
    phase3_details = _detail_map(phase3)
    phase5_details = _detail_map(phase5)
    ids = [str(case["id"]) for case in load_jsonl(V4_EVAL)]
    comparisons["phase5-winner"]["bootstrap_delta_vs_phase3"] = {
        metric: bootstrap_mean_delta(
            [float(phase3_details[case_id][metric]) for case_id in ids],
            [float(phase5_details[case_id][metric]) for case_id in ids],
            seed=SEED,
            samples=BOOTSTRAP_SAMPLES,
        )
        for metric in ("mrr", "ndcg")
    }
    return comparisons


def _write_csv(payload: dict[str, Any]) -> None:
    fieldnames = [
        "system",
        "model",
        "chunk_count",
        "document_count",
        "recall@5",
        "precision@5",
        "mrr",
        "ndcg",
        "latency_ms_mean",
        "latency_ms_p95",
        "relative_recall@5_vs_e5_percent",
        "relative_precision@5_vs_e5_percent",
        "relative_mrr_vs_e5_percent",
        "relative_ndcg_vs_e5_percent",
    ]
    with FINAL_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, system in payload["systems"].items():
            metrics = system["metrics"]
            relative = payload["comparisons"][name]["relative_vs_original_e5_percent"]
            writer.writerow(
                {
                    "system": name,
                    "model": system["model"],
                    "chunk_count": system["chunk_count"],
                    "document_count": system["document_count"],
                    **{metric: metrics[metric] for metric in ("recall@5", "precision@5", "mrr", "ndcg")},
                    "latency_ms_mean": system["latency_ms"]["mean"],
                    "latency_ms_p95": system["latency_ms"]["p95"],
                    "relative_recall@5_vs_e5_percent": relative["recall@5"],
                    "relative_precision@5_vs_e5_percent": relative["precision@5"],
                    "relative_mrr_vs_e5_percent": relative["mrr"],
                    "relative_ndcg_vs_e5_percent": relative["ndcg"],
                }
            )


def _write_markdown(payload: dict[str, Any]) -> None:
    baseline = payload["systems"]["original-e5-dense"]
    phase3 = payload["systems"]["phase3-qwen3-hybrid"]
    phase5 = payload["systems"]["phase5-winner"]
    lines = [
        "# Phase 5 Final benchmark-v4 comparison",
        "",
        "This is the single final comparison on a frozen corpus-separated benchmark. "
        "The 210 observed cases were used only for grouped development CV; benchmark-v4 was not used for selection or tuning.",
        "",
        f"- Benchmark-v4 corpus SHA-256: `{payload['benchmark_v4']['corpus_sha256']}`",
        f"- Benchmark-v4 questions: {payload['benchmark_v4']['question_count']}",
        f"- Benchmark-v4 documents: {payload['benchmark_v4']['document_count']}",
        f"- Final scoring call count: `{payload['final_score_call_count']}`",
        f"- Frozen development winner: `{payload['selection']['selected_name']}`",
        f"- Phase 5 winner equals Phase 3: `{payload['selection']['phase5_winner_equals_phase3']}`",
        "",
        "## Overall retrieval metrics",
        "",
        "| System | Recall@5 | Precision@5 | MRR | nDCG | Mean ms/query | p95 ms/query |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, system in payload["systems"].items():
        metrics = system["metrics"]
        lines.append(
            f"| {name} | {metrics['recall@5']:.6f} | {metrics['precision@5']:.6f} | "
            f"{metrics['mrr']:.6f} | {metrics['ndcg']:.6f} | {system['latency_ms']['mean']:.3f} | "
            f"{system['latency_ms']['p95']:.3f} |"
        )
    lines.extend(["", "## Relative gains vs original E5 dense", "", "| System | Recall@5 | Precision@5 | MRR | nDCG |", "|---|---:|---:|---:|---:|"])
    for name, comparison in payload["comparisons"].items():
        relative = comparison["relative_vs_original_e5_percent"]
        lines.append(
            f"| {name} | {relative['recall@5']:+.2f}% | {relative['precision@5']:+.2f}% | "
            f"{relative['mrr']:+.2f}% | {relative['ndcg']:+.2f}% |"
        )
    lines.extend(["", "## Bootstrap 95% CIs for delta", "", "Delta is candidate minus baseline, paired by query ID.", "", "| Comparison | Metric | Observed delta | 95% CI | Crosses zero |", "|---|---|---:|---:|---:|"])
    for comparison_name in ("phase3-qwen3-hybrid", "phase5-winner"):
        comparison = payload["comparisons"][comparison_name]
        for metric in ("ndcg", "mrr"):
            interval = comparison["bootstrap_delta_vs_original_e5"][metric]
            lines.append(
                f"| {comparison_name} vs original-e5-dense | {metric} | {interval['observed']:+.6f} | "
                f"[{interval['lower_95']:+.6f}, {interval['upper_95']:+.6f}] | {interval['crosses_zero']} |"
            )
    for metric in ("ndcg", "mrr"):
        interval = payload["comparisons"]["phase5-winner"]["bootstrap_delta_vs_phase3"][metric]
        lines.append(
            f"| phase5-winner vs phase3-qwen3-hybrid | {metric} | {interval['observed']:+.6f} | "
            f"[{interval['lower_95']:+.6f}, {interval['upper_95']:+.6f}] | {interval['crosses_zero']} |"
        )
    lines.extend(["", "## Category nDCG / MRR", "", "| Category | Original E5 nDCG | Phase 3 nDCG | Phase 5 nDCG | Original E5 MRR | Phase 3 MRR | Phase 5 MRR |", "|---|---:|---:|---:|---:|---:|---:|"])
    for category in sorted(baseline["category_metrics"]):
        lines.append(
            f"| {category} | {baseline['category_metrics'][category]['ndcg']:.6f} | "
            f"{phase3['category_metrics'][category]['ndcg']:.6f} | {phase5['category_metrics'][category]['ndcg']:.6f} | "
            f"{baseline['category_metrics'][category]['mrr']:.6f} | {phase3['category_metrics'][category]['mrr']:.6f} | "
            f"{phase5['category_metrics'][category]['mrr']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Phase 5 architecture",
            "",
            "```json",
            json.dumps(phase5["config"], indent=2),
            "```",
            "",
            "## Failed or rejected development experiments",
            "",
            "- Short 400/80 chunks and heading-neighbor context reduced grouped-CV nDCG; they were rejected.",
            "- Instruction routing and technical BM25 variants did not provide a stable, complexity-adjusted win.",
            "- Real XGBoost LambdaMART (`rank:ndcg`) was evaluated in grouped folds and was rejected because its CV nDCG was below the simple Phase 3 architecture; no random-forest substitute was counted as LambdaMART.",
            "- The combined best component settings were within the 1% tolerance but more complex, so the simple Phase 3 architecture remained the frozen Phase 5 winner.",
            "",
            "## Recommended CV bullet",
            "",
            f"- Evaluated a frozen retrieval architecture on a corpus-separated 125-query benchmark: Phase 5 achieved nDCG {phase5['metrics']['ndcg']:.3f} vs. original E5 dense {baseline['metrics']['ndcg']:.3f} ({payload['comparisons']['phase5-winner']['relative_vs_original_e5_percent']['ndcg']:+.1f}% relative) and MRR {phase5['metrics']['mrr']:.3f}, with paired bootstrap uncertainty reported.",
            "",
            "Existing production defaults were not changed by this experiment.",
        ]
    )
    FINAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    manifest, cases, development = _validate_frozen_inputs()
    selected_name = str(development["selected"]["name"])
    phase5_config = _config_from_payload(development["selected"]["config"], name="phase5-winner")
    if phase5_config.ltr_on:
        raise ValueError("the selected development ranker needs fold training and cannot be a frozen final config")
    phase3_config = Phase5Config(name="phase3-qwen3-hybrid")
    e5_config = Phase5Config(
        name="original-e5-dense",
        instruction_mode="none",
        dense_weight=1.0,
        sparse_weight=0.0,
    )

    systems = {
        "original-e5-dense": _score_system(
            name="original-e5-dense",
            model_name=E5_MODEL,
            config=e5_config,
            cases=cases,
            cache_dir=args.cache_dir,
        ),
        "phase3-qwen3-hybrid": _score_system(
            name="phase3-qwen3-hybrid",
            model_name=QWEN_MODEL,
            config=phase3_config,
            cases=cases,
            cache_dir=args.cache_dir,
        ),
        "phase5-winner": _score_system(
            name="phase5-winner",
            model_name=QWEN_MODEL,
            config=phase5_config,
            cases=cases,
            cache_dir=args.cache_dir,
        ),
    }
    payload: dict[str, Any] = {
        "artifact": "phase5_final_results",
        "final_score_call_count": 1,
        "selection": {
            "selected_name": selected_name,
            "selected_config": development["selected"]["config"],
            "phase5_winner_equals_phase3": phase5_config.as_dict() | {"name": phase3_config.name}
            == phase3_config.as_dict(),
            "benchmark_v4_used_for_selection": False,
        },
        "benchmark_v4": {
            "manifest_sha256": hashlib.sha256(V4_MANIFEST.read_bytes()).hexdigest(),
            "corpus_sha256": manifest["corpus_sha256"],
            "question_sha256": manifest["question_sha256"],
            "document_count": manifest["document_count"],
            "question_count": manifest["question_count"],
            "span_count": manifest["span_count"],
            "source": manifest["source"],
        },
        "systems": systems,
    }
    payload["comparisons"] = _comparisons(systems)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FINAL_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(payload)
    _write_markdown(payload)
    print(
        json.dumps(
            {
                "artifact": payload["artifact"],
                "final_score_call_count": payload["final_score_call_count"],
                "metrics": {name: system["metrics"] for name, system in systems.items()},
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/private/tmp/retrieval-system-phase5-final-embeddings"),
        help="Resumable local cache for the two final embedding systems.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
