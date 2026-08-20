"""Run Phase 5 development-only retrieval ablations on all 210 observed cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.evals.benchmark_protocol import load_jsonl  # noqa: E402
from app.retrieval.phase5 import (  # noqa: E402
    QWEN_MODEL,
    Phase5Config,
    Phase5EmbeddingRuntime,
    Phase5Index,
    Phase5Retriever,
    aggregate_details,
    build_records,
    convert_cases_to_source_spans,
    corpus_sha256,
    evaluate_cases,
    evaluate_folded_lambdamart,
)

CORPUS_ROOT = REPO_ROOT / "docs" / "benchmark_corpus"
OLD_EVAL = REPO_ROOT / "docs" / "benchmark_eval.jsonl"
V2_EVAL = REPO_ROOT / "docs" / "benchmark_v2" / "eval.jsonl"
V3_EVAL = REPO_ROOT / "docs" / "benchmark_v3" / "eval.jsonl"
OUTPUT_ROOT = REPO_ROOT / "docs" / "retrieval_results"
TOP_K = 5
SEED = 1729


def _sha256_jsonl(cases: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for case in cases)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_observed_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in (OLD_EVAL, V2_EVAL, V3_EVAL):
        for case in load_jsonl(path):
            case_id = str(case["id"])
            if case_id in seen:
                raise ValueError(f"duplicate observed case ID: {case_id}")
            seen.add(case_id)
            cases.append(case)
    if len(cases) != 210:
        raise ValueError(f"Phase 5 requires all 210 observed cases, found {len(cases)}")
    return cases


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _index_key(config: Phase5Config) -> tuple[str, int, int]:
    return config.chunking, config.chunk_size, config.chunk_overlap


def _static_configs() -> list[tuple[str, str, Phase5Config, int]]:
    base = Phase5Config()
    configs: list[tuple[str, str, Phase5Config, int]] = [("phase3-base", "base", base, 0)]

    for label, size, overlap in (
        ("char-400-80", 400, 80),
        ("char-600-100", 600, 100),
        ("char-800-150", 800, 150),
        ("char-1200-200", 1200, 200),
    ):
        configs.append(
            (
                label,
                "chunking",
                Phase5Config(name=label, chunk_size=size, chunk_overlap=overlap),
                1,
            )
        )
    for label, strategy, size, overlap, complexity in (
        ("paragraph-aware", "paragraph", 800, 0, 1),
        ("heading-aware", "heading", 800, 0, 1),
        ("section-aware", "section", 1000, 0, 1),
        ("heading-neighbor-context", "heading_context", 800, 0, 2),
    ):
        configs.append(
            (
                label,
                "chunking",
                Phase5Config(name=label, chunking=strategy, chunk_size=size, chunk_overlap=overlap),
                complexity,
            )
        )

    for mode in ("none", "precise", "semantic", "context", "multi", "routed"):
        label = f"instruction-{mode}"
        configs.append((label, "instruction", Phase5Config(name=label, instruction_mode=mode), 1))

    for label, k1, b, technical, complexity in (
        ("bm25-k1-0.9-b0.50", 0.9, 0.50, False, 1),
        ("bm25-k1-1.5-b0.75", 1.5, 0.75, False, 1),
        ("bm25-k1-1.8-b0.60", 1.8, 0.60, False, 1),
        ("bm25-technical-tokens", 1.2, 0.75, True, 1),
        ("bm25-technical-k1-1.5-b0.60", 1.5, 0.60, True, 2),
    ):
        configs.append(
            (
                label,
                "bm25",
                Phase5Config(name=label, bm25_k1=k1, bm25_b=b, technical_tokens=technical),
                complexity,
            )
        )

    for weight in (0.9, 0.8, 0.6, 0.5):
        label = f"fusion-{weight:.1f}-{1 - weight:.1f}"
        configs.append(
            (
                label,
                "fusion",
                Phase5Config(name=label, dense_weight=weight, sparse_weight=1 - weight),
                1,
            )
        )
    configs.append(
        (
            "fusion-weighted-rrf-0.7-0.3",
            "fusion",
            Phase5Config(name="fusion-weighted-rrf-0.7-0.3", fusion="weighted_rrf"),
            2,
        )
    )
    return configs


class LearnedInstructionRouter:
    """A small text-only router trained on development-fold query outcomes."""

    def __init__(self, queries: list[str], labels: list[str]):
        self.constant: str | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None
        if len(set(labels)) < 2:
            self.constant = labels[0] if labels else "generic"
            return
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
        matrix = self.vectorizer.fit_transform(queries)
        self.model = LogisticRegression(max_iter=1000, random_state=SEED)
        self.model.fit(matrix, labels)

    def predict(self, queries: list[str]) -> list[str]:
        if self.constant is not None:
            return [self.constant for _ in queries]
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("learned instruction router is not fitted")
        return [str(value) for value in self.model.predict(self.vectorizer.transform(queries))]


def _best_instruction_label(case: dict[str, Any], instruction_details: dict[str, dict[str, dict[str, Any]]]) -> str:
    case_id = str(case["id"])
    ranked = sorted(
        instruction_details.items(),
        key=lambda item: (
            float(item[1][case_id]["ndcg"]),
            float(item[1][case_id]["mrr"]),
            1 if item[0] == "generic" else 0,
        ),
        reverse=True,
    )
    best_mode = ranked[0][0]
    if best_mode == "routed":
        from app.retrieval.phase5 import query_instruction_mode

        return query_instruction_mode(case["question"])
    return best_mode


def _learned_instruction_cv(
    cases: list[dict[str, Any]],
    index: Phase5Index,
    queries: list[str],
    instruction_details: dict[str, list[dict[str, Any]]],
    base_config: Phase5Config,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from app.retrieval.ltr import grouped_query_folds
    from app.retrieval.phase5 import _case_metrics

    detail_maps = {
        mode: {str(detail["case_id"]): detail for detail in details}
        for mode, details in instruction_details.items()
    }
    groups = [str(case.get("query_group_id", case["id"])) for case in cases]
    folds = grouped_query_folds(groups, n_splits=5, seed=SEED)
    all_details: list[dict[str, Any]] = []
    fold_router_info: list[dict[str, Any]] = []
    for fold_number, (train_indexes, validation_indexes) in enumerate(folds, start=1):
        train_cases = [cases[index] for index in train_indexes]
        labels = [_best_instruction_label(case, detail_maps) for case in train_cases]
        router = LearnedInstructionRouter([case["question"] for case in train_cases], labels)
        retriever = Phase5Retriever(index, queries=queries, router=router)
        fold_rows: list[dict[str, Any]] = []
        for case_index in validation_indexes:
            case = cases[case_index]
            started = perf_counter()
            records = retriever.retrieve(case["question"], replace(base_config, instruction_mode="learned"))
            detail = _case_metrics(case, records, base_config.top_k)
            detail["retrieval_latency_ms"] = (perf_counter() - started) * 1000
            detail["fold"] = fold_number
            fold_rows.append(detail)
            all_details.append(detail)
        fold_router_info.append(
            {
                "fold": fold_number,
                "training_case_count": len(train_cases),
                "label_counts": dict(Counter(labels)),
                "validation_route_counts": dict(Counter(router.predict([case["question"] for case in [cases[i] for i in validation_indexes]]))),
            }
        )
    summary = aggregate_details(all_details, cases, seed=SEED)
    return all_details, summary, {"folds": fold_router_info, "uses_query_text_only": True}


def _static_result(
    label: str,
    component: str,
    config: Phase5Config,
    complexity: int,
    index: Phase5Index,
    cases: list[dict[str, Any]],
    queries: list[str],
) -> dict[str, Any]:
    retriever = Phase5Retriever(index, queries=queries)
    details = evaluate_cases(cases, retriever, config)
    summary = aggregate_details(details, cases, seed=SEED)
    return {
        "name": label,
        "component": component,
        "status": "ok",
        "selection_eligible": True,
        "config": config.as_dict(),
        "complexity": complexity,
        "summary": summary,
        "details": details,
    }


def _folded_ltr_result(
    index: Phase5Index,
    cases: list[dict[str, Any]],
    queries: list[str],
) -> dict[str, Any]:
    label = "lambdamart-vs-fixed-fusion"
    config = Phase5Config(name=label, ltr_on=True, candidate_depth=50)
    try:
        print("Starting real XGBoost LambdaMART grouped CV", flush=True)
        retriever = Phase5Retriever(index, queries=queries)
        details, _, backend = evaluate_folded_lambdamart(cases, retriever, config, seed=SEED)
        summary = aggregate_details(details, cases, seed=SEED)
        return {
            "name": label,
            "component": "lambdamart",
            "status": "ok",
            "selection_eligible": backend == "xgboost-lambdamart",
            "config": config.as_dict(),
            "complexity": 3,
            "model_backend": backend,
            "summary": summary,
            "details": details,
        }
    except Exception as exc:
        return {
            "name": label,
            "component": "lambdamart",
            "status": "blocked",
            "selection_eligible": False,
            "config": config.as_dict(),
            "complexity": 3,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _relative_gain(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    base = float(baseline["summary"]["mean"]["ndcg"])
    value = float(candidate["summary"]["mean"]["ndcg"])
    return (value - base) / base if base else 0.0


def _select(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [result for result in results if result.get("status") == "ok" and result.get("selection_eligible")]
    if not eligible:
        raise RuntimeError("no selection-eligible Phase 5 result")
    best_score = max(float(result["summary"]["mean"]["ndcg"]) for result in eligible)
    tolerance = best_score * 0.01
    near_best = [result for result in eligible if float(result["summary"]["mean"]["ndcg"]) >= best_score - tolerance]
    return min(
        near_best,
        key=lambda result: (
            int(result.get("complexity", 99)),
            -float(result["summary"]["mean"]["mrr"]),
            str(result["name"]),
        ),
    )


def _write_outputs(
    *,
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    selected: dict[str, Any],
    index_stats: dict[str, dict[str, Any]],
    corpus_hash: str,
    command: str,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "artifact": "phase5_development",
        "selection_eligible": True,
        "development_only": True,
        "case_count": len(cases),
        "case_ids": [str(case["id"]) for case in cases],
        "case_sha256": _sha256_jsonl(cases),
        "corpus_sha256": corpus_hash,
        "command": command,
        "phase3_base": Phase5Config().as_dict(),
        "selected": selected,
        "index_stats": index_stats,
        "results": results,
        "benchmark_v4_used_for_selection": False,
    }
    (output_root / "phase5_development.json").write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "name",
        "component",
        "status",
        "selection_eligible",
        "cv_ndcg_mean",
        "cv_ndcg_std",
        "cv_mrr_mean",
        "cv_mrr_std",
        "overall_ndcg",
        "overall_mrr",
        "latency_ms_mean",
        "latency_ms_p95",
        "complexity",
        "model_backend",
        "error",
    ]
    with (output_root / "phase5_development.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            summary = result.get("summary", {})
            writer.writerow(
                {
                    "name": result["name"],
                    "component": result["component"],
                    "status": result["status"],
                    "selection_eligible": result.get("selection_eligible", False),
                    "cv_ndcg_mean": summary.get("mean", {}).get("ndcg", ""),
                    "cv_ndcg_std": summary.get("std", {}).get("ndcg", ""),
                    "cv_mrr_mean": summary.get("mean", {}).get("mrr", ""),
                    "cv_mrr_std": summary.get("std", {}).get("mrr", ""),
                    "overall_ndcg": summary.get("overall", {}).get("ndcg", ""),
                    "overall_mrr": summary.get("overall", {}).get("mrr", ""),
                    "latency_ms_mean": summary.get("latency_ms", {}).get("mean", ""),
                    "latency_ms_p95": summary.get("latency_ms", {}).get("p95", ""),
                    "complexity": result.get("complexity", ""),
                    "model_backend": result.get("model_backend", ""),
                    "error": result.get("error", ""),
                }
            )

    lines = [
        "# Phase 5 Development CV",
        "",
        "All 210 existing cases (original, benchmark-v2, and benchmark-v3) are observed development data only.",
        "Benchmark-v4 was not created or used for selection.",
        "",
        f"Corpus SHA-256: `{corpus_hash}`",
        f"Cases SHA-256: `{json_payload['case_sha256']}`",
        f"Selected candidate: `{selected['name']}`",
        "Complexity rule: prefer the simplest candidate within 1% relative CV nDCG of the best eligible result.",
        "",
        "## Candidate results",
        "",
        "| Candidate | Component | Status | CV nDCG mean | CV nDCG std | CV MRR mean | CV MRR std | ms/query | Chunks |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        summary = result.get("summary", {})
        stats = index_stats.get(str(result.get("config", {}).get("chunking", "")) + ":" + str(result.get("config", {}).get("chunk_size", "")) + ":" + str(result.get("config", {}).get("chunk_overlap", "")), {})
        lines.append(
            f"| {result['name']} | {result['component']} | {result['status']} | "
            f"{summary.get('mean', {}).get('ndcg', '-')} | {summary.get('std', {}).get('ndcg', '-')} | "
            f"{summary.get('mean', {}).get('mrr', '-')} | {summary.get('std', {}).get('mrr', '-')} | "
            f"{summary.get('latency_ms', {}).get('mean', '-')} | {stats.get('chunks', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Selected architecture",
            "",
            "```json",
            json.dumps(selected.get("config", {}), indent=2),
            "```",
            "",
            "## Fold-level results for selected candidate",
            "",
            "| Fold | nDCG | MRR | Validation queries |",
            "|---:|---:|---:|---:|",
        ]
    )
    for fold in selected.get("summary", {}).get("folds", []):
        lines.append(
            f"| {fold['fold']} | {fold['metrics']['ndcg']:.6f} | {fold['metrics']['mrr']:.6f} | "
            f"{len(fold['validation_query_ids'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Legacy chunk-ID labels were converted once to source intervals using the frozen 1000/200 canonical splitter; all experiments then used span overlap.",
            "- Instruction routing uses query text only. No benchmark category, case ID, source filename, or relevance-derived runtime feature is passed to retrieval.",
            "- LambdaMART is selection-eligible only when the backend is real XGBoost `rank:ndcg`; a random-forest fallback is not accepted as LambdaMART.",
            "- Existing production defaults were not changed by this development run.",
        ]
    )
    (output_root / "phase5_development.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    observed = _load_observed_cases()
    canonical_config = Phase5Config(name="canonical-label-source")
    canonical_records = build_records(CORPUS_ROOT, canonical_config)
    cases = convert_cases_to_source_spans(observed, canonical_records)
    _write_jsonl(OUTPUT_ROOT / "phase5_development_cases.jsonl", cases)
    queries = [str(case["question"]) for case in cases]

    runtime = Phase5EmbeddingRuntime(
        QWEN_MODEL,
        local_files_only=args.offline_models,
        cache_dir=args.cache_dir,
    )
    index_cache: dict[tuple[str, int, int], Phase5Index] = {}
    index_stats: dict[str, dict[str, Any]] = {}

    def get_index(config: Phase5Config) -> Phase5Index:
        key = _index_key(config)
        if key not in index_cache:
            started = perf_counter()
            records = build_records(CORPUS_ROOT, config)
            index = Phase5Index(records, runtime)
            modes = {"generic", "none", "precise", "semantic", "context", "multi"}
            index.prepare_queries(queries, modes)
            index_cache[key] = index
            index_stats[":".join(str(value) for value in key)] = {
                "chunking": key[0],
                "chunk_size": key[1],
                "chunk_overlap": key[2],
                "chunks": len(records),
                "documents": len({record["metadata"]["file_name"] for record in records}),
                "index_seconds": perf_counter() - started,
            }
            print(f"Indexed {key[0]} {key[1]}/{key[2]}: {len(records)} chunks", flush=True)
        return index_cache[key]

    results: list[dict[str, Any]] = []
    static_instruction_details: dict[str, list[dict[str, Any]]] = {}
    for label, component, config, complexity in _static_configs():
        index = get_index(config)
        result = _static_result(label, component, config, complexity, index, cases, queries)
        results.append(result)
        print(
            f"Evaluated {label}: nDCG={result['summary']['mean']['ndcg']:.4f} "
            f"MRR={result['summary']['mean']['mrr']:.4f}",
            flush=True,
        )
        if component == "instruction":
            static_instruction_details[config.instruction_mode] = result["details"]

    baseline = next(result for result in results if result["name"] == "phase3-base")
    static_instruction_details["generic"] = baseline["details"]
    base_index = get_index(Phase5Config())
    learned_config = Phase5Config(name="instruction-learned-routing", instruction_mode="learned")
    try:
        learned_details, learned_summary, learned_info = _learned_instruction_cv(
            cases,
            base_index,
            queries,
            static_instruction_details,
            Phase5Config(name="learned-routing-base"),
        )
        results.append(
            {
                "name": learned_config.name,
                "component": "instruction",
                "status": "ok",
                "selection_eligible": True,
                "config": learned_config.as_dict(),
                "complexity": 2,
                "summary": learned_summary,
                "details": learned_details,
                "router": learned_info,
            }
        )
    except Exception as exc:
        results.append(
            {
                "name": learned_config.name,
                "component": "instruction",
                "status": "failed",
                "selection_eligible": False,
                "config": learned_config.as_dict(),
                "complexity": 2,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    results.append(_folded_ltr_result(base_index, cases, queries))

    eligible_static = [result for result in results if result.get("status") == "ok" and result.get("selection_eligible")]
    best_chunk = max((result for result in eligible_static if result["component"] == "chunking"), key=lambda r: r["summary"]["mean"]["ndcg"], default=baseline)
    best_instruction = max((result for result in eligible_static if result["component"] == "instruction" and result["name"] != "instruction-learned-routing"), key=lambda r: r["summary"]["mean"]["ndcg"], default=baseline)
    best_bm25 = max((result for result in eligible_static if result["component"] == "bm25"), key=lambda r: r["summary"]["mean"]["ndcg"], default=baseline)
    best_fusion = max((result for result in eligible_static if result["component"] == "fusion"), key=lambda r: r["summary"]["mean"]["ndcg"], default=baseline)
    combined_config = Phase5Config(
        name="phase5-combined-selected-components",
        chunking=best_chunk["config"]["chunking"],
        chunk_size=int(best_chunk["config"]["chunk_size"]),
        chunk_overlap=int(best_chunk["config"]["chunk_overlap"]),
        instruction_mode=best_instruction["config"]["instruction_mode"],
        dense_weight=float(best_fusion["config"]["dense_weight"]),
        sparse_weight=float(best_fusion["config"]["sparse_weight"]),
        fusion=str(best_fusion["config"]["fusion"]),
        rrf_k=int(best_fusion["config"]["rrf_k"]),
        bm25_k1=float(best_bm25["config"]["bm25_k1"]),
        bm25_b=float(best_bm25["config"]["bm25_b"]),
        technical_tokens=bool(best_bm25["config"]["technical_tokens"]),
    )
    combined = _static_result(
        "phase5-combined-selected-components",
        "combination",
        combined_config,
        int(best_chunk.get("complexity", 0)) + int(best_instruction.get("complexity", 0)) + int(best_bm25.get("complexity", 0)) + int(best_fusion.get("complexity", 0)),
        get_index(combined_config),
        cases,
        queries,
    )
    results.append(combined)

    selected = _select(results)
    selected["relative_ndcg_vs_phase3_cv"] = _relative_gain(selected, baseline)
    selected["relative_mrr_vs_phase3_cv"] = (
        float(selected["summary"]["mean"]["mrr"]) - float(baseline["summary"]["mean"]["mrr"])
    ) / float(baseline["summary"]["mean"]["mrr"])
    command = " ".join(["uv", "run", "scripts/phase5_development.py", *sys.argv[1:]])
    _write_outputs(
        cases=cases,
        results=results,
        selected=selected,
        index_stats=index_stats,
        corpus_hash=corpus_sha256(CORPUS_ROOT),
        command=command,
        output_root=OUTPUT_ROOT,
    )
    print(json.dumps({"selected": selected["name"], "metrics": selected["summary"]["mean"], "case_count": len(cases)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-models", action="store_true", help="Require all embedding models to be local.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/private/tmp/retrieval-system-phase5-embeddings"),
        help="Resumable local embedding cache directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
