"""Run Phase 4 observed-data selection and the one-time benchmark-v3 score."""

# Imports intentionally follow the repository-root path bootstrap below.
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.evals.benchmark_protocol import load_jsonl  # noqa: E402
from app.evals.evaluator import Evaluator  # noqa: E402
from app.evals.experiment_runner import grouped_query_folds  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.retrieval.ltr import GroupedLTR, LTRFeatureExtractor  # noqa: E402
from app.retrieval.phase4 import (  # noqa: E402
    MODEL_ALIASES,
    QWEN_INSTRUCTIONS,
    ChromaPhase4Backend,
    Phase4Config,
    Phase4Index,
    Phase4Retriever,
    span_or_chunk_gain,
)
from app.vector_store.chroma_store import VectorStore  # noqa: E402


CORPUS_ROOT = REPO_ROOT / "docs" / "benchmark_corpus"
OLD_EVAL = REPO_ROOT / "docs" / "benchmark_eval.jsonl"
V2_EVAL = REPO_ROOT / "docs" / "benchmark_v2" / "eval.jsonl"
V3_ROOT = REPO_ROOT / "docs" / "benchmark_v3"
OUTPUT_ROOT = REPO_ROOT / "docs" / "retrieval_results"
INDEX_ROOT = REPO_ROOT / "storage" / "benchmark_experiments"
TOP_K = 5


def _corpus_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(file for file in path.rglob("*") if file.is_file() and not file.name.startswith(".")):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _find_index(model_name: str, *, instruction_mode: str | None = None) -> Path:
    matches: list[Path] = []
    for manifest_path in sorted(INDEX_ROOT.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != model_name:
            continue
        if manifest.get("chunk_size") != 1000 or manifest.get("chunk_overlap") != 200:
            continue
        if instruction_mode is not None and manifest.get("qwen_instruction_mode") != instruction_mode:
            continue
        matches.append(manifest_path.parent)
    if not matches:
        raise FileNotFoundError(
            f'precomputed dense index for model "{model_name}" and instruction "{instruction_mode}" not found '
            f"under {INDEX_ROOT}"
        )
    return matches[0] / "chroma"


def _build_backend_index(chunks: list[Any], *, local_files_only: bool) -> Phase4Index:
    del local_files_only  # The Chroma indexes already contain document vectors; loaders stay local in this path.
    qwen_store = VectorStore(
        persist_dir=str(_find_index(MODEL_ALIASES["qwen"], instruction_mode="generic")),
        embedding_model=MODEL_ALIASES["qwen"],
        query_instruction=QWEN_INSTRUCTIONS["generic"],
    )
    bge_store = VectorStore(
        persist_dir=str(_find_index(MODEL_ALIASES["bge"])),
        embedding_model=MODEL_ALIASES["bge"],
    )
    e5_store = VectorStore(
        persist_dir=str(_find_index(MODEL_ALIASES["e5"])),
        embedding_model=MODEL_ALIASES["e5"],
    )
    qwen_backends = {
        mode: ChromaPhase4Backend(qwen_store, query_instruction=instruction)
        for mode, instruction in QWEN_INSTRUCTIONS.items()
    }
    return Phase4Index.from_vector_backends(
        chunks,
        dense_backends={
            "qwen": qwen_backends,
            "bge": {"default": ChromaPhase4Backend(bge_store)},
            "e5": {"default": ChromaPhase4Backend(e5_store)},
        },
        bm25_config=Phase4Config(field_aware_bm25=True),
    )


def _observed_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in (OLD_EVAL, V2_EVAL):
        for case in load_jsonl(path):
            case_id = str(case["id"])
            if case_id in seen:
                raise ValueError(f"duplicate observed case ID: {case_id}")
            seen.add(case_id)
            cases.append(case)
    return cases


def _write_temp_eval(cases: list[dict[str, Any]]) -> Path:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n")
    return Path(handle.name)


def _make_evaluator(engine: Phase4Retriever, config: Phase4Config) -> Evaluator:
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = config
    evaluator.top_k = config.top_k
    evaluator.skip_generation = True
    evaluator.retriever = engine
    evaluator.generator = None
    return evaluator


def _evaluate(
    engine: Phase4Retriever,
    config: Phase4Config,
    cases: list[dict[str, Any]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    eval_path = _write_temp_eval(cases)
    try:
        return _make_evaluator(engine, config).evaluate_cases(str(eval_path))
    finally:
        eval_path.unlink(missing_ok=True)


def _category_metrics(details: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for detail in details:
        grouped.setdefault(str(detail.get("category", "uncategorized")), []).append(detail)
    return {
        category: {
            "cases": len(rows),
            "recall": round(mean(float(row.get("recall", 0.0)) for row in rows), 4),
            "precision": round(mean(float(row.get("precision", 0.0)) for row in rows), 4),
            "mrr": round(mean(float(row.get("mrr", 0.0)) for row in rows), 4),
            "ndcg": round(mean(float(row.get("ndcg", 0.0)) for row in rows), 4),
        }
        for category, rows in sorted(grouped.items())
    }


def _fold_summary(details: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = [str(case["id"]) for case in cases]
    folds = grouped_query_folds(case_ids, n_splits=5, seed=1729)
    by_id = {str(detail["case_id"]): detail for detail in details}
    fold_rows: list[dict[str, Any]] = []
    for fold_index, (_, validation_indexes) in enumerate(folds, start=1):
        validation_ids = {case_ids[index] for index in validation_indexes}
        rows = [by_id[case_id] for case_id in validation_ids]
        fold_rows.append(
            {
                "fold": fold_index,
                "validation_case_ids": sorted(validation_ids),
                "metrics": {
                    "recall": round(mean(float(row.get("recall", 0.0)) for row in rows), 4),
                    "precision": round(mean(float(row.get("precision", 0.0)) for row in rows), 4),
                    "mrr": round(mean(float(row.get("mrr", 0.0)) for row in rows), 4),
                    "ndcg": round(mean(float(row.get("ndcg", 0.0)) for row in rows), 4),
                },
                "category_metrics": _category_metrics(rows),
            }
        )
    fold_metrics = [row["metrics"] for row in fold_rows]
    latency = [float(detail.get("retrieval_latency_ms", 0.0)) for detail in details]
    escalated = [bool(detail.get("escalated", False)) for detail in details]
    return {
        "n_splits": 5,
        "folds": fold_rows,
        "mean": {
            key: round(mean(float(metrics[key]) for metrics in fold_metrics), 4)
            for key in ("recall", "precision", "mrr", "ndcg")
        },
        "std": {
            key: round(pstdev(float(metrics[key]) for metrics in fold_metrics), 4)
            for key in ("recall", "precision", "mrr", "ndcg")
        },
        "per_category": _category_metrics(details),
        "latency_ms": {
            "mean": round(mean(latency), 3) if latency else 0.0,
            "p95": round(sorted(latency)[min(len(latency) - 1, math.ceil(len(latency) * 0.95) - 1)], 3)
            if latency
            else 0.0,
        },
        "escalation_rate_percent": round(100 * mean(escalated), 4) if escalated else 0.0,
    }


def _config_payload(config: Phase4Config) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.__dict__.items()
        if key not in {"stream_weights"}
    } | {"stream_weights": dict(config.stream_weights)}


def _candidate_configs() -> list[tuple[str, Phase4Config]]:
    return [
        (
            "phase4-multi-static-rrf",
            Phase4Config(
                stream_weights=(("qwen", 0.25), ("bge", 0.25), ("e5", 0.2), ("bm25", 0.3)),
                fusion_strategy="rrf",
                field_aware_bm25=True,
                candidate_depth=50,
            ),
        ),
        (
            "phase4-multi-weighted-router",
            Phase4Config(
                stream_weights=(("qwen", 0.3), ("bge", 0.25), ("e5", 0.2), ("bm25", 0.25)),
                fusion_strategy="weighted_rrf",
                router_on=True,
                field_aware_bm25=True,
                candidate_depth=50,
            ),
        ),
        (
            "phase4-routed-instruction-field-bm25",
            Phase4Config(
                stream_weights=(("qwen", 0.25), ("bge", 0.2), ("e5", 0.15), ("bm25", 0.4)),
                fusion_strategy="weighted_rrf",
                router_on=True,
                qwen_instruction_mode="routed",
                qwen_instruction_routing=True,
                field_aware_bm25=True,
                bm25_k1=1.4,
                bm25_b=0.65,
                candidate_depth=50,
            ),
        ),
        (
            "phase4-routed-prf-cascade",
            Phase4Config(
                stream_weights=(("qwen", 0.3), ("bge", 0.2), ("e5", 0.2), ("bm25", 0.3)),
                fusion_strategy="weighted_rrf",
                router_on=True,
                cascade_on=True,
                cascade_initial_depth=20,
                qwen_instruction_mode="routed",
                qwen_instruction_routing=True,
                field_aware_bm25=True,
                hierarchical_on=True,
                prf_on=True,
                candidate_depth=50,
            ),
        ),
    ]


def _ranked_metrics(case: dict[str, Any], records: list[dict[str, Any]], top_k: int) -> dict[str, float]:
    gains: list[float] = []
    relevant: list[int] = []
    matched: set[str] = set()
    for rank, record in enumerate(records[:top_k], start=1):
        label_id, gain = span_or_chunk_gain(case, record.get("metadata", {}))
        gains.append(gain)
        if gain > 0:
            relevant.append(rank)
            if label_id:
                matched.add(label_id)
    from app.evals.evaluator import Evaluator

    span_labels = Evaluator._parse_span_relevance(case)
    labels = Evaluator._parse_relevance(case)
    if span_labels is not None:
        relevant_ids = {str(label["span_id"]) for label in span_labels if float(label["gain"]) > 0}
        ideal = sorted((float(label["gain"]) for label in span_labels), reverse=True)[:top_k]
    elif labels is not None:
        relevant_ids = {label_id for label_id, gain in labels.items() if gain > 0}
        ideal = sorted((labels[label_id] for label_id in relevant_ids), reverse=True)[:top_k]
    else:
        relevant_ids = {"source"} if relevant else set()
        ideal = [1.0] * min(len(relevant_ids), top_k)
    idcg = Evaluator._dcg(ideal)
    return {
        "recall": len(matched & relevant_ids) / len(relevant_ids) if relevant_ids else (1.0 if relevant else 0.0),
        "precision": len(relevant) / top_k,
        "mrr": 1.0 / relevant[0] if relevant else 0.0,
        "ndcg": Evaluator._dcg(gains) / idcg if idcg else 0.0,
    }


def _ranker_cv(
    index: Phase4Index,
    config: Phase4Config,
    cases: list[dict[str, Any]],
) -> tuple[dict[str, Any], GroupedLTR | None]:
    engine = Phase4Retriever(index)
    examples: list[dict[str, Any]] = []
    for case in cases:
        engine.retrieve(str(case.get("question", "")), config=config)
        records = list(engine.last_trace.get("candidate_features", []))
        feature_rows = [LTRFeatureExtractor.extract(str(case.get("question", "")), record) for record in records]
        labels = [float(span_or_chunk_gain(case, record.get("metadata", {}))[1]) for record in records]
        examples.append({"case": case, "records": records, "features": feature_rows, "labels": labels})

    row_query_ids = [str(example["case"]["id"]) for example in examples for _ in example["labels"]]
    folds = grouped_query_folds(row_query_ids, n_splits=5, seed=1729)
    row_lookup = [
        (example_index, local_index)
        for example_index, example in enumerate(examples)
        for local_index in range(len(example["labels"]))
    ]
    fold_summaries: list[dict[str, Any]] = []
    for fold_index, (train_rows, validation_rows) in enumerate(folds, start=1):
        ranker = GroupedLTR(model_name="pairwise-linear", random_state=1729)
        train_lookup = [row_lookup[row_index] for row_index in train_rows]
        if not train_lookup:
            continue
        ranker.fit(
            LTRFeatureExtractor.matrix(
                [examples[example_index]["features"][local_index] for example_index, local_index in train_lookup]
            ),
            np.asarray(
                [examples[example_index]["labels"][local_index] for example_index, local_index in train_lookup],
                dtype=float,
            ),
            [row_query_ids[row_index] for row_index in train_rows],
        )
        validation_ids = {row_query_ids[row_index] for row_index in validation_rows}
        case_metrics: list[dict[str, float]] = []
        for example in examples:
            if str(example["case"]["id"]) not in validation_ids or not example["records"]:
                continue
            predictions = ranker.predict(LTRFeatureExtractor.matrix(example["features"]))
            ranked = [
                record
                for _, record in sorted(
                    zip(predictions, example["records"], strict=True),
                    key=lambda value: (-float(value[0]), str(value[1].get("metadata", {}).get("chunk_id", ""))),
                )
            ]
            case_metrics.append(_ranked_metrics(example["case"], ranked, config.top_k))
        fold_summaries.append(
            {
                "fold": fold_index,
                "validation_case_ids": sorted(validation_ids),
                "metrics": {
                    key: round(mean(metric[key] for metric in case_metrics), 4)
                    for key in ("recall", "precision", "mrr", "ndcg")
                },
            }
        )

    final_ranker = GroupedLTR(model_name="pairwise-linear", random_state=1729)
    all_features = [feature for example in examples for feature in example["features"]]
    all_labels = [label for example in examples for label in example["labels"]]
    final_ranker.fit(LTRFeatureExtractor.matrix(all_features), np.asarray(all_labels), row_query_ids)
    return (
        {
            "backend": final_ranker.backend_name,
            "grouped_cv": {
                "n_splits": 5,
                "folds": fold_summaries,
                "mean": {
                    key: round(mean(fold["metrics"][key] for fold in fold_summaries), 4)
                    for key in ("recall", "precision", "mrr", "ndcg")
                },
                "std": {
                    key: round(pstdev(fold["metrics"][key] for fold in fold_summaries), 4)
                    for key in ("recall", "precision", "mrr", "ndcg")
                },
            },
            "lambda_mart_status": "blocked: xgboost is not installed; pairwise-linear grouped ranking used as the equivalent fallback",
        },
        final_ranker,
    )


def _write_outputs(payload: dict[str, Any]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "phase4_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for name, result in payload["development_candidates"].items():
        summary = result["cv"]
        rows.append(
            {
                "name": name,
                "ndcg_mean": summary["mean"]["ndcg"],
                "ndcg_std": summary["std"]["ndcg"],
                "mrr_mean": summary["mean"]["mrr"],
                "mrr_std": summary["std"]["mrr"],
                "latency_ms_mean": summary["latency_ms"]["mean"],
                "escalation_rate_percent": summary["escalation_rate_percent"],
            }
        )
    with (OUTPUT_ROOT / "phase4_development.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Phase 4 Retrieval Results",
        "",
        f"Observed development cases: {payload['observed_case_count']} (legacy DEV/TEST + benchmark-v2).",
        "Latency is measured per query in the shared process; later candidates may use warm stream-query caches.",
        "Benchmark-v3 was sealed before selection and was not used for tuning.",
        "",
        "## Development selection",
        "",
        f"Selected architecture: `{payload['selected']['name']}`.",
        f"Grouped-CV nDCG: {payload['selected']['cv']['mean']['ndcg']} +/- {payload['selected']['cv']['std']['ndcg']}; "
        f"MRR: {payload['selected']['cv']['mean']['mrr']} +/- {payload['selected']['cv']['std']['mrr']}.",
        "",
        "| Candidate | nDCG mean | nDCG std | MRR mean | MRR std | latency ms | escalation % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['name']} | {row['ndcg_mean']:.4f} | {row['ndcg_std']:.4f} | {row['mrr_mean']:.4f} | "
            f"{row['mrr_std']:.4f} | {row['latency_ms_mean']:.1f} | {row['escalation_rate_percent']:.1f} |"
        )
    report.extend(
        [
            "",
            "## One-time benchmark-v3 comparison",
            "",
            "| System | nDCG | MRR | Recall@5 | Precision@5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, result in payload["benchmark_v3_final"].items():
        metrics = result["metrics"]
        report.append(
            f"| {name} | {metrics.get('ndcg', 0.0):.4f} | {metrics.get('mrr', 0.0):.4f} | "
            f"{metrics.get('recall@5', 0.0):.4f} | {metrics.get('precision@5', 0.0):.4f} |"
        )
    report.extend(
        [
            "",
            f"v3 scoring calls recorded: {payload['v3_scoring_call_count']} (required: 3).",
            f"Relative nDCG vs original E5: {payload['relative_improvement_percent']['ndcg']:.2f}%.",
            f"Relative MRR vs original E5: {payload['relative_improvement_percent']['mrr']:.2f}%.",
            "",
            *( [f"Measurement note: {payload['measurement_note']}", ""] if payload.get("measurement_note") else [] ),
            "## Blocked or unavailable components",
            "",
            f"{payload['blocked_components']}",
        ]
    )
    (OUTPUT_ROOT / "phase4_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-models", action="store_true")
    args = parser.parse_args()
    if args.offline_models:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    v3_manifest = json.loads((V3_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if v3_manifest.get("selection_eligible") is not False or v3_manifest.get("sealed_before_phase4_selection") is not True:
        raise ValueError("benchmark-v3 is not sealed as a final-only artifact")
    if v3_manifest.get("corpus_sha256") != _corpus_sha256(CORPUS_ROOT):
        raise ValueError("benchmark-v3 corpus fingerprint no longer matches the source corpus")

    chunks = IngestionPipeline(chunk_size=1000, chunk_overlap=200).process_directory(str(CORPUS_ROOT))
    observed = _observed_cases()
    v3_cases = load_jsonl(V3_ROOT / "eval.jsonl")
    index = _build_backend_index(chunks, local_files_only=args.offline_models)
    development: dict[str, Any] = {}
    candidate_configs = dict(_candidate_configs())
    baseline_config = Phase4Config(
        top_k=TOP_K,
        candidate_depth=20,
        stream_weights=(("e5", 1.0),),
        fusion_strategy="weighted_linear",
        field_aware_bm25=False,
    )
    phase3_config = Phase4Config(
        top_k=TOP_K,
        candidate_depth=20,
        stream_weights=(("qwen", 0.7), ("bm25", 0.3)),
        fusion_strategy="weighted_linear",
        qwen_instruction_mode="generic",
        field_aware_bm25=False,
    )
    control_results: dict[str, Any] = {}
    for name, config in (("original-e5-dense", baseline_config), ("phase3-qwen3-hybrid", phase3_config)):
        metrics, details = _evaluate(Phase4Retriever(index), config, observed)
        control_results[name] = {"metrics": metrics, "category_metrics": _category_metrics(details), "config": _config_payload(config)}

    for name, config in candidate_configs.items():
        print(f"[phase4-dev] {name}", flush=True)
        metrics, details = _evaluate(Phase4Retriever(index), config, observed)
        cv = _fold_summary(details, observed)
        development[name] = {
            "metrics": metrics,
            "category_metrics": _category_metrics(details),
            "cv": cv,
            "config": _config_payload(config),
        }
        print(f"[phase4-dev] {name} {metrics}", flush=True)

    selected_name = max(
        development,
        key=lambda name: (
            development[name]["cv"]["mean"]["ndcg"],
            development[name]["cv"]["mean"]["mrr"],
            development[name]["cv"]["mean"]["recall"],
            -development[name]["cv"]["latency_ms"]["mean"],
        ),
    )
    selected_config = candidate_configs[selected_name]
    ranker_report, ranker = _ranker_cv(index, selected_config, observed)
    base_cv = development[selected_name]["cv"]["mean"]
    ranker_cv_mean = ranker_report["grouped_cv"]["mean"]
    ranker_used = ranker_cv_mean["ndcg"] > base_cv["ndcg"] and ranker_cv_mean["mrr"] >= base_cv["mrr"]
    final_ranker = ranker if ranker_used else None
    final_config = selected_config

    # This is the only place benchmark-v3 is evaluated. Keep the three calls explicit.
    final_systems = {
        "original-e5-dense": (baseline_config, Phase4Retriever(index)),
        "phase3-qwen3-hybrid": (phase3_config, Phase4Retriever(index)),
        "phase4-final": (final_config, Phase4Retriever(index, ranker=final_ranker)),
    }
    benchmark_v3_final: dict[str, Any] = {}
    for name, (config, engine) in final_systems.items():
        metrics, details = _evaluate(engine, config, v3_cases)
        benchmark_v3_final[name] = {
            "metrics": metrics,
            "category_metrics": _category_metrics(details),
            "config": _config_payload(config),
            "ranker_used": bool(final_ranker) if name == "phase4-final" else False,
        }

    baseline_ndcg = benchmark_v3_final["original-e5-dense"]["metrics"].get("ndcg", 0.0)
    baseline_mrr = benchmark_v3_final["original-e5-dense"]["metrics"].get("mrr", 0.0)
    final_ndcg = benchmark_v3_final["phase4-final"]["metrics"].get("ndcg", 0.0)
    final_mrr = benchmark_v3_final["phase4-final"]["metrics"].get("mrr", 0.0)
    payload = {
        "protocol": "Phase 4 observed-data grouped cross-validation followed by one-time benchmark-v3 scoring",
        "corpus_sha256": _corpus_sha256(CORPUS_ROOT),
        "observed_case_count": len(observed),
        "observed_case_ids": [str(case["id"]) for case in observed],
        "benchmark_v3_case_count": len(v3_cases),
        "benchmark_v3_manifest_sha256": hashlib.sha256((V3_ROOT / "manifest.json").read_bytes()).hexdigest(),
        "v3_used_for_selection": False,
        "development_controls": control_results,
        "development_candidates": development,
        "selected": {
            "name": selected_name,
            "config": _config_payload(selected_config),
            "cv": development[selected_name]["cv"],
        },
        "ranker": ranker_report | {"used_for_final": ranker_used},
        "benchmark_v3_final": benchmark_v3_final,
        "v3_scoring_call_count": len(final_systems),
        "relative_improvement_percent": {
            "ndcg": round((final_ndcg - baseline_ndcg) / baseline_ndcg * 100, 4) if baseline_ndcg else 0.0,
            "mrr": round((final_mrr - baseline_mrr) / baseline_mrr * 100, 4) if baseline_mrr else 0.0,
        },
        "blocked_components": (
            index.model_errors
            | {
                "lambda_mart": "blocked: xgboost is not installed; grouped pairwise-linear ranking was evaluated",
                "qwen_hyde": "blocked unless a local hypothetical-answer provider is explicitly configured",
                "qwen_fast_reranker": "not selected: the Phase 3 CPU smoke test measured 51.58 seconds for 16 real chunks",
            }
        ),
    }
    _write_outputs(payload)
    print(json.dumps(payload["benchmark_v3_final"], sort_keys=True))


if __name__ == "__main__":
    main()
