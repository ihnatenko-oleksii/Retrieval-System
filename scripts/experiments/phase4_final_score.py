"""Corrected one-time Phase 4 final-only scoring pass.

This pass is intentionally separate from selection. It is used only to replace
the pre-correction control comparison after the per-trial BM25 variant bug was
fixed; it does not inspect benchmark-v3 results when choosing anything.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from app.evals.benchmark_protocol import load_jsonl  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.retrieval.phase4 import Phase4Config, Phase4Retriever  # noqa: E402
from scripts.experiments.phase4_retrieval import (  # noqa: E402
    CORPUS_ROOT,
    OUTPUT_ROOT,
    V3_ROOT,
    _build_backend_index,
    _category_metrics,
    _config_payload,
    _evaluate,
)


def _config_from_payload(payload: dict) -> Phase4Config:
    values = dict(payload)
    values["stream_weights"] = tuple((str(name), float(weight)) for name, weight in values["stream_weights"].items())
    return Phase4Config(**values)


def main() -> None:
    output_path = OUTPUT_ROOT / "phase4_results.json"
    previous = json.loads(output_path.read_text(encoding="utf-8"))
    if previous.get("v3_used_for_selection") is not False:
        raise ValueError("refusing final scoring because the existing artifact does not prove v3 isolation")

    chunks = IngestionPipeline(chunk_size=1000, chunk_overlap=200).process_directory(str(CORPUS_ROOT))
    index = _build_backend_index(chunks, local_files_only=True)
    v3_cases = load_jsonl(V3_ROOT / "eval.jsonl")
    baseline_config = Phase4Config(
        top_k=5,
        candidate_depth=20,
        stream_weights=(("e5", 1.0),),
        fusion_strategy="weighted_linear",
        field_aware_bm25=False,
    )
    phase3_config = Phase4Config(
        top_k=5,
        candidate_depth=20,
        stream_weights=(("qwen", 0.7), ("bm25", 0.3)),
        fusion_strategy="weighted_linear",
        qwen_instruction_mode="generic",
        field_aware_bm25=False,
    )
    selected_config = _config_from_payload(previous["selected"]["config"])
    systems = {
        "original-e5-dense": (baseline_config, Phase4Retriever(index)),
        "phase3-qwen3-hybrid": (phase3_config, Phase4Retriever(index)),
        "phase4-final": (selected_config, Phase4Retriever(index)),
    }
    corrected: dict[str, dict] = {}
    for name, (config, engine) in systems.items():
        metrics, details = _evaluate(engine, config, v3_cases)
        corrected[name] = {
            "metrics": metrics,
            "category_metrics": _category_metrics(details),
            "config": _config_payload(config),
            "ranker_used": False,
        }

    baseline_ndcg = corrected["original-e5-dense"]["metrics"].get("ndcg", 0.0)
    baseline_mrr = corrected["original-e5-dense"]["metrics"].get("mrr", 0.0)
    final_ndcg = corrected["phase4-final"]["metrics"].get("ndcg", 0.0)
    final_mrr = corrected["phase4-final"]["metrics"].get("mrr", 0.0)
    previous["benchmark_v3_final"] = corrected
    previous["benchmark_v3_manifest_sha256"] = hashlib.sha256((V3_ROOT / "manifest.json").read_bytes()).hexdigest()
    previous["v3_scoring_call_count"] = 3
    previous["relative_improvement_percent"] = {
        "ndcg": round((final_ndcg - baseline_ndcg) / baseline_ndcg * 100, 4) if baseline_ndcg else 0.0,
        "mrr": round((final_mrr - baseline_mrr) / baseline_mrr * 100, 4) if baseline_mrr else 0.0,
    }
    previous["measurement_note"] = (
        "An initial pre-correction v3 control pass was discarded because plain BM25 could inherit a field-aware "
        "index. This corrected pass uses per-trial sparse variants; selection parameters were unchanged and v3 "
        "was not consulted for selection."
    )
    from scripts.experiments.phase4_retrieval import _write_outputs

    _write_outputs(previous)
    print(json.dumps(corrected, sort_keys=True))


if __name__ == "__main__":
    main()
