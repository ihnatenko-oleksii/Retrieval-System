"""Record candidate-stream ranks for DEV cases missed by the frozen Phase 3 retriever."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=REPO_ROOT / "docs" / "benchmark_splits" / "dev.jsonl")
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "docs" / "benchmark_corpus")
    parser.add_argument("--index-root", type=Path, default=REPO_ROOT / "storage" / "benchmark_experiments")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "retrieval_results")
    parser.add_argument("--offline-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline_models:
        import os

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from app.evals.benchmark_protocol import load_jsonl
    from app.evals.evaluator import Evaluator
    from app.evals.experiment_runner import DEFAULT_QWEN3_EMBEDDING_MODEL, ExperimentRunner, ExperimentSpec

    spec = ExperimentSpec(
        name="qwen3-hybrid-generic-instruction",
        embedding_model=DEFAULT_QWEN3_EMBEDDING_MODEL,
        dense_weight=0.7,
        sparse_weight=0.3,
        qwen_instruction_mode="generic",
    )
    runner = ExperimentRunner(corpus_dir=args.corpus, index_root=args.index_root, skip_generation=True)
    cases = load_jsonl(args.eval)
    index, _ = runner._get_index(spec, cases)

    evaluator = Evaluator(
        config=spec.to_retrieval_config(),
        vector_store=index.vector_store,
        bm25_store=index.bm25_store,
        reranker=None,
        rewriter=None,
        skip_generation=True,
    )
    metrics, details = evaluator.evaluate_cases(str(args.eval))

    failures: list[dict[str, object]] = []
    for detail in details:
        if detail.get("first_relevant_rank") == 1:
            continue
        labels = Evaluator._parse_relevance(next(case for case in cases if str(case.get("id")) == detail["case_id"]))
        trace = json.loads(str(detail.get("retrieval_trace") or "[]"))
        by_id = {
            Evaluator.chunk_id_for_metadata(record.get("metadata", {})): record
            for record in trace
            if isinstance(record, dict)
        }
        relevant_ranks = []
        for label_id in sorted(labels or {}):
            record = by_id.get(label_id)
            relevant_ranks.append(
                {
                    "chunk_id": label_id,
                    "dense_rank": record.get("dense_rank") if record else None,
                    "bm25_rank": record.get("sparse_rank") if record else None,
                    "learned_sparse_rank": record.get("native_sparse_rank") if record else None,
                    "colbert_rank": record.get("late_interaction_rank") if record else None,
                    "reranker_rank": None,
                    "final_rank": record.get("candidate_rank") if record else None,
                }
            )
        top_candidates = []
        for record in trace[:5]:
            chunk_id = Evaluator.chunk_id_for_metadata(record.get("metadata", {}))
            top_candidates.append(
                {
                    "chunk_id": chunk_id,
                    "dense_rank": record.get("dense_rank"),
                    "bm25_rank": record.get("sparse_rank"),
                    "learned_sparse_rank": record.get("native_sparse_rank"),
                    "colbert_rank": record.get("late_interaction_rank"),
                    "reranker_rank": None,
                    "final_rank": record.get("candidate_rank"),
                }
            )
        failures.append(
            {
                "case_id": detail["case_id"],
                "category": detail.get("category"),
                "question": detail.get("question"),
                "first_relevant_rank": detail.get("first_relevant_rank"),
                "ndcg": detail.get("ndcg"),
                "relevant_ranks": relevant_ranks,
                "top_candidates": top_candidates,
            }
        )

    payload = {
        "configuration": spec.as_dict(),
        "phase": "dev_failure_analysis",
        "case_count": len(cases),
        "metrics": metrics,
        "missed_or_non_rank1_cases": failures,
        "stream_notes": {
            "dense_rank": "Qwen3 dense first-stage rank",
            "bm25_rank": "existing BM25 rank",
            "learned_sparse_rank": "null because the frozen Phase 3 configuration does not use native BGE learned sparse retrieval",
            "colbert_rank": "null because the frozen Phase 3 configuration does not use native BGE ColBERT",
            "reranker_rank": "null because the frozen Phase 3 configuration does not use a reranker",
            "final_rank": "Qwen3 hybrid candidate rank before any optional second-stage reranker",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase3_dev_failure_analysis.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase 3 DEV Failure Analysis",
        "",
        f"Configuration: `{spec.name}`; cases: {len(cases)}; misses or non-rank-1 cases: {len(failures)}.",
        "",
        "Ranks are recorded without using relevance labels as runtime features. Learned-sparse, ColBERT, and reranker ranks are explicitly null because they are not part of the frozen configuration.",
        "",
        "| Case | Category | First relevant rank | nDCG | Relevant chunk stream ranks |",
        "|---|---|---:|---:|---|",
    ]
    for failure in failures:
        rank_summary = "; ".join(
            f"{item['chunk_id']} dense={item['dense_rank']} bm25={item['bm25_rank']} final={item['final_rank']}"
            for item in failure["relevant_ranks"]
        )
        lines.append(
            f"| {failure['case_id']} | {failure['category']} | {failure['first_relevant_rank'] or '-'} | {failure['ndcg']} | {rank_summary} |"
        )
    (args.output_dir / "phase3_dev_failure_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(failures)} DEV failure cases to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
