"""Run the integrity-preserving DEV/TEST retrieval experiment protocol."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / "docs" / "benchmark_corpus")
    parser.add_argument("--eval", type=Path, default=REPO_ROOT / "docs" / "benchmark_eval.jsonl")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "docs" / "benchmark_splits")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "retrieval_results")
    parser.add_argument("--index-root", type=Path, default=REPO_ROOT / "storage" / "benchmark_experiments")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="After non-LLM DEV selection, evaluate query rewriting and expansion variants on DEV.",
    )
    parser.add_argument(
        "--offline-models",
        action="store_true",
        help="Prevent model loaders from making network requests; unavailable models are recorded as failed experiments.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model used only for the post-retrieval rewrite/expansion DEV experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.offline_models:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from app.evals.experiment_runner import run_protocol

    command = shlex.join(["uv", "run", "scripts/retrieval_experiments.py", *sys.argv[1:]])
    result = run_protocol(
        corpus_dir=args.corpus,
        eval_path=args.eval,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        index_root=args.index_root,
        seed=args.seed,
        include_llm=args.include_llm,
        llm_model=args.llm_model,
        command=command,
    )
    selected = result["selected"]
    print(f"Selected DEV configuration: {selected.name}")
    print(f"Wrote experiment artifacts to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
