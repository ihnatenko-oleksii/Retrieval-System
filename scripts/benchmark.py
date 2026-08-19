"""
Compare dense-only, hybrid (dense+BM25), and hybrid+reranked retrieval on a
fixed sample corpus and eval set, and write the results into docs/benchmark.md.

Usage:
    uv run scripts/benchmark.py
    uv run scripts/benchmark.py --skip-generation   # retrieval metrics only, no Ollama required
    uv run scripts/benchmark.py --top-k 5

Requires:
    - The configured embedding model downloadable/cached (for dense retrieval).
    - Ollama running locally with the configured model pulled, unless
      --skip-generation is passed (keyword_hit_rate is then omitted, since it
      needs a generated answer; recall/precision/mrr/ndcg do not).

This script intentionally ingests into an isolated storage path
(./storage/benchmark/) so it never touches a real corpus you may already have
indexed under ./storage.
"""

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_START = "<!-- BENCHMARK_RESULTS_START -->"
RESULTS_END = "<!-- BENCHMARK_RESULTS_END -->"

CONFIGS = {
    "A. Dense-only": {"dense_weight": 1.0, "sparse_weight": 0.0, "reranker_on": False},
    "B. Hybrid (dense + BM25)": {"dense_weight": 0.7, "sparse_weight": 0.3, "reranker_on": False},
    "C. Hybrid + reranking": {"dense_weight": 0.7, "sparse_weight": 0.3, "reranker_on": True},
}


def run(args: argparse.Namespace) -> None:
    from app.core.config import settings
    from app.core.runtime_config import RetrievalConfig
    from app.evals.evaluator import Evaluator
    from app.ingestion.pipeline import IngestionPipeline
    from app.retrieval.reranker import Reranker
    from app.vector_store.bm25_store import BM25Store
    from app.vector_store.chroma_store import VectorStore

    corpus_dir = Path(args.corpus).resolve()
    eval_path = Path(args.eval).resolve()
    bench_storage = REPO_ROOT / "storage" / "benchmark"

    if not corpus_dir.exists():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")
    if not eval_path.exists():
        raise SystemExit(f"Eval file not found: {eval_path}")

    if bench_storage.exists() and not args.keep_index:
        shutil.rmtree(bench_storage)
    bench_storage.mkdir(parents=True, exist_ok=True)

    # Isolate from any real index the caller may already have under ./storage.
    settings.vector_db_path = str(bench_storage / "chroma")

    print(f"Ingesting {corpus_dir} into isolated benchmark storage at {bench_storage} ...")
    pipeline = IngestionPipeline()
    chunks = pipeline.process_directory(str(corpus_dir))
    if not chunks:
        raise SystemExit("No chunks were produced from the benchmark corpus.")

    vector_store = VectorStore()
    bm25_store = BM25Store(persist_dir=str(bench_storage))
    if not args.keep_index:
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)
    print(f"Indexed {len(chunks)} chunks from {corpus_dir}.\n")

    needs_reranker = any(cfg["reranker_on"] for cfg in CONFIGS.values())
    reranker = Reranker(force_load=needs_reranker) if needs_reranker else None

    rows = []
    for label, overrides in CONFIGS.items():
        cfg = RetrievalConfig(
            top_k=args.top_k,
            dense_weight=overrides["dense_weight"],
            sparse_weight=overrides["sparse_weight"],
            reranker_on=overrides["reranker_on"],
            rerank_top_n=args.top_k,
            query_rewriting_on=False,
            query_expansion_on=False,
            llm_model=settings.llm_model,
        )
        evaluator = Evaluator(
            config=cfg,
            vector_store=vector_store,
            bm25_store=bm25_store,
            reranker=reranker,
            skip_generation=args.skip_generation,
        )
        print(f"Running: {label} ...")
        t0 = time.time()
        metrics, _ = evaluator.evaluate_cases(str(eval_path))
        elapsed = time.time() - t0
        if not metrics:
            raise SystemExit(f"Evaluation produced no metrics for config: {label}")
        rows.append((label, metrics, elapsed))
        print(f"  done in {elapsed:.1f}s: {metrics}")

    write_report(args, corpus_dir, eval_path, rows)


def write_report(args: argparse.Namespace, corpus_dir: Path, eval_path: Path, rows: list) -> None:
    from app.core.config import settings

    metric_keys = [f"recall@{args.top_k}", f"precision@{args.top_k}", "mrr", "ndcg"]
    if not args.skip_generation:
        metric_keys.append("keyword_hit_rate")

    header = "| Configuration | " + " | ".join(metric_keys) + " |"
    separator = "|" + "---|" * (len(metric_keys) + 1)
    lines = [header, separator]
    for label, metrics, _ in rows:
        cells = [str(metrics.get(k, "-")) for k in metric_keys]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    n_cases = sum(1 for _ in eval_path.open(encoding="utf-8") if _.strip())
    generation_note = (
        "keyword_hit_rate omitted (`--skip-generation`): no LLM calls were made."
        if args.skip_generation
        else "keyword_hit_rate requires a running local LLM via Ollama."
    )

    block = f"""{RESULTS_START}
_Last run: {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}_
_Corpus: `{corpus_dir.relative_to(REPO_ROOT)}` ({n_cases} eval cases) — embedding model: `{settings.embedding_model}`, top_k={args.top_k}._
_{generation_note}_

{table}
{RESULTS_END}"""

    benchmark_md = REPO_ROOT / "docs" / "benchmark.md"
    text = benchmark_md.read_text(encoding="utf-8")
    if RESULTS_START not in text or RESULTS_END not in text:
        raise SystemExit(f"{benchmark_md} is missing the {RESULTS_START}/{RESULTS_END} markers.")
    new_text = re.sub(
        re.escape(RESULTS_START) + r".*?" + re.escape(RESULTS_END),
        block.replace("\\", "\\\\"),
        text,
        flags=re.DOTALL,
    )
    benchmark_md.write_text(new_text, encoding="utf-8")
    print(f"\nWrote results into {benchmark_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--corpus", default=str(REPO_ROOT / "docs" / "benchmark_corpus"), help="Directory of documents to ingest."
    )
    parser.add_argument(
        "--eval", default=str(REPO_ROOT / "docs" / "benchmark_eval.jsonl"), help="Eval JSONL file to run."
    )
    parser.add_argument("--top-k", type=int, default=5, help="top_k used for every configuration.")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Compute recall/precision/mrr/ndcg only, without calling the LLM (no Ollama required).",
    )
    parser.add_argument(
        "--keep-index",
        action="store_true",
        help="Reuse an existing benchmark index instead of re-ingesting from scratch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
