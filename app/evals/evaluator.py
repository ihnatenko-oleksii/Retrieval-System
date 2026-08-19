import json
import logging
import math
import os
from typing import Any

from app.core.runtime_config import RetrievalConfig
from app.generation.generator import Generator
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.vector_store.bm25_store import BM25Store
from app.vector_store.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        top_k: int = 5,
        model_name: str | None = None,
        config: RetrievalConfig | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        reranker: Reranker | None = None,
        skip_generation: bool = False,
    ):
        # `config` takes precedence over stand-alone top_k / model_name so the
        # tuning UI can drive full trial configs.
        self.config = config
        self.top_k = config.top_k if config is not None else top_k
        effective_model = (config.llm_model if config is not None else None) or model_name

        # Retrieval quality (recall/precision/mrr/ndcg) only needs the
        # retriever; keyword_hit_rate needs an LLM. Decoupling the two lets
        # retrieval-only benchmarking run without Ollama available.
        self.skip_generation = skip_generation

        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.retriever = Retriever(self.vector_store, self.bm25_store, reranker=reranker)
        self.generator = Generator(model_name=effective_model)

    def _load_cases(self, jsonl_path: str) -> list[dict[str, Any]]:
        with open(jsonl_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _norm_source(self, value: str) -> str:
        if not value:
            return ""
        # Normalize slashes + case for robust matching across OS / ingestion formats.
        return str(value).replace("\\", "/").strip().lower()

    def _source_matches(self, expected_source: str, retrieved_meta: dict) -> bool:
        """
        Match eval `expected_source` against retrieved metadata.
        Supports:
        - full relative paths (preferred)
        - basename-only (legacy ingestions)
        - endswith matching for cases where expected_source includes folders
        """
        exp = self._norm_source(expected_source)
        if not exp:
            return False

        file_name = self._norm_source(retrieved_meta.get("file_name", ""))
        file_path = self._norm_source(retrieved_meta.get("file_path", ""))
        candidates = [c for c in (file_name, file_path) if c]
        if not candidates:
            return False

        exp_base = self._norm_source(os.path.basename(exp))
        for cand in candidates:
            if exp in cand:
                return True
            if exp_base and (exp_base == os.path.basename(cand)):
                return True
            if exp and cand.endswith(exp):
                return True
        return False

    def evaluate_cases(self, jsonl_path: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
        try:
            cases = self._load_cases(jsonl_path)
        except Exception as e:
            logger.error(f"Failed to read eval file: {e}")
            return {}, []

        if not cases:
            logger.warning("No evaluation cases found.")
            return {}, []

        metrics = {
            f"recall@{self.top_k}": 0.0,
            f"precision@{self.top_k}": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "keyword_hit_rate": 0.0,
        }
        case_details: list[dict[str, Any]] = []

        for case in cases:
            question = case.get("question", "")
            expected_source = case.get("expected_source", "")
            expected_keywords = case.get("expected_keywords", [])

            chunks = self.retriever.retrieve(question, top_k=self.top_k, config=self.config)
            retrieved_sources = [chunk[1].get("file_name", "") for chunk in chunks]

            relevant_ranks = []
            for rank, _src in enumerate(retrieved_sources, start=1):
                meta = chunks[rank - 1][1] if rank - 1 < len(chunks) else {}
                if self._source_matches(expected_source, meta):
                    relevant_ranks.append(rank)

            source_hit = bool(relevant_ranks)
            first_relevant_rank = relevant_ranks[0] if source_hit else None

            if source_hit:
                metrics[f"recall@{self.top_k}"] += 1.0
                metrics[f"precision@{self.top_k}"] += len(relevant_ranks) / self.top_k
                metrics["mrr"] += 1.0 / first_relevant_rank

                dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
                # Ideal DCG places every relevant chunk we actually found at the
                # front of the ranking. We don't know the true total number of
                # relevant chunks in the corpus, so the number found (already
                # capped at top_k by construction) is the best available bound.
                idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, len(relevant_ranks) + 1))
                metrics["ndcg"] += dcg / idcg

            case_keyword_score = None
            if not getattr(self, "skip_generation", False):
                answer_dict = self.generator.generate_answer(question, chunks)
                answer_text = answer_dict.get("final_answer", "").lower()

                case_keyword_score = 1.0
                if expected_keywords:
                    hits = sum(1 for kw in expected_keywords if str(kw).lower() in answer_text)
                    case_keyword_score = hits / len(expected_keywords)
                metrics["keyword_hit_rate"] += case_keyword_score

            case_details.append(
                {
                    "question": question,
                    "expected_source": expected_source,
                    "source_hit": source_hit,
                    "first_relevant_rank": first_relevant_rank,
                    "retrieved_sources": " | ".join(retrieved_sources),
                    "keyword_hit_score": round(case_keyword_score, 4) if case_keyword_score is not None else None,
                }
            )

        num_cases = len(cases)
        if getattr(self, "skip_generation", False):
            del metrics["keyword_hit_rate"]
        for k in metrics:
            metrics[k] = round(metrics[k] / num_cases, 4)

        return metrics, case_details

    def evaluate_file(self, jsonl_path: str) -> dict[str, float]:
        metrics, _ = self.evaluate_cases(jsonl_path)
        return metrics
