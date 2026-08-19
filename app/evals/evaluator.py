import json
import logging
import math
import os
from typing import Any

from app.core.runtime_config import RetrievalConfig
from app.generation.generator import Generator
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.retrieval.rewriter import QueryRewriter
from app.vector_store.bm25_store import BM25Store
from app.vector_store.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate retrieval with document-, chunk-, or graded relevance labels.

    New benchmark cases should use a ``relevance`` mapping from stable chunk
    IDs to positive gains, for example ``{"atlas/api-retries.md::1": 3}``.
    The older ``expected_source`` field remains supported for existing users
    and tests, but it can only provide binary, source-level relevance.
    """

    def __init__(
        self,
        top_k: int = 5,
        model_name: str | None = None,
        config: RetrievalConfig | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
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
        self.retriever = Retriever(
            self.vector_store,
            self.bm25_store,
            reranker=reranker,
            rewriter=rewriter or QueryRewriter(model_name=effective_model),
        )
        self.generator = Generator(model_name=effective_model)

    def _load_cases(self, jsonl_path: str) -> list[dict[str, Any]]:
        with open(jsonl_path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    @staticmethod
    def _norm_source(value: Any) -> str:
        if not value:
            return ""
        # Normalize slashes + case for robust matching across OS / ingestion formats.
        return str(value).replace("\\", "/").strip().lower()

    @classmethod
    def chunk_id_for_metadata(cls, retrieved_meta: dict[str, Any]) -> str:
        """Return the stable evaluation ID exposed by a retrieved chunk."""
        explicit_chunk_id = retrieved_meta.get("chunk_id") if retrieved_meta else None
        if explicit_chunk_id:
            return cls._norm_source(explicit_chunk_id)

        source = (retrieved_meta.get("file_name") or retrieved_meta.get("file_path", "")) if retrieved_meta else ""
        source = cls._norm_source(source)
        chunk_index = retrieved_meta.get("chunk_index") if retrieved_meta else None
        if source and chunk_index is not None:
            return f"{source}::{chunk_index}"
        return source

    @classmethod
    def _chunk_id_aliases(cls, retrieved_meta: dict[str, Any]) -> set[str]:
        """Return stable and legacy aliases that may identify one chunk."""
        aliases: set[str] = set()
        explicit_chunk_id = retrieved_meta.get("chunk_id") if retrieved_meta else None
        if explicit_chunk_id:
            aliases.add(cls._norm_source(explicit_chunk_id))

        chunk_index = retrieved_meta.get("chunk_index") if retrieved_meta else None
        for source_key in ("file_name", "file_path"):
            source = cls._norm_source(retrieved_meta.get(source_key, "") if retrieved_meta else "")
            if source and chunk_index is not None:
                aliases.add(f"{source}::{chunk_index}")
        return aliases

    @staticmethod
    def _source_matches(expected_source: str, retrieved_meta: dict) -> bool:
        """
        Match eval `expected_source` against retrieved metadata.
        Supports full relative paths, basename-only legacy ingestions, and
        suffix matching when the expected value includes folders.
        """
        exp = Evaluator._norm_source(expected_source)
        if not exp:
            return False

        file_name = Evaluator._norm_source(retrieved_meta.get("file_name", ""))
        file_path = Evaluator._norm_source(retrieved_meta.get("file_path", ""))
        candidates = [c for c in (file_name, file_path) if c]
        if not candidates:
            return False

        exp_base = Evaluator._norm_source(os.path.basename(exp))
        for cand in candidates:
            if exp in cand:
                return True
            if exp_base and exp_base == os.path.basename(cand):
                return True
            if cand.endswith(exp):
                return True
        return False

    @classmethod
    def _parse_relevance(cls, case: dict[str, Any]) -> dict[str, float] | None:
        """Parse new graded labels and the binary chunk-ID shorthand."""
        raw_relevance = case.get("relevance")
        raw_chunk_ids = case.get("expected_chunk_ids")
        if raw_relevance is None and raw_chunk_ids is None:
            return None

        labels: dict[str, float] = {}
        if isinstance(raw_relevance, dict):
            for chunk_id, gain in raw_relevance.items():
                try:
                    numeric_gain = float(gain)
                except (TypeError, ValueError):
                    continue
                if numeric_gain > 0:
                    labels[cls._norm_source(chunk_id)] = numeric_gain
        elif isinstance(raw_relevance, list):
            for entry in raw_relevance:
                if isinstance(entry, str):
                    labels[cls._norm_source(entry)] = 1.0
                    continue
                if not isinstance(entry, dict):
                    continue
                chunk_id = entry.get("chunk_id") or entry.get("id")
                if not chunk_id:
                    continue
                gain = entry.get("gain", entry.get("relevance", 1))
                try:
                    numeric_gain = float(gain)
                except (TypeError, ValueError):
                    continue
                if numeric_gain > 0:
                    labels[cls._norm_source(chunk_id)] = numeric_gain

        if raw_chunk_ids is not None:
            if isinstance(raw_chunk_ids, dict):
                for chunk_id, gain in raw_chunk_ids.items():
                    try:
                        numeric_gain = float(gain)
                    except (TypeError, ValueError):
                        numeric_gain = 1.0
                    if numeric_gain > 0:
                        labels[cls._norm_source(chunk_id)] = numeric_gain
            elif isinstance(raw_chunk_ids, list):
                for chunk_id in raw_chunk_ids:
                    if isinstance(chunk_id, str) and chunk_id.strip():
                        labels[cls._norm_source(chunk_id)] = 1.0

        return labels

    @classmethod
    def _match_label(
        cls, labels: dict[str, float], retrieved_meta: dict[str, Any]
    ) -> tuple[str | None, float]:
        aliases = cls._chunk_id_aliases(retrieved_meta)
        for label_id, gain in labels.items():
            if label_id in aliases:
                return label_id, gain
        return None, 0.0

    @staticmethod
    def _dcg(gains: list[float]) -> float:
        return sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))

    @classmethod
    def _expected_sources(cls, case: dict[str, Any]) -> list[str]:
        expected_sources = case.get("expected_sources")
        if expected_sources is None:
            expected_source = case.get("expected_source", "")
            return [expected_source] if isinstance(expected_source, str) else list(expected_source or [])
        if isinstance(expected_sources, str):
            return [expected_sources]
        return list(expected_sources)

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

        for case_index, case in enumerate(cases, start=1):
            question = case.get("question", "")
            expected_source = case.get("expected_source", "")
            expected_keywords = case.get("expected_keywords", [])
            labels = self._parse_relevance(case)
            expected_sources = self._expected_sources(case)

            chunks = self.retriever.retrieve(question, top_k=self.top_k, config=self.config)
            retrieved_sources = [chunk[1].get("file_name", "") for chunk in chunks]
            retrieved_chunk_ids = [self.chunk_id_for_metadata(chunk[1]) for chunk in chunks]

            relevant_ranks: list[int] = []
            matched_label_ids: list[str] = []
            gains_by_rank: list[float] = []
            case_recall = 0.0
            case_precision = 0.0
            case_mrr = 0.0
            case_ndcg = 0.0
            for rank, (_, meta, _) in enumerate(chunks, start=1):
                if labels is not None:
                    matched_label_id, gain = self._match_label(labels, meta)
                    gains_by_rank.append(gain)
                    if matched_label_id is not None and gain > 0:
                        relevant_ranks.append(rank)
                        matched_label_ids.append(matched_label_id)
                else:
                    is_relevant = any(self._source_matches(source, meta) for source in expected_sources)
                    gains_by_rank.append(1.0 if is_relevant else 0.0)
                    if is_relevant:
                        relevant_ranks.append(rank)

            source_hit = bool(relevant_ranks)
            first_relevant_rank = relevant_ranks[0] if source_hit else None

            if labels is not None:
                relevant_label_ids = {label_id for label_id, gain in labels.items() if gain > 0}
                matched_ids = set(matched_label_ids)
                case_recall = (
                    len(matched_ids & relevant_label_ids) / len(relevant_label_ids) if relevant_label_ids else 0.0
                )
                case_precision = len(relevant_ranks) / self.top_k

                if first_relevant_rank is not None:
                    case_mrr = 1.0 / first_relevant_rank

                ideal_gains = sorted((labels[label_id] for label_id in relevant_label_ids), reverse=True)
                idcg = self._dcg(ideal_gains[: self.top_k])
                case_ndcg = self._dcg(gains_by_rank[: self.top_k]) / idcg if idcg else 0.0
            elif source_hit:
                # Legacy source-only cases cannot know how many relevant chunks
                # exist outside the retrieved list. Preserve their historical
                # binary recall and found-hit nDCG semantics.
                case_recall = 1.0
                case_precision = len(relevant_ranks) / self.top_k
                case_mrr = 1.0 / first_relevant_rank
                dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
                idcg = self._dcg([1.0] * len(relevant_ranks))
                case_ndcg = dcg / idcg if idcg else 0.0

            metrics[f"recall@{self.top_k}"] += case_recall
            metrics[f"precision@{self.top_k}"] += case_precision
            metrics["mrr"] += case_mrr
            metrics["ndcg"] += case_ndcg

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
                    "case_id": case.get("id", f"case-{case_index:03d}"),
                    "category": case.get("category", "uncategorized"),
                    "question": question,
                    "expected_source": expected_source,
                    "relevant_chunk_ids": " | ".join(sorted(labels or {})),
                    "retrieved_chunk_ids": " | ".join(retrieved_chunk_ids),
                    "source_hit": source_hit,
                    "first_relevant_rank": first_relevant_rank,
                    "recall": round(case_recall, 4),
                    "precision": round(case_precision, 4),
                    "mrr": round(case_mrr, 4),
                    "ndcg": round(case_ndcg, 4),
                    "retrieved_sources": " | ".join(retrieved_sources),
                    "keyword_hit_score": round(case_keyword_score, 4) if case_keyword_score is not None else None,
                }
            )

        num_cases = len(cases)
        if getattr(self, "skip_generation", False):
            del metrics["keyword_hit_rate"]
        for key in metrics:
            metrics[key] = round(metrics[key] / num_cases, 4)

        return metrics, case_details

    def evaluate_file(self, jsonl_path: str) -> dict[str, float]:
        metrics, _ = self.evaluate_cases(jsonl_path)
        return metrics
