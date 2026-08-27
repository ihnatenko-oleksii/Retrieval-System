"""Simple, leakage-safe retrieval primitives for Phase 5 experiments.

This module deliberately keeps the Phase 3 path small: one Qwen dense stream,
one BM25 stream, and a transparent fusion/ranking layer.  The experiment
script is responsible for selection and reporting; these classes only expose
deterministic records, scores, and source-span evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking.phase5 import build_phase5_chunks
from app.evals.evaluator import Evaluator
from app.evals.span_relevance import span_matches_metadata
from app.retrieval.ltr import GroupedLTR, LTRFeatureExtractor, grouped_query_folds

QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
E5_MODEL = "intfloat/multilingual-e5-base"
GENERIC_INSTRUCTION = (
    "Given a technical support or engineering question, retrieve passages that directly answer the question "
    "or contain the necessary implementation details."
)

INSTRUCTIONS: dict[str, str] = {
    "generic": GENERIC_INSTRUCTION,
    "precise": (
        "Given a technical question containing an identifier, code, number, boundary, or exact distinction, "
        "retrieve passages with the precise matching implementation detail."
    ),
    "semantic": (
        "Given a technical question, retrieve passages that explain the same underlying concept, behavior, or "
        "cause even when the wording differs."
    ),
    "context": (
        "Given an underspecified technical question, retrieve enough supporting context and distinctions to resolve "
        "the plausible interpretations."
    ),
    "multi": (
        "Given a multi-part technical question, retrieve every passage needed to answer the requested parts, "
        "preferring complementary evidence over repeated topical text."
    ),
}

_TECHNICAL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_.:/-][A-Za-z0-9]+)*")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "with",
    "why",
}


@dataclass(frozen=True)
class Phase5Config:
    name: str = "phase3-base"
    chunking: str = "character"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    instruction_mode: str = "generic"
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    technical_tokens: bool = False
    fusion: str = "weighted_linear"
    rrf_k: int = 60
    top_k: int = 5
    candidate_depth: int = 50
    ltr_on: bool = False

    def __post_init__(self) -> None:
        if self.chunk_size < 1 or self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("Phase 5 chunking requires size > overlap >= 0")
        if self.instruction_mode not in {"none", *INSTRUCTIONS, "routed", "learned"}:
            raise ValueError(f"unsupported instruction mode: {self.instruction_mode}")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("fusion weights cannot be negative")
        if self.dense_weight == 0 and self.sparse_weight == 0:
            raise ValueError("at least one fusion weight must be positive")
        if self.bm25_k1 <= 0 or not 0 <= self.bm25_b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        if self.fusion not in {"weighted_linear", "weighted_rrf"}:
            raise ValueError(f"unsupported fusion: {self.fusion}")
        if self.top_k < 1 or self.candidate_depth < self.top_k:
            raise ValueError("Phase 5 requires candidate_depth >= top_k >= 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.__dict__.items()
        }


def query_instruction_mode(query: str) -> str:
    """Choose an instruction using query text only, never benchmark metadata."""
    text = str(query or "")
    folded = text.casefold()
    identifiers = _TECHNICAL_TOKEN_RE.findall(text)
    if identifiers and any(any(char.isdigit() for char in token) or any(mark in token for mark in "_-./:") for token in identifiers):
        return "precise"
    if any(marker in folded for marker in ("difference", "distinguish", "what happens", "under what", "across")):
        return "context"
    if folded.count(" and ") >= 1 or any(marker in folded for marker in ("both", "each", "all parts", "respectively")):
        return "multi"
    if any(marker in folded for marker in ("explain", "why", "concept", "same idea", "how does")):
        return "semantic"
    return "generic"


def tokenize(text: str, *, technical: bool = False) -> list[str]:
    """Tokenize without stemming; technical mode preserves identifier forms and parts."""
    raw = (_TECHNICAL_TOKEN_RE if technical else _WORD_RE).findall(str(text or "").casefold())
    tokens: list[str] = []
    for token in raw:
        if len(token) <= 1 or token in _STOPWORDS:
            continue
        tokens.append(token)
        if technical and any(mark in token for mark in "_.:/-"):
            tokens.extend(part for part in re.split(r"[_.:/-]+", token) if len(part) > 1 and part not in _STOPWORDS)
    return tokens


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(np.min(values))
    high = float(np.max(values))
    if high == low:
        return np.ones_like(values, dtype=float)
    return (values - low) / (high - low)


class Phase5BM25:
    def __init__(self, records: list[dict[str, Any]], *, k1: float, b: float, technical_tokens: bool):
        self.records = records
        self.k1 = k1
        self.b = b
        self.technical_tokens = technical_tokens
        corpus = [tokenize(record["content"], technical=technical_tokens) for record in records]
        self.index = BM25Okapi(corpus, k1=k1, b=b)
        self._cache: dict[str, np.ndarray] = {}

    def scores(self, query: str) -> np.ndarray:
        if query not in self._cache:
            self._cache[query] = np.asarray(
                self.index.get_scores(tokenize(query, technical=self.technical_tokens)), dtype=float
            )
        return self._cache[query]


class Phase5EmbeddingRuntime:
    """Load one local embedding model and cache document/query representations."""

    def __init__(
        self,
        model_name: str = QWEN_MODEL,
        *,
        local_files_only: bool = True,
        cache_dir: Path | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.alias = "e5" if model_name.casefold().startswith("intfloat/") else "qwen"
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        # Phase 5 chunks are character-bounded and substantially shorter than
        # the model's 32k-token default.  Cap tokenization at 512 so CPU
        # experiments do not pay for an enormous padded attention window.
        self.model.max_seq_length = min(int(self.model.max_seq_length), 256)
        self.cache_dir = cache_dir.resolve() if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._document_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._query_cache: dict[tuple[str, ...], np.ndarray] = {}

    def _disk_cache_path(self, kind: str, key: tuple[str, ...]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            (self.model_name + "\0" + kind + "\0" + "\0".join(key)).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{kind}-{digest}.npy"

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        key = tuple(texts)
        if key not in self._document_cache:
            cache_path = self._disk_cache_path("documents", key)
            if cache_path is not None and cache_path.exists():
                self._document_cache[key] = np.load(cache_path, allow_pickle=False)
            else:
                prepared = [f"passage: {text}" for text in texts] if self.alias == "e5" else texts
                self._document_cache[key] = np.asarray(
                    self.model.encode(
                        prepared,
                        batch_size=64,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                    dtype=float,
                )
                if cache_path is not None:
                    np.save(cache_path, self._document_cache[key], allow_pickle=False)
        return self._document_cache[key]

    def encode_queries(self, queries: list[str], instruction_mode: str) -> np.ndarray:
        prepared: list[str] = []
        for query in queries:
            if self.alias == "e5":
                prepared.append(f"query: {query}")
            elif instruction_mode in {"none", ""}:
                prepared.append(query)
            else:
                instruction = INSTRUCTIONS.get(instruction_mode, GENERIC_INSTRUCTION)
                prepared.append(f"Instruct: {instruction}\nQuery: {query}")
        key = tuple(prepared)
        if key not in self._query_cache:
            cache_path = self._disk_cache_path(f"queries-{instruction_mode}", key)
            if cache_path is not None and cache_path.exists():
                self._query_cache[key] = np.load(cache_path, allow_pickle=False)
            else:
                self._query_cache[key] = np.asarray(
                    self.model.encode(
                        prepared,
                        batch_size=64,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ),
                    dtype=float,
                )
                if cache_path is not None:
                    np.save(cache_path, self._query_cache[key], allow_pickle=False)
        return self._query_cache[key]


class Phase5Index:
    def __init__(self, records: list[dict[str, Any]], runtime: Phase5EmbeddingRuntime):
        self.records = records
        self.runtime = runtime
        self.document_embeddings = runtime.encode_documents([str(record["content"]) for record in records])
        self._bm25: dict[tuple[float, float, bool], Phase5BM25] = {}
        self._query_vectors: dict[str, np.ndarray] = {}

    def prepare_queries(self, queries: list[str], modes: set[str]) -> None:
        preferred_order = ("generic", "none", "precise", "semantic", "context", "multi")
        ordered_modes = [mode for mode in preferred_order if mode in modes]
        ordered_modes.extend(sorted(modes - set(ordered_modes)))
        for mode in ordered_modes:
            self._query_vectors[mode] = self.runtime.encode_queries(queries, mode)

    def bm25(self, config: Phase5Config) -> Phase5BM25:
        key = (config.bm25_k1, config.bm25_b, config.technical_tokens)
        if key not in self._bm25:
            self._bm25[key] = Phase5BM25(
                self.records,
                k1=config.bm25_k1,
                b=config.bm25_b,
                technical_tokens=config.technical_tokens,
            )
        return self._bm25[key]

    def query_vectors(self, queries: list[str], mode: str) -> np.ndarray:
        if mode not in self._query_vectors:
            self._query_vectors[mode] = self.runtime.encode_queries(queries, mode)
        return self._query_vectors[mode]


def corpus_sha256(corpus_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in corpus_root.rglob("*") if path.is_file() and not path.name.startswith(".")):
        digest.update(path.relative_to(corpus_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_records(corpus_root: Path, config: Phase5Config) -> list[dict[str, Any]]:
    if config.chunking == "character" and config.chunk_size == 1000 and config.chunk_overlap == 200:
        return build_canonical_records(corpus_root)
    records: list[dict[str, Any]] = []
    for path in sorted(corpus_root.rglob("*.md")):
        document_id = path.relative_to(corpus_root).as_posix()
        text = path.read_text(encoding="utf-8")
        source_chunks = build_phase5_chunks(
            text,
            config.chunking,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        for index, chunk in enumerate(source_chunks):
            metadata = {
                "document_id": document_id,
                "chunk_id": f"{document_id}::{index}",
                "file_path": str(path),
                "file_name": document_id,
                "extension": ".md",
                "loader_type": "MarkdownTextLoader",
                "chunk_index": index,
                "source_char_start": chunk.start,
                "source_char_end": chunk.end,
                "heading_path": chunk.heading_path,
            }
            records.append({"id": f"{document_id}::{index}", "content": chunk.text, "metadata": metadata})
    if not records:
        raise ValueError(f"no Markdown chunks produced from {corpus_root}")
    return records


def build_canonical_records(corpus_root: Path) -> list[dict[str, Any]]:
    """Reproduce the Phase 3 1000/200 chunk inventory for baseline labels."""
    from app.ingestion.pipeline import IngestionPipeline

    records: list[dict[str, Any]] = []
    pipeline = IngestionPipeline(chunk_size=1000, chunk_overlap=200)
    for chunk in pipeline.process_directory(str(corpus_root)):
        metadata = chunk.metadata.model_dump()
        document_path = corpus_root / str(metadata["file_name"])
        source_text = document_path.read_text(encoding="utf-8")
        start = metadata.get("source_char_start")
        end = metadata.get("source_char_end")
        if start is None or end is None or int(end) <= int(start):
            first_line = next((line.strip() for line in chunk.content.splitlines() if line.strip()), "")
            start = source_text.find(first_line) if first_line else -1
            if start < 0:
                start = 0
            heading_match = re.search(r"(?m)^#{1,6}[ \t]+(.+?)[ \t]*$", chunk.content)
            if heading_match:
                heading_start = source_text.find(heading_match.group(0).strip())
                if heading_start >= 0:
                    start = heading_start
                    current_line_end = source_text.find("\n", start)
                    search_start = current_line_end + 1 if current_line_end >= 0 else len(source_text)
                    next_heading = re.search(r"(?m)^#{1,6}[ \t]+.+?[ \t]*$", source_text[search_start:])
                    end = search_start + next_heading.start() if next_heading else len(source_text)
                else:
                    end = min(len(source_text), start + max(1, len(chunk.content)))
            else:
                end = min(len(source_text), start + max(1, len(chunk.content)))
        metadata["source_char_start"] = int(start)
        metadata["source_char_end"] = int(min(len(source_text), end))
        records.append({"id": str(metadata.get("chunk_id") or chunk.id), "content": chunk.content, "metadata": metadata})
    if not records:
        raise ValueError(f"no canonical chunks produced from {corpus_root}")
    return records


def convert_cases_to_source_spans(
    cases: list[dict[str, Any]],
    canonical_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert legacy chunk labels once into immutable source-interval labels."""
    by_chunk_id = {str(record["metadata"].get("chunk_id")): record for record in canonical_records}
    converted: list[dict[str, Any]] = []
    for original in cases:
        case = copy.deepcopy(original)
        raw_spans = case.get("relevance_spans")
        if raw_spans is not None:
            case["ground_truth_origin"] = "source_span"
            case["query_group_id"] = str(case.get("query_group_id", case.get("id")))
            converted.append(case)
            continue
        labels = Evaluator._parse_relevance(case) or {}
        spans: list[dict[str, Any]] = []
        for chunk_id, gain in labels.items():
            record = by_chunk_id.get(chunk_id)
            if record is None:
                raise ValueError(f"legacy label {chunk_id} is absent from canonical chunks")
            metadata = record["metadata"]
            spans.append(
                {
                    "document_id": metadata["file_name"],
                    "span_id": str(chunk_id),
                    "heading": metadata.get("heading_path", ""),
                    "level": str(metadata.get("heading_path", "")).count(" > ") + 1,
                    "start": int(metadata["source_char_start"]),
                    "end": int(metadata["source_char_end"]),
                    "gain": float(gain),
                }
            )
        case.pop("relevance", None)
        case.pop("expected_chunk_ids", None)
        case["relevance_spans"] = spans
        case["ground_truth_origin"] = "legacy_chunk_id_mapped_to_canonical_source_interval"
        case["query_group_id"] = str(case.get("query_group_id", case.get("id")))
        converted.append(case)
    return converted


def _candidate_payload(
    record: dict[str, Any],
    *,
    dense_score: float,
    dense_rank: int,
    sparse_score: float,
    sparse_rank: int,
) -> dict[str, Any]:
    return {
        **record,
        "dense_score": float(dense_score),
        "dense_rank": float(dense_rank),
        "sparse_score": float(sparse_score),
        "sparse_rank": float(sparse_rank),
        "qwen_score": float(dense_score),
        "qwen_rank": float(dense_rank),
        "bm25_score": float(sparse_score),
        "bm25_rank": float(sparse_rank),
        "stream_count": float((dense_score != 0.0) + (sparse_score != 0.0)),
    }


class Phase5Retriever:
    def __init__(self, index: Phase5Index, *, queries: list[str], router: Any | None = None):
        self.index = index
        self.queries = queries
        self.query_to_position = {query: index for index, query in enumerate(queries)}
        self.router = router

    def _mode(self, query: str, config: Phase5Config) -> str:
        if config.instruction_mode == "routed":
            return query_instruction_mode(query)
        if config.instruction_mode == "learned":
            if self.router is None:
                raise ValueError("learned instruction mode requires a fitted router")
            return str(self.router.predict([query])[0])
        return config.instruction_mode

    def score_candidates(self, query: str, config: Phase5Config) -> tuple[list[int], list[dict[str, Any]], np.ndarray]:
        mode = self._mode(query, config)
        query_position = self.query_to_position[query]
        dense_scores = self.index.query_vectors(self.queries, mode)[query_position] @ self.index.document_embeddings.T
        sparse_scores = self.index.bm25(config).scores(query)
        dense_order = np.argsort(-dense_scores, kind="stable")
        sparse_order = np.argsort(-sparse_scores, kind="stable")
        dense_ranks = {int(index): rank + 1 for rank, index in enumerate(dense_order)}
        sparse_ranks = {int(index): rank + 1 for rank, index in enumerate(sparse_order)}
        if config.fusion == "weighted_linear":
            fused = config.dense_weight * _minmax(dense_scores) + config.sparse_weight * _minmax(sparse_scores)
        else:
            fused = np.asarray(
                [
                    config.dense_weight / (config.rrf_k + dense_ranks[index])
                    + config.sparse_weight / (config.rrf_k + sparse_ranks[index])
                    for index in range(len(self.index.records))
                ],
                dtype=float,
            )
        order = np.argsort(-fused, kind="stable")[: config.candidate_depth]
        payloads = [
            _candidate_payload(
                self.index.records[int(index)],
                dense_score=float(dense_scores[index]),
                dense_rank=dense_ranks[int(index)],
                sparse_score=float(sparse_scores[index]),
                sparse_rank=sparse_ranks[int(index)],
            )
            for index in order
        ]
        feature_matrix = np.asarray(
            [LTRFeatureExtractor.matrix([LTRFeatureExtractor.extract(query, payload)])[0] for payload in payloads],
            dtype=float,
        )
        return [int(index) for index in order], payloads, feature_matrix

    def retrieve(
        self,
        query: str,
        config: Phase5Config,
        *,
        ltr_ranker: GroupedLTR | None = None,
    ) -> list[dict[str, Any]]:
        indexes, payloads, features = self.score_candidates(query, config)
        if config.ltr_on:
            if ltr_ranker is None:
                raise ValueError("LTR configuration requires a fitted ranker")
            predictions = ltr_ranker.predict(features)
            order = np.argsort(-predictions, kind="stable")[: config.top_k]
            return [self.index.records[indexes[int(position)]] for position in order]
        return [self.index.records[index] for index in indexes[: config.top_k]]


def _case_metrics(case: dict[str, Any], records: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    labels = Evaluator._parse_span_relevance(case) or []
    gains: list[float] = []
    relevant_ranks: list[int] = []
    matched_ids: set[str] = set()
    for rank, record in enumerate(records[:top_k], start=1):
        matching = [span for span in labels if span_matches_metadata(span, record["metadata"])]
        new_matching = [span for span in matching if str(span["span_id"]) not in matched_ids]
        gain = max((float(span["gain"]) for span in new_matching), default=0.0)
        gains.append(gain)
        if gain > 0:
            relevant_ranks.append(rank)
            matched_ids.update(str(span["span_id"]) for span in new_matching)
    relevant_ids = {str(span["span_id"]) for span in labels if float(span["gain"]) > 0}
    recall = len(matched_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0
    precision = len(relevant_ranks) / top_k
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    ideal = sorted((float(span["gain"]) for span in labels), reverse=True)[:top_k]
    idcg = Evaluator._dcg(ideal)
    ndcg = Evaluator._dcg(gains) / idcg if idcg else 0.0
    return {
        "case_id": str(case["id"]),
        "query_group_id": str(case.get("query_group_id", case["id"])),
        "category": str(case.get("category", "uncategorized")),
        "recall@5": recall,
        "precision@5": precision,
        "mrr": mrr,
        "ndcg": ndcg,
        "retrieved_chunk_ids": [str(record["metadata"].get("chunk_id", record["id"])) for record in records[:top_k]],
        "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    retriever: Phase5Retriever,
    config: Phase5Config,
    *,
    ltr_rankers_by_fold: dict[int, GroupedLTR] | None = None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for case in cases:
        started = perf_counter()
        ranker = None
        if ltr_rankers_by_fold:
            raise ValueError("fold-specific rankers require evaluate_folded_cases")
        records = retriever.retrieve(case["question"], config, ltr_ranker=ranker)
        detail = _case_metrics(case, records, config.top_k)
        detail["retrieval_latency_ms"] = (perf_counter() - started) * 1000
        details.append(detail)
    return details


def aggregate_details(details: list[dict[str, Any]], cases: list[dict[str, Any]], *, seed: int = 1729) -> dict[str, Any]:
    by_group = {str(case.get("query_group_id", case["id"])): case for case in cases}
    groups = [str(case.get("query_group_id", case["id"])) for case in cases]
    folds = grouped_query_folds(groups, n_splits=5, seed=seed)
    by_id = {str(detail["case_id"]): detail for detail in details}
    fold_rows: list[dict[str, Any]] = []
    metric_names = ("recall@5", "precision@5", "mrr", "ndcg")
    for fold_number, (_, validation_indexes) in enumerate(folds, start=1):
        validation_cases = [cases[index] for index in validation_indexes]
        rows = [by_id[str(case["id"])] for case in validation_cases]
        fold_rows.append(
            {
                "fold": fold_number,
                "validation_query_ids": [str(case["id"]) for case in validation_cases],
                "metrics": {metric: mean(float(row[metric]) for row in rows) for metric in metric_names},
            }
        )
    fold_metrics = [row["metrics"] for row in fold_rows]
    overall = {metric: mean(float(detail[metric]) for detail in details) for metric in metric_names}
    return {
        "n_splits": 5,
        "case_count": len(details),
        "overall": overall,
        "mean": {metric: mean(float(row[metric]) for row in fold_metrics) for metric in metric_names},
        "std": {metric: pstdev(float(row[metric]) for row in fold_metrics) for metric in metric_names},
        "folds": fold_rows,
        "latency_ms": {
            "mean": mean(float(detail["retrieval_latency_ms"]) for detail in details),
            "p95": float(np.percentile([float(detail["retrieval_latency_ms"]) for detail in details], 95)),
        },
        "category_metrics": {
            category: {
                metric: mean(float(row[metric]) for row in rows)
                for metric in metric_names
            }
            for category in sorted({str(case.get("category", "uncategorized")) for case in cases})
            for rows in [[by_id[str(case["id"])] for case in cases if str(case.get("category", "uncategorized")) == category]]
        },
        "group_count": len(by_group),
    }


def fit_lambdamart(
    cases: list[dict[str, Any]],
    retriever: Phase5Retriever,
    config: Phase5Config,
    train_indexes: list[int],
    *,
    random_state: int = 1729,
) -> GroupedLTR:
    """Fit real XGBoost LambdaMART on training-query candidate pools only."""
    ranker = GroupedLTR(model_name="xgboost-lambdamart", n_estimators=80, random_state=random_state)
    feature_rows: list[np.ndarray] = []
    labels: list[float] = []
    query_ids: list[str] = []
    for index in train_indexes:
        case = cases[index]
        candidate_indexes, payloads, features = retriever.score_candidates(case["question"], config)
        del candidate_indexes
        feature_rows.extend(features)
        case_labels = [
            max(
                (float(span["gain"]) for span in Evaluator._parse_span_relevance(case) or [] if span_matches_metadata(span, payload["metadata"])),
                default=0.0,
            )
            for payload in payloads
        ]
        labels.extend(case_labels)
        query_ids.extend([str(case.get("query_group_id", case["id"]))] * len(payloads))
    if not feature_rows or len(set(query_ids)) < 2 or len(set(labels)) < 2:
        raise ValueError("LambdaMART training data has insufficient grouped relevance variation")
    fitted = ranker.fit(np.asarray(feature_rows), np.asarray(labels), query_ids)
    if fitted.backend_name != "xgboost-lambdamart":
        raise RuntimeError(
            f"real XGBoost LambdaMART is unavailable; refusing backend {fitted.backend_name!r} as a substitute"
        )
    return fitted


def evaluate_folded_lambdamart(
    cases: list[dict[str, Any]],
    retriever: Phase5Retriever,
    config: Phase5Config,
    *,
    seed: int = 1729,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    groups = [str(case.get("query_group_id", case["id"])) for case in cases]
    folds = grouped_query_folds(groups, n_splits=5, seed=seed)
    details: list[dict[str, Any]] = []
    backend = "unknown"
    worker_path = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "phase5_ltr_worker.py"
    for fold_number, (train_indexes, validation_indexes) in enumerate(folds, start=1):
        train_feature_rows: list[np.ndarray] = []
        train_labels: list[float] = []
        train_query_ids: list[str] = []
        for index in train_indexes:
            case = cases[index]
            _, payloads, features = retriever.score_candidates(case["question"], config)
            train_feature_rows.extend(features)
            train_labels.extend(
                [
                    max(
                        (
                            float(span["gain"])
                            for span in Evaluator._parse_span_relevance(case) or []
                            if span_matches_metadata(span, payload["metadata"])
                        ),
                        default=0.0,
                    )
                    for payload in payloads
                ]
            )
            train_query_ids.extend([str(case.get("query_group_id", case["id"]))] * len(payloads))

        validation_payloads: list[tuple[int, list[dict[str, Any]], np.ndarray]] = []
        validation_feature_rows: list[np.ndarray] = []
        for index in validation_indexes:
            case = cases[index]
            candidate_indexes, payloads, features = retriever.score_candidates(case["question"], config)
            validation_payloads.append((index, [retriever.index.records[i] for i in candidate_indexes], features))
            validation_feature_rows.extend(features)

        with tempfile.TemporaryDirectory(prefix="phase5-ltr-") as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "fold.npz"
            output_path = temp_root / "predictions.npy"
            np.savez(
                input_path,
                train_features=np.asarray(train_feature_rows, dtype=float),
                train_labels=np.asarray(train_labels, dtype=float),
                train_query_ids=np.asarray(train_query_ids),
                validation_features=np.asarray(validation_feature_rows, dtype=float),
            )
            try:
                completed = subprocess.run(
                    [sys.executable, str(worker_path), str(input_path), str(output_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    cwd=str(worker_path.parents[1]),
                )
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or exc.stdout or "worker produced no diagnostic").strip()
                raise RuntimeError(f"LambdaMART worker failed: {detail[-2000:]}") from exc
            backend = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "unknown"
            predictions = np.load(output_path, allow_pickle=False)

        cursor = 0
        for index, candidate_records, _ in validation_payloads:
            case = cases[index]
            candidate_count = len(candidate_records)
            candidate_predictions = predictions[cursor : cursor + candidate_count]
            cursor += candidate_count
            order = np.argsort(-candidate_predictions, kind="stable")[: config.top_k]
            records = [candidate_records[int(position)] for position in order]
            started = perf_counter()
            detail = _case_metrics(case, records, config.top_k)
            detail["retrieval_latency_ms"] = (perf_counter() - started) * 1000
            detail["fold"] = fold_number
            details.append(detail)
    summary = aggregate_details(details, cases, seed=seed)
    return details, summary["folds"], backend


def bootstrap_mean_delta(
    left: list[float],
    right: list[float],
    *,
    seed: int = 1729,
    samples: int = 5000,
) -> dict[str, float | bool]:
    if len(left) != len(right) or not left:
        raise ValueError("bootstrap inputs must be non-empty and equal length")
    rng = np.random.default_rng(seed)
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        sample = rng.integers(0, len(left_values), len(left_values))
        deltas[index] = float(np.mean(right_values[sample] - left_values[sample]))
    lower, upper = np.quantile(deltas, [0.025, 0.975])
    observed = float(np.mean(right_values - left_values))
    return {
        "observed": observed,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "crosses_zero": bool(lower <= 0 <= upper),
        "samples": samples,
        "seed": seed,
    }


def config_with(config: Phase5Config, **updates: Any) -> Phase5Config:
    return replace(config, **updates)
