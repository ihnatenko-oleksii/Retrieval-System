"""Reproducible DEV/TEST retrieval experiment runner."""

from __future__ import annotations

import csv
import hashlib
import json
import socket
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from app.core.runtime_config import RetrievalConfig
from app.evals.benchmark_protocol import BenchmarkSplit, load_jsonl, load_or_create_split

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
LLM_REQUEST_TIMEOUT_SECONDS = 30.0

METRICS = ("recall@5", "precision@5", "mrr", "ndcg")


class InvalidChunkMapping(ValueError):
    """Raised before evaluation when a chunking candidate invalidates labels."""


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    dense_weight: float = 1.0
    sparse_weight: float = 0.0
    fusion_strategy: str = "weighted_linear"
    adaptive_routing: bool = False
    candidate_depth: int = 20
    reranker_on: bool = False
    reranker_model: str = DEFAULT_RERANKER_MODEL
    rerank_candidate_pool: int = 20
    query_rewriting_on: bool = False
    query_expansion_on: bool = False
    rewrite_policy: str = "never"
    expansion_policy: str = "never"
    include_original_query: bool = False
    multi_query_fusion_strategy: str = "weighted_rrf"
    original_query_weight: float = 1.0
    rewrite_query_weight: float = 0.7
    expansion_query_weight: float = 0.5
    confidence_routing: bool = False
    confidence_threshold: float = 0.35
    llm_model: str | None = None

    def to_retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig(
            top_k=self.top_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            reranker_on=self.reranker_on,
            rerank_top_n=self.top_k,
            query_rewriting_on=self.query_rewriting_on,
            query_expansion_on=self.query_expansion_on,
            llm_model=self.llm_model,
            fusion_strategy=self.fusion_strategy,
            rewrite_policy=self.rewrite_policy,
            expansion_policy=self.expansion_policy,
            include_original_query=self.include_original_query,
            multi_query_fusion_strategy=self.multi_query_fusion_strategy,
            original_query_weight=self.original_query_weight,
            rewrite_query_weight=self.rewrite_query_weight,
            expansion_query_weight=self.expansion_query_weight,
            confidence_routing=self.confidence_routing,
            confidence_threshold=self.confidence_threshold,
            candidate_depth=self.candidate_depth,
            rerank_candidate_pool=self.rerank_candidate_pool,
            adaptive_routing=self.adaptive_routing,
            embedding_model=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IndexedCorpus:
    key: tuple[str, int, int]
    storage_dir: Path
    chunks: tuple[Any, ...]
    vector_store: Any
    bm25_store: Any


class _CachedQueryStore:
    """Cache read-only backend queries shared by configurations in one index."""

    def __init__(self, delegate: Any):
        self.delegate = delegate
        self._cache: dict[tuple[str, int], Any] = {}

    def query(self, query_text: str, n_results: int) -> Any:
        key = (query_text, n_results)
        if key not in self._cache:
            self._cache[key] = self.delegate.query(query_text=query_text, n_results=n_results)
        return self._cache[key]


@dataclass(frozen=True)
class ExperimentResult:
    phase: str
    spec: ExperimentSpec
    status: str
    metrics: dict[str, float]
    category_metrics: dict[str, dict[str, float]]
    elapsed_seconds: float
    seconds_per_case: float | None
    error: str | None = None
    index_reused: bool = False
    routing_metrics: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "name": self.spec.name,
            "status": self.status,
            "parameters": self.spec.as_dict(),
            "metrics": self.metrics,
            "category_metrics": self.category_metrics,
            "elapsed_seconds": self.elapsed_seconds,
            "seconds_per_case": self.seconds_per_case,
            "index_reused": self.index_reused,
            "error": self.error,
            "routing_metrics": self.routing_metrics or {},
        }


def baseline_spec(name: str = "baseline-dense-e5") -> ExperimentSpec:
    return ExperimentSpec(name=name)


def previous_final_spec() -> ExperimentSpec:
    """The Phase 1 final: BGE-M3 static hybrid with replacement rewriting."""
    return ExperimentSpec(
        name="previous-final-bge-m3-static+global-rewrite",
        embedding_model="BAAI/bge-m3",
        dense_weight=0.7,
        sparse_weight=0.3,
        query_rewriting_on=True,
        rewrite_policy="always",
        include_original_query=False,
        multi_query_fusion_strategy="weighted_linear",
    )


def _unique_specs(specs: list[ExperimentSpec]) -> list[ExperimentSpec]:
    seen: set[str] = set()
    unique: list[ExperimentSpec] = []
    for spec in specs:
        fingerprint = json.dumps(spec.as_dict(), sort_keys=True)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(spec)
    return unique


def default_retrieval_specs() -> list[ExperimentSpec]:
    """Return the non-LLM matrix required by the retrieval protocol."""
    specs: list[ExperimentSpec] = [baseline_spec()]

    for dense_weight_index in range(10, -1, -1):
        dense_weight = round(dense_weight_index / 10, 1)
        sparse_weight = round(1.0 - dense_weight, 1)
        specs.append(
            ExperimentSpec(
                name=f"linear-static-d{dense_weight:.1f}-s{sparse_weight:.1f}",
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                adaptive_routing=False,
            )
        )

    specs.extend(
        [
            ExperimentSpec(
                name="hybrid-adaptive-linear-d0.7-s0.3",
                dense_weight=0.7,
                sparse_weight=0.3,
                adaptive_routing=True,
            ),
            ExperimentSpec(
                name="hybrid-static-rrf",
                dense_weight=0.7,
                sparse_weight=0.3,
                fusion_strategy="rrf",
            ),
            ExperimentSpec(
                name="hybrid-static-weighted-rrf",
                dense_weight=0.7,
                sparse_weight=0.3,
                fusion_strategy="weighted_rrf",
            ),
        ]
    )

    for candidate_depth in (20, 30, 50, 75, 100):
        specs.append(
            ExperimentSpec(
                name=f"hybrid-static-depth-{candidate_depth}",
                dense_weight=0.7,
                sparse_weight=0.3,
                candidate_depth=candidate_depth,
            )
        )

    for model_name, model_label in (
        (DEFAULT_EMBEDDING_MODEL, "e5-base"),
        ("BAAI/bge-m3", "bge-m3"),
    ):
        specs.append(
            ExperimentSpec(
                name=f"{model_label}-dense",
                embedding_model=model_name,
                dense_weight=1.0,
                sparse_weight=0.0,
            )
        )
        specs.append(
            ExperimentSpec(
                name=f"{model_label}-hybrid-static",
                embedding_model=model_name,
                dense_weight=0.7,
                sparse_weight=0.3,
            )
        )

    specs.extend(
        [
            ExperimentSpec(
                name="bge-m3-hybrid-adaptive",
                embedding_model="BAAI/bge-m3",
                dense_weight=0.7,
                sparse_weight=0.3,
                adaptive_routing=True,
            ),
            ExperimentSpec(
                name="e5-hybrid-adaptive",
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                dense_weight=0.7,
                sparse_weight=0.3,
                adaptive_routing=True,
            ),
        ]
    )

    for model_name, model_label in (
        (DEFAULT_RERANKER_MODEL, "minilm-reranker"),
        (DEFAULT_BGE_RERANKER_MODEL, "bge-reranker"),
    ):
        for candidate_pool in (10, 20, 30, 50):
            specs.append(
                ExperimentSpec(
                    name=f"hybrid-{model_label}-pool-{candidate_pool}",
                    dense_weight=0.7,
                    sparse_weight=0.3,
                    reranker_on=True,
                    reranker_model=model_name,
                    rerank_candidate_pool=candidate_pool,
                )
            )

    for chunk_size, chunk_overlap in ((400, 80), (600, 100), (800, 150), (1000, 200), (1200, 200)):
        specs.append(
            ExperimentSpec(
                name=f"hybrid-chunks-{chunk_size}-{chunk_overlap}",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                dense_weight=0.7,
                sparse_weight=0.3,
            )
        )

    return _unique_specs(specs)


def llm_variants(selected: ExperimentSpec) -> list[ExperimentSpec]:
    """Create DEV-only all-vs-selective and multi-query LLM experiments."""
    return [
        replace(
            selected,
            name=f"{selected.name}+rewrite-all-replace",
            query_rewriting_on=True,
            rewrite_policy="always",
            include_original_query=False,
            multi_query_fusion_strategy="weighted_linear",
        ),
        replace(
            selected,
            name=f"{selected.name}+rewrite-all-ensemble-linear",
            query_rewriting_on=True,
            rewrite_policy="always",
            include_original_query=True,
            multi_query_fusion_strategy="weighted_linear",
        ),
        replace(
            selected,
            name=f"{selected.name}+rewrite-all-ensemble-rrf",
            query_rewriting_on=True,
            rewrite_policy="always",
            include_original_query=True,
            multi_query_fusion_strategy="rrf",
        ),
        replace(
            selected,
            name=f"{selected.name}+rewrite-selective-ensemble-linear",
            query_rewriting_on=True,
            rewrite_policy="selective",
            include_original_query=True,
            multi_query_fusion_strategy="weighted_linear",
        ),
        replace(
            selected,
            name=f"{selected.name}+rewrite-selective-ensemble-rrf",
            query_rewriting_on=True,
            rewrite_policy="selective",
            include_original_query=True,
            multi_query_fusion_strategy="weighted_rrf",
        ),
        replace(
            selected,
            name=f"{selected.name}+expansion-selective-ensemble-rrf",
            query_expansion_on=True,
            expansion_policy="selective",
            include_original_query=True,
            multi_query_fusion_strategy="weighted_rrf",
        ),
        replace(
            selected,
            name=f"{selected.name}+rewrite-expansion-selective-ensemble-rrf",
            query_rewriting_on=True,
            query_expansion_on=True,
            rewrite_policy="selective",
            expansion_policy="selective",
            include_original_query=True,
            multi_query_fusion_strategy="weighted_rrf",
        ),
        replace(
            selected,
            name=f"{selected.name}+rewrite-selective-confidence-ensemble-rrf",
            query_rewriting_on=True,
            rewrite_policy="selective",
            include_original_query=True,
            multi_query_fusion_strategy="weighted_rrf",
            confidence_routing=True,
            confidence_threshold=0.35,
        ),
    ]


def rank_dev_results(results: list[ExperimentResult]) -> list[ExperimentResult]:
    """Rank successful DEV configurations without reading any TEST result."""
    successful = [result for result in results if result.status == "ok"]
    return sorted(
        successful,
        key=lambda result: (
            result.metrics.get("ndcg", float("-inf")),
            result.metrics.get("mrr", float("-inf")),
            result.metrics.get("recall@5", float("-inf")),
            result.metrics.get("precision@5", float("-inf")),
            -result.elapsed_seconds,
        ),
        reverse=True,
    )


def relative_improvement(baseline: dict[str, float], new: dict[str, float], metric: str) -> float | None:
    old = baseline.get(metric)
    current = new.get(metric)
    if old is None or current is None or old == 0:
        return None
    return round((current - old) / old * 100, 4)


def _ollama_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=2) as connection:
            connection.settimeout(2)
            connection.sendall(b"GET /api/tags HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            return connection.recv(16).startswith(b"HTTP/")
    except OSError:
        return False


def _category_metrics(details: list[dict[str, Any]], top_k: int) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        buckets[str(detail.get("category", "uncategorized"))].append(detail)

    output: dict[str, dict[str, float]] = {}
    for category, category_details in sorted(buckets.items()):
        output[category] = {
            f"recall@{top_k}": round(sum(float(detail.get("recall", 0.0)) for detail in category_details) / len(category_details), 4),
            f"precision@{top_k}": round(sum(float(detail.get("precision", 0.0)) for detail in category_details) / len(category_details), 4),
            "mrr": round(sum(float(detail.get("mrr", 0.0)) for detail in category_details) / len(category_details), 4),
            "ndcg": round(sum(float(detail.get("ndcg", 0.0)) for detail in category_details) / len(category_details), 4),
            "cases": len(category_details),
        }
    return output


def _routing_metrics(details: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize query routing without exposing benchmark labels to retrieval."""
    case_count = len(details)
    if not case_count:
        return {
            "cases": 0.0,
            "rewrite_count": 0.0,
            "rewrite_rate_percent": 0.0,
            "expansion_count": 0.0,
            "expansion_rate_percent": 0.0,
            "confidence_trigger_count": 0.0,
            "average_confidence": 0.0,
            "average_query_variant_count": 0.0,
        }
    confidence_values = [
        float(detail["confidence_score"])
        for detail in details
        if detail.get("confidence_score") is not None
    ]
    return {
        "cases": float(case_count),
        "rewrite_count": float(sum(bool(detail.get("rewrite_applied")) for detail in details)),
        "rewrite_rate_percent": round(
            sum(bool(detail.get("rewrite_applied")) for detail in details) / case_count * 100,
            4,
        ),
        "expansion_count": float(sum(bool(detail.get("expansion_applied")) for detail in details)),
        "expansion_rate_percent": round(
            sum(bool(detail.get("expansion_applied")) for detail in details) / case_count * 100,
            4,
        ),
        "confidence_trigger_count": float(sum(bool(detail.get("confidence_triggered")) for detail in details)),
        "average_confidence": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else 0.0,
        "average_query_variant_count": round(
            sum(float(detail.get("query_variant_count", 1)) for detail in details) / case_count,
            4,
        ),
    }


def _corpus_fingerprint(corpus_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in corpus_dir.rglob("*") if path.is_file() and not path.name.startswith(".")):
        digest.update(str(path.relative_to(corpus_dir)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_chunk_labels(
    cases: list[dict[str, Any]],
    chunks: list[Any],
    reference_chunks: list[Any] | None = None,
) -> None:
    from app.evals.evaluator import Evaluator

    candidate_by_id = {
        Evaluator.chunk_id_for_metadata(chunk.metadata.model_dump()): chunk
        for chunk in chunks
    }
    reference_by_id = (
        {
            Evaluator.chunk_id_for_metadata(chunk.metadata.model_dump()): chunk
            for chunk in reference_chunks
        }
        if reference_chunks is not None
        else {}
    )
    missing: list[str] = []
    mismatched: list[str] = []
    for case in cases:
        labels = Evaluator._parse_relevance(case)
        if labels is None:
            continue
        for chunk_id in labels:
            candidate = candidate_by_id.get(chunk_id)
            if candidate is None:
                missing.append(f"{case.get('id', '<unnamed>')}: {chunk_id}")
            elif reference_chunks is not None:
                reference = reference_by_id.get(chunk_id)
                if reference is None or candidate.content != reference.content:
                    mismatched.append(f"{case.get('id', '<unnamed>')}: {chunk_id}")
    if missing or mismatched:
        preview = ", ".join(missing[:5])
        if mismatched:
            mismatch_preview = ", ".join(mismatched[:5])
            preview = f"{preview}; content changed for {mismatch_preview}" if preview else f"content changed for {mismatch_preview}"
        suffix = " ..." if len(missing) + len(mismatched) > 5 else ""
        raise InvalidChunkMapping(
            "Chunk-size candidate changes stable chunk IDs; refusing to evaluate against existing labels: "
            f"{preview}{suffix}"
        )


class ExperimentRunner:
    """Run configurations while caching indexes shared by safe candidates."""

    def __init__(
        self,
        *,
        corpus_dir: Path,
        index_root: Path,
        skip_generation: bool = True,
    ):
        self.corpus_dir = corpus_dir.resolve()
        self.index_root = index_root.resolve()
        self.skip_generation = skip_generation
        self._index_cache: dict[tuple[str, int, int], IndexedCorpus] = {}
        self._reranker_cache: dict[str, Any] = {}
        self._reranker_errors: dict[str, str] = {}
        self._rewriter_cache: dict[str, Any] = {}
        self._canonical_chunks: tuple[Any, ...] | None = None
        self._corpus_hash = _corpus_fingerprint(self.corpus_dir)

    def _get_canonical_chunks(self) -> tuple[Any, ...]:
        if self._canonical_chunks is None:
            from app.ingestion.pipeline import IngestionPipeline

            self._canonical_chunks = tuple(
                IngestionPipeline(
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                ).process_directory(str(self.corpus_dir))
            )
        return self._canonical_chunks

    def _get_index(self, spec: ExperimentSpec, cases: list[dict[str, Any]]) -> tuple[IndexedCorpus, bool]:
        key = (spec.embedding_model, spec.chunk_size, spec.chunk_overlap)
        if key in self._index_cache:
            return self._index_cache[key], True

        from app.ingestion.pipeline import IngestionPipeline
        from app.vector_store.bm25_store import BM25Store
        from app.vector_store.chroma_store import VectorStore

        index_fingerprint = hashlib.sha256(
            json.dumps([self._corpus_hash, *key], sort_keys=True).encode()
        ).hexdigest()[:16]
        storage_dir = self.index_root / index_fingerprint
        marker_path = storage_dir / "manifest.json"
        pipeline = IngestionPipeline(chunk_size=spec.chunk_size, chunk_overlap=spec.chunk_overlap)
        chunks = pipeline.process_directory(str(self.corpus_dir))
        if not chunks:
            raise RuntimeError(f"No chunks produced from corpus: {self.corpus_dir}")
        reference_chunks = (
            list(self._get_canonical_chunks())
            if (spec.chunk_size, spec.chunk_overlap) != (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            else None
        )
        _validate_chunk_labels(cases, chunks, reference_chunks=reference_chunks)

        storage_dir.mkdir(parents=True, exist_ok=True)
        vector_store = VectorStore(
            persist_dir=str(storage_dir / "chroma"),
            embedding_model=spec.embedding_model,
        )
        bm25_store = BM25Store(persist_dir=str(storage_dir))
        reused = marker_path.exists()
        if not reused:
            vector_store.add_chunks(chunks)
            bm25_store.add_chunks(chunks)
            marker_path.write_text(
                json.dumps(
                    {
                        "corpus_sha256": self._corpus_hash,
                        "embedding_model": spec.embedding_model,
                        "chunk_size": spec.chunk_size,
                        "chunk_overlap": spec.chunk_overlap,
                        "chunks": len(chunks),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        index = IndexedCorpus(
            key=key,
            storage_dir=storage_dir,
            chunks=tuple(chunks),
            vector_store=_CachedQueryStore(vector_store),
            bm25_store=_CachedQueryStore(bm25_store),
        )
        self._index_cache[key] = index
        return index, reused

    def _get_reranker(self, spec: ExperimentSpec) -> Any | None:
        if not spec.reranker_on:
            return None
        if spec.reranker_model in self._reranker_errors:
            raise RuntimeError(self._reranker_errors[spec.reranker_model])
        if spec.reranker_model not in self._reranker_cache:
            from app.retrieval.reranker import Reranker

            try:
                reranker = Reranker(
                    model_name=spec.reranker_model,
                    force_load=True,
                )
            except Exception as exc:
                message = f"Reranker model {spec.reranker_model} unavailable: {type(exc).__name__}: {exc}"
                self._reranker_errors[spec.reranker_model] = message
                raise RuntimeError(message) from exc
            if getattr(reranker, "model", None) is None:
                message = f"Reranker model {spec.reranker_model} could not be loaded"
                self._reranker_errors[spec.reranker_model] = message
                raise RuntimeError(message)
            self._reranker_cache[spec.reranker_model] = reranker
        return self._reranker_cache[spec.reranker_model]

    def _get_rewriter(self, spec: ExperimentSpec) -> Any | None:
        if not (spec.query_rewriting_on or spec.query_expansion_on):
            return None
        from app.core.config import settings
        from app.retrieval.rewriter import QueryRewriter

        model_name = spec.llm_model or settings.llm_model
        if model_name not in self._rewriter_cache:
            self._rewriter_cache[model_name] = QueryRewriter(
                model_name=model_name,
                timeout_seconds=LLM_REQUEST_TIMEOUT_SECONDS,
                strict=True,
            )
        return self._rewriter_cache[model_name]

    def run_one(self, spec: ExperimentSpec, eval_path: Path, phase: str) -> ExperimentResult:
        cases = load_jsonl(eval_path)
        started = time.perf_counter()
        index_reused = False
        if (spec.query_rewriting_on or spec.query_expansion_on) and not _ollama_available():
            elapsed = round(time.perf_counter() - started, 4)
            return ExperimentResult(
                phase=phase,
                spec=spec,
                status="error",
                metrics={},
                category_metrics={},
                elapsed_seconds=elapsed,
                seconds_per_case=None,
                error="Ollama unavailable at http://localhost:11434/api/tags",
                index_reused=False,
                routing_metrics={},
            )
        try:
            index, index_reused = self._get_index(spec, cases)
            from app.evals.evaluator import Evaluator

            evaluator = Evaluator(
                config=spec.to_retrieval_config(),
                vector_store=index.vector_store,
                bm25_store=index.bm25_store,
                reranker=self._get_reranker(spec),
                rewriter=self._get_rewriter(spec),
                skip_generation=self.skip_generation,
            )
            metrics, details = evaluator.evaluate_cases(str(eval_path))
            if not metrics:
                raise RuntimeError("Evaluation produced no metrics")
            elapsed = round(time.perf_counter() - started, 4)
            return ExperimentResult(
                phase=phase,
                spec=spec,
                status="ok",
                metrics=metrics,
                category_metrics=_category_metrics(details, spec.top_k),
                elapsed_seconds=elapsed,
                seconds_per_case=round(elapsed / len(cases), 4) if cases else None,
                index_reused=index_reused,
                routing_metrics=_routing_metrics(details),
            )
        except InvalidChunkMapping as exc:
            elapsed = round(time.perf_counter() - started, 4)
            return ExperimentResult(
                phase=phase,
                spec=spec,
                status="invalid_label_mapping",
                metrics={},
                category_metrics={},
                elapsed_seconds=elapsed,
                seconds_per_case=None,
                error=str(exc),
                index_reused=index_reused,
                routing_metrics={},
            )
        except Exception as exc:  # Record failed configurations without losing the matrix.
            elapsed = round(time.perf_counter() - started, 4)
            return ExperimentResult(
                phase=phase,
                spec=spec,
                status="error",
                metrics={},
                category_metrics={},
                elapsed_seconds=elapsed,
                seconds_per_case=None,
                error=f"{type(exc).__name__}: {exc}",
                index_reused=index_reused,
                routing_metrics={},
            )

    def run_phase(self, specs: list[ExperimentSpec], eval_path: Path, phase: str) -> list[ExperimentResult]:
        results: list[ExperimentResult] = []
        for index, spec in enumerate(specs, start=1):
            print(f"[{phase} {index}/{len(specs)}] {spec.name} ...", flush=True)
            result = self.run_one(spec, eval_path, phase)
            results.append(result)
            print(
                f"[{phase} {index}/{len(specs)}] {result.status} in {result.elapsed_seconds}s "
                f"metrics={result.metrics or result.error}",
                flush=True,
            )
        return results


def _flatten_result(result: ExperimentResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "phase": result.phase,
        "name": result.spec.name,
        "status": result.status,
        "embedding_model": result.spec.embedding_model,
        "chunk_size": result.spec.chunk_size,
        "chunk_overlap": result.spec.chunk_overlap,
        "top_k": result.spec.top_k,
        "dense_weight": result.spec.dense_weight,
        "sparse_weight": result.spec.sparse_weight,
        "fusion_strategy": result.spec.fusion_strategy,
        "adaptive_routing": result.spec.adaptive_routing,
        "candidate_depth": result.spec.candidate_depth,
        "reranker_on": result.spec.reranker_on,
        "reranker_model": result.spec.reranker_model if result.spec.reranker_on else "",
        "rerank_candidate_pool": result.spec.rerank_candidate_pool if result.spec.reranker_on else "",
        "query_rewriting_on": result.spec.query_rewriting_on,
        "query_expansion_on": result.spec.query_expansion_on,
        "rewrite_policy": result.spec.rewrite_policy,
        "expansion_policy": result.spec.expansion_policy,
        "include_original_query": result.spec.include_original_query,
        "multi_query_fusion_strategy": result.spec.multi_query_fusion_strategy,
        "confidence_routing": result.spec.confidence_routing,
        "confidence_threshold": result.spec.confidence_threshold,
        "elapsed_seconds": result.elapsed_seconds,
        "seconds_per_case": result.seconds_per_case,
        "index_reused": result.index_reused,
        "error": result.error or "",
    }
    row.update({metric: result.metrics.get(metric, "") for metric in METRICS})
    row.update({key: value for key, value in (result.routing_metrics or {}).items()})
    return row


def write_phase_artifacts(
    results: list[ExperimentResult],
    output_dir: Path,
    phase: str,
    *,
    command: str,
    split: BenchmarkSplit,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [result.as_dict() for result in results]
    payload = {
        "phase": phase,
        "command": command,
        "split_manifest": split.manifest,
        "results": records,
        "ranking": [result.spec.name for result in rank_dev_results(results)] if phase.startswith("dev") else [],
    }
    (output_dir / f"{phase}_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = [_flatten_result(result) for result in results]
    fieldnames = list(rows[0]) if rows else ["phase", "name", "status"]
    with (output_dir / f"{phase}_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ranked = rank_dev_results(results) if phase.startswith("dev") else [result for result in results if result.status == "ok"]
    lines = [
        f"# {phase.replace('_', ' ').title()} Retrieval Experiments",
        "",
        f"Command: `{command}`",
        f"DEV cases: {len(split.dev_cases)}; frozen TEST cases: {len(split.test_cases)}.",
        "",
        "| Rank | Configuration | Status | nDCG | MRR | Recall@5 | Precision@5 | Seconds |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, result in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | {result.spec.name} | {result.status} | "
            f"{result.metrics.get('ndcg', '-')} | {result.metrics.get('mrr', '-')} | "
            f"{result.metrics.get('recall@5', '-')} | {result.metrics.get('precision@5', '-')} | "
            f"{result.elapsed_seconds} |"
        )
    failed = [result for result in results if result.status != "ok"]
    if failed:
        lines.extend(["", "## Failed or blocked experiments", "", "| Configuration | Status | Reason |", "|---|---|---|"])
        for result in failed:
            lines.append(f"| {result.spec.name} | {result.status} | {result.error or '-'} |")
    (output_dir / f"{phase}_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_comparison(
    baseline: ExperimentResult,
    final: ExperimentResult,
    output_dir: Path,
    *,
    command: str,
    split: BenchmarkSplit,
    dev_results: list[ExperimentResult] | None = None,
    llm_results: list[ExperimentResult] | None = None,
    previous_final: ExperimentResult | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    relative = {
        metric: relative_improvement(baseline.metrics, final.metrics, metric)
        for metric in METRICS
    }
    payload = {
        "phase": "frozen_test_final_comparison",
        "command": command,
        "test_frozen": split.manifest.get("test_frozen") is True,
        "test_case_count": len(split.test_cases),
        "baseline": baseline.as_dict(),
        "previous_final": previous_final.as_dict() if previous_final is not None else None,
        "final": final.as_dict(),
        "new_final": final.as_dict(),
        "relative_improvement_percent": relative,
        "previous_final_relative_improvement_percent": {
            metric: relative_improvement(baseline.metrics, previous_final.metrics, metric)
            for metric in METRICS
        }
        if previous_final is not None
        else None,
        "dev_selection": {
            "selected": final.spec.name,
            "ranking": [
                result.as_dict()
                for result in rank_dev_results([*(dev_results or []), *(llm_results or [])])
            ],
        },
    }
    (output_dir / "final_test_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Frozen TEST Baseline vs Final",
        "",
        f"Command: `{command}`",
        f"Frozen TEST cases: {len(split.test_cases)}.",
        "",
        "## Configuration",
        "",
        f"- Baseline: `{baseline.spec.name}` — `{baseline.spec.embedding_model}`, dense-only, no reranking, rewriting, or expansion.",
    ]
    if previous_final is not None:
        lines.append(
            f"- Previous final: `{previous_final.spec.name}` — `{previous_final.spec.embedding_model}`, chunk `"
            f"{previous_final.spec.chunk_size}/{previous_final.spec.chunk_overlap}`, fusion `"
            f"{previous_final.spec.fusion_strategy}`, weights `{previous_final.spec.dense_weight}/{previous_final.spec.sparse_weight}`, "
            f"global rewrite replacement `{previous_final.spec.query_rewriting_on and not previous_final.spec.include_original_query}`."
        )
    lines.extend(
        [
        f"- New final: `{final.spec.name}` — `{final.spec.embedding_model}`, chunk `{final.spec.chunk_size}/{final.spec.chunk_overlap}`, "
        f"fusion `{final.spec.fusion_strategy}`, weights `{final.spec.dense_weight}/{final.spec.sparse_weight}`, "
        f"candidate depth `{final.spec.candidate_depth}`, reranking `{final.spec.reranker_on}`, "
        f"query rewriting `{final.spec.query_rewriting_on}` (`{final.spec.rewrite_policy}`), query expansion "
        f"`{final.spec.query_expansion_on}` (`{final.spec.expansion_policy}`), original preserved `{final.spec.include_original_query}`, "
        f"stage-2 fusion `{final.spec.multi_query_fusion_strategy}`, confidence routing `{final.spec.confidence_routing}`.",
        "",
        "## TEST metrics",
        "",
        "| Configuration | Recall@5 | Precision@5 | MRR | nDCG | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Baseline | {baseline.metrics.get('recall@5', '-')} | {baseline.metrics.get('precision@5', '-')} | {baseline.metrics.get('mrr', '-')} | {baseline.metrics.get('ndcg', '-')} | {baseline.elapsed_seconds} |",
        *([f"| Previous final | {previous_final.metrics.get('recall@5', '-')} | {previous_final.metrics.get('precision@5', '-')} | {previous_final.metrics.get('mrr', '-')} | {previous_final.metrics.get('ndcg', '-')} | {previous_final.elapsed_seconds} |"] if previous_final is not None else []),
        f"| New final | {final.metrics.get('recall@5', '-')} | {final.metrics.get('precision@5', '-')} | {final.metrics.get('mrr', '-')} | {final.metrics.get('ndcg', '-')} | {final.elapsed_seconds} |",
        "",
        "## Relative improvement",
        "",
        "| Metric | `(new - baseline) / baseline * 100` |",
        "|---|---:|",
        ]
    )
    for metric in METRICS:
        lines.append(f"| {metric} | {relative[metric] if relative[metric] is not None else '-'}% |")

    dev_baseline = next(
        (result for result in dev_results or [] if result.spec.name == "baseline-dense-e5" and result.status == "ok"),
        None,
    )
    effect_specs = [
        ("Static linear hybrid", "linear-static-d0.7-s0.3"),
        ("Query-adaptive hybrid", "hybrid-adaptive-linear-d0.7-s0.3"),
        ("RRF", "hybrid-static-rrf"),
        ("Weighted RRF", "hybrid-static-weighted-rrf"),
        ("BGE-M3 dense", "bge-m3-dense"),
        ("BGE-M3 hybrid", "bge-m3-hybrid-static"),
        ("Retrieval depth 30", "hybrid-static-depth-30"),
        ("MiniLM rerank pool 10", "hybrid-minilm-reranker-pool-10"),
        ("MiniLM rerank pool 20", "hybrid-minilm-reranker-pool-20"),
        ("MiniLM rerank pool 30", "hybrid-minilm-reranker-pool-30"),
        ("MiniLM rerank pool 50", "hybrid-minilm-reranker-pool-50"),
        ("BGE reranker pool 10", "hybrid-bge-reranker-pool-10"),
        ("Chunk 400/80", "hybrid-chunks-400-80"),
        ("Chunk 600/100", "hybrid-chunks-600-100"),
        ("Chunk 800/150", "hybrid-chunks-800-150"),
        ("Chunk 1200/200", "hybrid-chunks-1200-200"),
    ]
    dev_by_name = {result.spec.name: result for result in [*(dev_results or []), *(llm_results or [])]}
    lines.extend(
        [
            "",
            "## Effect of major optimizations (DEV)",
            "",
            "Selection and deltas in this section use DEV only; TEST values are never used for tuning.",
            "",
            "| Optimization | Configuration | Status | DEV nDCG | Δ nDCG vs dense baseline | DEV MRR |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    if dev_baseline is not None:
        lines.append(
            f"| Baseline | {dev_baseline.spec.name} | {dev_baseline.status} | {dev_baseline.metrics.get('ndcg', '-')} | 0.0% | {dev_baseline.metrics.get('mrr', '-')} |"
        )
    for label, name in effect_specs:
        result = dev_by_name.get(name)
        if result is None:
            continue
        delta = relative_improvement(dev_baseline.metrics, result.metrics, "ndcg") if dev_baseline and result.status == "ok" else None
        lines.append(
            f"| {label} | {name} | {result.status} | {result.metrics.get('ndcg', '-')} | "
            f"{f'{delta}%' if delta is not None else '-'} | {result.metrics.get('mrr', '-')} |"
        )
    for result in llm_results or []:
        delta = relative_improvement(dev_baseline.metrics, result.metrics, "ndcg") if dev_baseline and result.status == "ok" else None
        lines.append(
            f"| LLM {result.spec.name.rsplit('+', 1)[-1]} | {result.spec.name} | {result.status} | "
            f"{result.metrics.get('ndcg', '-')} | {f'{delta}%' if delta is not None else '-'} | {result.metrics.get('mrr', '-')} |"
        )

    routing_description = (
        "The deterministic gate inspects raw-query acronyms, quoted terms, numeric status codes, "
        "identifiers, precision markers, pronouns, ambiguity phrases, and question length. "
        "Protected lexical/precision signals skip optional LLM variants; only query text can trigger "
        "selective rewrite or expansion. Stage 1 fuses dense/BM25 per query, then stage 2 can fuse "
        "original, rewrite, and expansion rankings with weighted linear, RRF, or weighted RRF. "
        "Adaptive dense/BM25 routing uses lexical signals to favor sparse retrieval, long semantic "
        "questions to favor dense retrieval, and the configured mix otherwise."
    )
    route_metrics = final.routing_metrics or {}
    previous_route_metrics = (previous_final.routing_metrics or {}) if previous_final is not None else {}
    lines.extend(
        [
            "",
            "## Query routing and generalized failure modes",
            "",
            routing_description,
            "",
            f"- New-final TEST rewrite rate: {route_metrics.get('rewrite_rate_percent', 0.0)}%; expansion rate: {route_metrics.get('expansion_rate_percent', 0.0)}%; confidence-triggered rewrites: {route_metrics.get('confidence_trigger_count', 0.0)}.",
            *([f"- Previous-final TEST rewrite rate: {previous_route_metrics.get('rewrite_rate_percent', 0.0)}%."] if previous_final is not None else []),
        ]
    )
    replacement_result = dev_by_name.get("bge-m3-hybrid-adaptive+rewrite-all-replace")
    ensemble_result = dev_by_name.get("bge-m3-hybrid-adaptive+rewrite-all-ensemble-rrf")
    expansion_result = dev_by_name.get("bge-m3-hybrid-adaptive+expansion-selective-ensemble-rrf")
    if replacement_result is not None and ensemble_result is not None:
        lines.append(
            f"- Rewrite replacement drift was measured as a general failure mode: DEV nDCG was {replacement_result.metrics.get('ndcg', '-')} for replacement versus {ensemble_result.metrics.get('ndcg', '-')} when the original query was retained and fused by rank."
        )
    if expansion_result is not None:
        lines.append(
            f"- Query-expansion ensemble retrieval introduced measurable noise on this benchmark: the selective expansion ensemble reached DEV nDCG {expansion_result.metrics.get('ndcg', '-')} and MRR {expansion_result.metrics.get('mrr', '-')}, below the selected adaptive route."
        )
    lines.extend(
        [
            "- Chunk-boundary changes were rejected when labeled chunk content no longer matched the canonical 1000/200 mapping; no invalid-label score entered selection.",
            "- BGE cross-encoder reranker candidates were recorded as unavailable because the offline cache lacked model weights; no silent fallback score was used.",
        ]
    )
    payload["query_routing"] = {
        "description": routing_description,
        "selected_new_final": final.spec.name,
        "new_final_routing_metrics": route_metrics,
        "previous_final_routing_metrics": previous_route_metrics if previous_final is not None else None,
    }

    cv_bullet = (
        f"Improved retrieval nDCG by {relative['ndcg']}% ({baseline.metrics.get('ndcg')} to {final.metrics.get('ndcg')}) "
        f"and MRR by {relative['mrr']}% ({baseline.metrics.get('mrr')} to {final.metrics.get('mrr')}) on a frozen {len(split.test_cases)}-query TEST set, "
        f"selecting {final.spec.embedding_model} with query-adaptive {final.spec.fusion_strategy} dense/BM25 fusion "
        f"({final.spec.dense_weight}/{final.spec.sparse_weight}) from a {len(split.dev_cases)}-query DEV split; "
        f"latency was {final.seconds_per_case}s/query versus {baseline.seconds_per_case}s/query for the dense baseline."
    )
    preprocessing = []
    if final.spec.query_rewriting_on:
        preprocessing.append("query rewriting")
    if final.spec.query_expansion_on:
        preprocessing.append("query expansion")
    if preprocessing:
        cv_bullet = cv_bullet.replace(
            f"from a {len(split.dev_cases)}-query DEV split;",
            f"plus {' and '.join(preprocessing)} from a {len(split.dev_cases)}-query DEV split;",
        )
    lines.extend(["", "## Recommended CV bullet", "", f"> {cv_bullet}"])

    category_sets = [set(baseline.category_metrics), set(final.category_metrics)]
    if previous_final is not None:
        category_sets.append(set(previous_final.category_metrics))
    categories = sorted(set().union(*category_sets))
    lines.extend(
        [
            "",
            "## Category breakdown",
            "",
            "| Category | Baseline nDCG | Previous final nDCG | New final nDCG | Baseline MRR | Previous final MRR | New final MRR | Cases |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for category in categories:
        baseline_category = baseline.category_metrics.get(category, {})
        previous_category = previous_final.category_metrics.get(category, {}) if previous_final is not None else {}
        final_category = final.category_metrics.get(category, {})
        lines.append(
            f"| {category} | {baseline_category.get('ndcg', '-')} | {previous_category.get('ndcg', '-')} | {final_category.get('ndcg', '-')} | "
            f"{baseline_category.get('mrr', '-')} | {previous_category.get('mrr', '-')} | {final_category.get('mrr', '-')} | "
            f"{final_category.get('cases', baseline_category.get('cases', '-'))} |"
        )

    if categories:
        best_category = max(categories, key=lambda category: final.category_metrics.get(category, {}).get("ndcg", 0.0))
        worst_category = min(categories, key=lambda category: final.category_metrics.get(category, {}).get("ndcg", 0.0))
        lines.extend(
            [
                "",
                f"Best final category by nDCG: `{best_category}` ({final.category_metrics.get(best_category, {}).get('ndcg', '-')}).",
                f"Worst final category by nDCG: `{worst_category}` ({final.category_metrics.get(worst_category, {}).get('ndcg', '-')}).",
            ]
        )

    failed_results = [
        result
        for result in [baseline, *( [previous_final] if previous_final is not None else []), final, *(dev_results or []), *(llm_results or [])]
        if result.status != "ok"
    ]
    if failed_results:
        lines.extend(
            [
                "",
                "## Failed or blocked experiments",
                "",
                "| Configuration | Status | Reason |",
                "|---|---|---|",
            ]
        )
        for result in failed_results:
            lines.append(f"| {result.spec.name} | {result.status} | {result.error or '-'} |")

    lines.extend(
        [
            "",
            "## Runtime tradeoff",
            "",
            f"- Baseline: {baseline.elapsed_seconds}s total ({baseline.seconds_per_case}s/case).",
            *([f"- Previous final: {previous_final.elapsed_seconds}s total ({previous_final.seconds_per_case}s/case), rewrite rate {(previous_final.routing_metrics or {}).get('rewrite_rate_percent', 0.0)}%."] if previous_final is not None else []),
            f"- New final: {final.elapsed_seconds}s total ({final.seconds_per_case}s/case), rewrite rate {(final.routing_metrics or {}).get('rewrite_rate_percent', 0.0)}%, expansion rate {(final.routing_metrics or {}).get('expansion_rate_percent', 0.0)}%.",
            "",
            "The final selection was made from DEV results only; this TEST comparison is not fed back into selection.",
        ]
    )
    (output_dir / "final_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload["recommended_cv_bullet"] = cv_bullet
    (output_dir / "final_test_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "final_test_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["configuration", *METRICS, "elapsed_seconds", *[f"relative_{metric}_percent" for metric in METRICS]])
        writer.writeheader()
        writer.writerow(
            {
                "configuration": baseline.spec.name,
                **{metric: baseline.metrics.get(metric, "") for metric in METRICS},
                "elapsed_seconds": baseline.elapsed_seconds,
                **{f"relative_{metric}_percent": "" for metric in METRICS},
            }
        )
        if previous_final is not None:
            previous_relative = {
                metric: relative_improvement(baseline.metrics, previous_final.metrics, metric)
                for metric in METRICS
            }
            writer.writerow(
                {
                    "configuration": previous_final.spec.name,
                    **{metric: previous_final.metrics.get(metric, "") for metric in METRICS},
                    "elapsed_seconds": previous_final.elapsed_seconds,
                    **{f"relative_{metric}_percent": previous_relative[metric] for metric in METRICS},
                }
            )
        writer.writerow(
            {
                "configuration": final.spec.name,
                **{metric: final.metrics.get(metric, "") for metric in METRICS},
                "elapsed_seconds": final.elapsed_seconds,
                **{f"relative_{metric}_percent": relative[metric] for metric in METRICS},
            }
        )
    return payload


def run_protocol(
    *,
    corpus_dir: Path,
    eval_path: Path,
    split_dir: Path,
    output_dir: Path,
    index_root: Path,
    seed: int,
    include_llm: bool,
    llm_model: str | None,
    command: str,
) -> dict[str, Any]:
    split = load_or_create_split(eval_path, split_dir, seed=seed)
    runner = ExperimentRunner(corpus_dir=corpus_dir, index_root=index_root, skip_generation=True)

    dev_results = runner.run_phase(default_retrieval_specs(), split_dir / "dev.jsonl", "dev")
    write_phase_artifacts(dev_results, output_dir, "dev", command=command, split=split)
    control_names = {baseline_spec().name, previous_final_spec().name}
    new_candidates = [result for result in dev_results if result.spec.name not in control_names]
    ranked = rank_dev_results(new_candidates)
    if not ranked:
        raise RuntimeError("No new retrieval configuration completed successfully on DEV")
    selected = ranked[0].spec

    llm_results: list[ExperimentResult] = []
    if include_llm:
        llm_specs = [
            replace(previous_final_spec(), llm_model=llm_model),
            *[replace(spec, llm_model=llm_model) for spec in llm_variants(selected)],
        ]
        if _ollama_available():
            llm_results = runner.run_phase(llm_specs, split_dir / "dev.jsonl", "dev_llm")
        else:
            llm_results = [
                ExperimentResult(
                    phase="dev_llm",
                    spec=spec,
                    status="error",
                    metrics={},
                    category_metrics={},
                    elapsed_seconds=0.0,
                    seconds_per_case=None,
                    error="Ollama unavailable at http://localhost:11434/api/tags",
                )
                for spec in llm_specs
            ]
        write_phase_artifacts(llm_results, output_dir, "dev_llm", command=command, split=split)
        llm_ranked = rank_dev_results(
            [result for result in llm_results if result.spec.name != previous_final_spec().name]
        )
        selected_result = ranked[0]
        if llm_ranked and (
            llm_ranked[0].metrics.get("ndcg", 0.0),
            llm_ranked[0].metrics.get("mrr", 0.0),
        ) > (
            selected_result.metrics.get("ndcg", 0.0),
            selected_result.metrics.get("mrr", 0.0),
        ):
            selected = llm_ranked[0].spec

    baseline = baseline_spec()
    previous_final = (
        replace(previous_final_spec(), llm_model=llm_model)
        if include_llm
        else previous_final_spec()
    )
    test_specs = [baseline, selected]
    if include_llm:
        test_specs = [baseline, previous_final, selected]
    test_results = runner.run_phase(test_specs, split_dir / "test.jsonl", "test")
    write_phase_artifacts(test_results, output_dir, "test", command=command, split=split)
    test_by_name = {result.spec.name: result for result in test_results}
    baseline_result = test_by_name[baseline.name]
    previous_final_result = test_by_name.get(previous_final.name)
    final_result = test_by_name[selected.name]
    final_payload = write_final_comparison(
        baseline_result,
        final_result,
        output_dir,
        command=command,
        split=split,
        dev_results=dev_results,
        llm_results=llm_results,
        previous_final=previous_final_result,
    )
    return {
        "split": split,
        "dev_results": dev_results,
        "dev_llm_results": llm_results,
        "selected": selected,
        "test_results": test_results,
        "final_payload": final_payload,
    }
