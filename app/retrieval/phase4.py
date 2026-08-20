"""Modular Phase 4 retrieval primitives.

The validated simple retriever remains the application path. This module
provides an experimentable, in-memory path for the historical Phase 4 work:
every dense model is a separate stream, sparse retrieval is field-aware BM25,
and all routing decisions are made from the query and observed stream signals.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.evals.span_relevance import span_matches_metadata
from app.retrieval.ltr import LTRFeatureExtractor

QWEN_INSTRUCTIONS: dict[str, str] = {
    "generic": (
        "Given a technical support or engineering question, retrieve passages that directly answer the question "
        "or contain the necessary implementation details."
    ),
    "semantic": (
        "Given a technical question, retrieve passages that explain the same underlying concept, behavior, or "
        "cause even when the wording differs."
    ),
    "precision": (
        "Given a technical question containing an identifier, code, number, boundary, or exact distinction, "
        "retrieve passages with the precise matching implementation detail."
    ),
    "ambiguity": (
        "Given an underspecified technical question, retrieve the passages covering its plausible interpretations "
        "and the distinctions needed to resolve them."
    ),
    "multiple": (
        "Given a multi-part technical question, retrieve every passage needed to answer the separate requested "
        "parts, preferring complementary evidence over repeated topical text."
    ),
}

MODEL_ALIASES = {
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "bge": "BAAI/bge-m3",
    "e5": "intfloat/multilingual-e5-base",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[_-][A-Za-z0-9]*)+|[A-Za-z0-9]+", re.UNICODE)
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
class Phase4Config:
    """Immutable knobs for one Phase 4 retrieval trial."""

    top_k: int = 5
    candidate_depth: int = 50
    stream_weights: tuple[tuple[str, float], ...] = (
        ("qwen", 0.3),
        ("bge", 0.25),
        ("e5", 0.2),
        ("bm25", 0.25),
    )
    fusion_strategy: str = "weighted_rrf"
    rrf_k: int = 60
    router_on: bool = False
    lexical_route_delta: float = 0.2
    disagreement_threshold: float = 0.5
    cascade_on: bool = False
    cascade_initial_depth: int = 20
    cascade_confidence_threshold: float = 0.42
    qwen_instruction_mode: str = "generic"
    qwen_instruction_routing: bool = False
    qwen_instruction_ensemble: bool = False
    context_aware: bool = False
    field_aware_bm25: bool = False
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_title_weight: float = 3.0
    bm25_heading_weight: float = 2.0
    bm25_body_weight: float = 1.0
    bm25_ngrams: bool = True
    hierarchical_on: bool = False
    hierarchy_weight: float = 0.08
    prf_on: bool = False
    prf_depth: int = 1
    prf_max_terms: int = 8
    prf_weight: float = 0.25
    prf_confidence_threshold: float = 0.35
    hyde_on: bool = False

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.candidate_depth < self.top_k:
            raise ValueError("Phase 4 requires candidate_depth >= top_k >= 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        if self.fusion_strategy not in {"weighted_linear", "rrf", "weighted_rrf"}:
            raise ValueError(f"unsupported Phase 4 fusion strategy: {self.fusion_strategy}")
        if self.qwen_instruction_mode not in {"none", *QWEN_INSTRUCTIONS, "routed", "ensemble"}:
            raise ValueError(f"unsupported Qwen instruction mode: {self.qwen_instruction_mode}")
        if any(weight < 0 for _, weight in self.stream_weights):
            raise ValueError("stream weights cannot be negative")
        if self.bm25_k1 <= 0 or not 0 <= self.bm25_b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")

    @property
    def weights(self) -> dict[str, float]:
        return {name: float(weight) for name, weight in self.stream_weights}


def _tokens(text: str, *, ngrams: bool = True) -> list[str]:
    raw = [match.group(0).casefold() for match in _TOKEN_RE.finditer(text or "")]
    values: list[str] = []
    for token in raw:
        if len(token) <= 1 or (token in _STOPWORDS and not any(char.isdigit() for char in token)):
            continue
        values.append(token)
        if "-" in token or "_" in token:
            values.extend(part for part in re.split(r"[-_]", token) if len(part) > 1)
    if ngrams:
        values.extend(f"{left} {right}" for left, right in zip(values, values[1:], strict=False))
    return values


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def _identity(metadata: dict[str, Any], content: str) -> tuple[Any, ...]:
    return (
        metadata.get("file_name") or metadata.get("file_path") or "",
        metadata.get("chunk_index", ""),
        content,
    )


def _query_signal_summary(query: str) -> dict[str, Any]:
    text = query or ""
    tokens = _tokens(text, ngrams=False)
    identifiers = tuple(
        token for token in _TOKEN_RE.findall(text) if "-" in token or "_" in token or token.isupper() or token.isdigit()
    )
    quoted = tuple(match.group(1) for match in re.finditer(r"[`\"']([^`\"']+)[`\"']", text))
    ambiguity_markers = tuple(
        marker
        for marker in ("what happens", "how does", "what does", "across", "difference", "which", "something")
        if marker in text.casefold()
    )
    return {
        "token_count": len(tokens),
        "identifier_count": len(identifiers),
        "quoted_count": len(quoted),
        "has_lexical_signal": bool(identifiers or quoted),
        "is_long_semantic": len(tokens) >= 10,
        "is_ambiguous": bool(ambiguity_markers),
        "identifiers": identifiers,
        "quoted_terms": quoted,
        "ambiguity_markers": ambiguity_markers,
    }


class FieldAwareBM25:
    """Independent field BM25 streams with configurable parameters."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
        title_weight: float = 3.0,
        heading_weight: float = 2.0,
        body_weight: float = 1.0,
        field_aware: bool = True,
        ngrams: bool = True,
    ):
        self.records = records
        self.k1 = k1
        self.b = b
        self.field_weights = {
            "title": title_weight if field_aware else 0.0,
            "heading": heading_weight if field_aware else 0.0,
            "body": body_weight,
        }
        if not field_aware:
            self.field_weights = {"body": 1.0}
        self.ngrams = ngrams
        self._indexes: dict[str, BM25Okapi] = {}
        self._cache: dict[tuple[str, int], list[tuple[int, float, dict[str, float]]]] = {}
        for field, weight in self.field_weights.items():
            if weight <= 0:
                continue
            corpus = [_tokens(self._field_text(record, field), ngrams=ngrams) for record in records]
            self._indexes[field] = BM25Okapi(corpus, k1=k1, b=b)

    @staticmethod
    def _field_text(record: dict[str, Any], field: str) -> str:
        metadata = record["metadata"]
        if field == "title":
            path = str(metadata.get("heading_path", ""))
            return path.split(" > ", 1)[0] if path else ""
        if field == "heading":
            return str(metadata.get("heading_path", ""))
        return str(record.get("content", ""))

    def query(self, query: str, n_results: int) -> list[tuple[int, float, dict[str, float]]]:
        cache_key = (query, n_results)
        if cache_key in self._cache:
            return self._cache[cache_key]
        query_tokens = _tokens(query, ngrams=self.ngrams)
        if not query_tokens:
            return []
        per_field: dict[str, np.ndarray] = {}
        for field, index in self._indexes.items():
            per_field[field] = np.asarray(index.get_scores(query_tokens), dtype=float)

        combined = np.zeros(len(self.records), dtype=float)
        field_details: list[dict[str, float]] = [dict() for _ in self.records]
        for field, scores in per_field.items():
            normalised = _normalise(scores.tolist())
            weight = self.field_weights[field]
            for index, (raw_score, normalised_score) in enumerate(zip(scores, normalised, strict=True)):
                field_details[index][f"bm25_{field}_score"] = float(raw_score)
                combined[index] += weight * normalised_score

        order = sorted(range(len(combined)), key=lambda index: (-float(combined[index]), index))[:n_results]
        results = [
            (index, float(combined[index]), field_details[index])
            for index in order
            if combined[index] > 0
        ]
        self._cache[cache_key] = results
        return results


class _DenseStream:
    def __init__(self, alias: str, model_name: str, records: list[dict[str, Any]], *, device: str, local_files_only: bool):
        from sentence_transformers import SentenceTransformer

        self.alias = alias
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        raw_texts = [str(record["content"]) for record in records]
        context_texts = [
            f"{record['metadata'].get('heading_path', '')}\n{record['content']}".strip()
            for record in records
        ]
        self.embeddings = {
            "raw": self._encode_documents(raw_texts),
            "context": self._encode_documents(context_texts),
        }

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        prepared = [f"passage: {text}" for text in texts] if self.alias == "e5" else texts
        vectors = self.model.encode(
            prepared,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=float)

    def _query_text(self, query: str, instruction_mode: str) -> str:
        if self.alias == "e5":
            return f"query: {query}"
        if self.alias != "qwen" or instruction_mode in {"none", ""}:
            return query
        instruction = QWEN_INSTRUCTIONS.get(instruction_mode, QWEN_INSTRUCTIONS["generic"])
        return f"Instruct: {instruction}\nQuery: {query}"

    def query(self, query: str, *, representation: str, instruction_mode: str) -> np.ndarray:
        vector = self.model.encode(
            [self._query_text(query, instruction_mode)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=float) @ self.embeddings[representation].T


class ChromaPhase4Backend:
    """Normalize the existing VectorStore API to a Phase 4 dense stream."""

    def __init__(self, vector_store: Any, *, query_instruction: str | None = None):
        self.vector_store = vector_store
        self.query_instruction = query_instruction
        self._cache: dict[tuple[str, int, str | None], list[tuple[str, dict[str, Any], float]]] = {}

    def query(self, query: str, *, n_results: int) -> list[tuple[str, dict[str, Any], float]]:
        cache_key = (query, n_results, self.query_instruction)
        if cache_key in self._cache:
            return self._cache[cache_key]
        if self.query_instruction is not None:
            embedding_function = getattr(self.vector_store, "embedding_function", None)
            if embedding_function is not None and hasattr(embedding_function, "query_instruction"):
                embedding_function.query_instruction = self.query_instruction
        payload = self.vector_store.query(query_text=query, n_results=n_results)
        if not payload or not payload.get("documents"):
            return []
        documents = payload["documents"][0]
        metadatas = payload.get("metadatas", [[]])[0] or [{}] * len(documents)
        distances = payload.get("distances", [[]])[0] or [0.0] * len(documents)
        results = [
            (content, metadata, float(distance))
            for content, metadata, distance in zip(documents, metadatas, distances, strict=True)
        ]
        self._cache[cache_key] = results
        return results


class Phase4Index:
    """A shared corpus index holding separate dense and sparse streams."""

    def __init__(
        self,
        chunks: list[Any],
        *,
        model_names: dict[str, str] | None = None,
        device: str = "cpu",
        local_files_only: bool = True,
        bm25_config: Phase4Config | None = None,
    ):
        self.chunks = tuple(chunks)
        self.records = [
            {"content": chunk.content, "metadata": chunk.metadata.model_dump()}
            for chunk in chunks
        ]
        self.record_lookup = {
            (
                metadata.get("file_name") or metadata.get("file_path") or "",
                metadata.get("chunk_index", ""),
            ): index
            for index, record in enumerate(self.records)
            for metadata in [record["metadata"]]
        }
        self.dense_backends: dict[str, dict[str, Any]] = {}
        self.streams: dict[str, _DenseStream] = {}
        self.model_errors: dict[str, str] = {}
        for alias, model_name in (model_names or MODEL_ALIASES).items():
            try:
                self.streams[alias] = _DenseStream(
                    alias,
                    model_name,
                    self.records,
                    device=device,
                    local_files_only=local_files_only,
                )
            except Exception as exc:  # Optional streams remain explicitly diagnosable.
                self.model_errors[alias] = f"{type(exc).__name__}: {exc}"
        bm25_config = bm25_config or Phase4Config()
        self.bm25 = FieldAwareBM25(
            self.records,
            k1=bm25_config.bm25_k1,
            b=bm25_config.bm25_b,
            title_weight=bm25_config.bm25_title_weight,
            heading_weight=bm25_config.bm25_heading_weight,
            body_weight=bm25_config.bm25_body_weight,
            field_aware=bm25_config.field_aware_bm25,
            ngrams=bm25_config.bm25_ngrams,
        )
        self._bm25_cache = {self._bm25_key(bm25_config): self.bm25}

    @staticmethod
    def _bm25_key(config: Phase4Config) -> tuple[Any, ...]:
        return (
            config.field_aware_bm25,
            config.bm25_k1,
            config.bm25_b,
            config.bm25_title_weight,
            config.bm25_heading_weight,
            config.bm25_body_weight,
            config.bm25_ngrams,
        )

    def bm25_for(self, config: Phase4Config) -> FieldAwareBM25:
        """Return the exact sparse variant requested by this trial."""
        key = self._bm25_key(config)
        if key not in self._bm25_cache:
            self._bm25_cache[key] = FieldAwareBM25(
                self.records,
                k1=config.bm25_k1,
                b=config.bm25_b,
                title_weight=config.bm25_title_weight,
                heading_weight=config.bm25_heading_weight,
                body_weight=config.bm25_body_weight,
                field_aware=config.field_aware_bm25,
                ngrams=config.bm25_ngrams,
            )
        return self._bm25_cache[key]

    @classmethod
    def from_vector_backends(
        cls,
        chunks: list[Any],
        *,
        dense_backends: dict[str, dict[str, Any]],
        bm25_config: Phase4Config | None = None,
    ) -> Phase4Index:
        """Build an index around precomputed Chroma streams without re-encoding documents."""
        index = cls.__new__(cls)
        index.chunks = tuple(chunks)
        index.records = [
            {"content": chunk.content, "metadata": chunk.metadata.model_dump()}
            for chunk in chunks
        ]
        index.record_lookup = {
            (
                metadata.get("file_name") or metadata.get("file_path") or "",
                metadata.get("chunk_index", ""),
            ): item_index
            for item_index, record in enumerate(index.records)
            for metadata in [record["metadata"]]
        }
        index.dense_backends = dense_backends
        index.streams = {}
        index.model_errors = {}
        bm25_config = bm25_config or Phase4Config()
        index.bm25 = FieldAwareBM25(
            index.records,
            k1=bm25_config.bm25_k1,
            b=bm25_config.bm25_b,
            title_weight=bm25_config.bm25_title_weight,
            heading_weight=bm25_config.bm25_heading_weight,
            body_weight=bm25_config.bm25_body_weight,
            field_aware=bm25_config.field_aware_bm25,
            ngrams=bm25_config.bm25_ngrams,
        )
        index._bm25_cache = {index._bm25_key(bm25_config): index.bm25}
        return index


class Phase4Retriever:
    """Query a Phase4Index with fusion, label-free routing, and traceability."""

    def __init__(self, index: Phase4Index, *, ranker: Any | None = None, hyde_provider: Any | None = None):
        self.index = index
        self.ranker = ranker
        self.hyde_provider = hyde_provider
        self.last_trace: dict[str, Any] = {}

    def _instruction_modes(self, query: str, config: Phase4Config) -> tuple[str, ...]:
        mode = config.qwen_instruction_mode
        if config.qwen_instruction_ensemble or mode == "ensemble":
            return ("generic", "semantic", "precision", "ambiguity", "multiple")
        if config.qwen_instruction_routing or mode == "routed":
            signals = _query_signal_summary(query)
            if signals["has_lexical_signal"]:
                return ("precision",)
            if signals["is_ambiguous"] or signals["token_count"] <= 6:
                return ("ambiguity",)
            if signals["token_count"] >= 14 or " and " in query.casefold():
                return ("multiple",)
            return ("semantic",)
        return (mode,)

    def _query_streams(self, query: str, config: Phase4Config, depth: int) -> dict[str, list[dict[str, Any]]]:
        streams: dict[str, list[dict[str, Any]]] = {}
        representation = "context" if config.context_aware else "raw"
        for alias, weight in config.weights.items():
            if weight <= 0:
                continue
            if alias == "bm25":
                bm25 = self.index.bm25_for(config) if hasattr(self.index, "bm25_for") else self.index.bm25
                sparse = bm25.query(query, depth)
                streams[alias] = [
                    {"index": index, "score": score, **field_scores}
                    for index, score, field_scores in sparse
                ]
                continue
            backend_options = getattr(self.index, "dense_backends", {}).get(alias)
            if backend_options:
                modes = self._instruction_modes(query, config) if alias == "qwen" else ("default",)
                by_index: dict[int, dict[str, Any]] = {}
                for mode in modes:
                    backend = backend_options.get(mode) or backend_options.get("default")
                    if backend is None:
                        continue
                    for content, metadata, distance in backend.query(query, n_results=depth):
                        key = (
                            metadata.get("file_name") or metadata.get("file_path") or "",
                            metadata.get("chunk_index", ""),
                        )
                        index = self.index.record_lookup.get(key)
                        if index is None:
                            index = next(
                                (
                                    item_index
                                    for item_index, record in enumerate(self.index.records)
                                    if record["content"] == content
                                ),
                                None,
                            )
                        if index is None:
                            continue
                        score = 1.0 - float(distance)
                        current = by_index.get(index)
                        if current is None or score > current["score"]:
                            by_index[index] = {
                                "index": index,
                                "score": score,
                                "instruction_modes": modes,
                            }
                streams[alias] = sorted(
                    by_index.values(), key=lambda item: (-float(item["score"]), int(item["index"]))
                )[:depth]
                continue
            stream = self.index.streams.get(alias)
            if stream is None:
                continue
            modes = self._instruction_modes(query, config) if alias == "qwen" else ("none",)
            scores = [stream.query(query, representation=representation, instruction_mode=mode) for mode in modes]
            combined_scores = np.max(np.vstack(scores), axis=0) if len(scores) > 1 else scores[0]
            order = np.argsort(-combined_scores, kind="stable")[:depth]
            streams[alias] = [
                {"index": int(index), "score": float(combined_scores[index]), "instruction_modes": modes}
                for index in order
            ]
        return streams

    @staticmethod
    def _stream_agreement(streams: dict[str, list[dict[str, Any]]]) -> tuple[float, float]:
        top_indexes = [values[0]["index"] for values in streams.values() if values]
        if not top_indexes:
            return 0.0, 0.0
        counts = Counter(top_indexes)
        agreement = max(counts.values()) / len(top_indexes)
        margins: list[float] = []
        for values in streams.values():
            if not values:
                continue
            top_score = float(values[0]["score"])
            second_score = float(values[1]["score"]) if len(values) > 1 else 0.0
            denominator = max(abs(top_score), 1e-8)
            margins.append(min(1.0, max(0.0, (top_score - second_score) / denominator)))
        return round(agreement, 4), round(sum(margins) / len(margins), 4) if margins else 0.0

    @staticmethod
    def _route_weights(
        query: str,
        config: Phase4Config,
        streams: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        weights = {name: weight for name, weight in config.weights.items() if name in streams and weight > 0}
        signals = _query_signal_summary(query)
        agreement, margin = Phase4Retriever._stream_agreement(streams)
        reasons: list[str] = []
        if not config.router_on:
            return weights, {
                "agreement": agreement,
                "margin": margin,
                "confidence": round(0.5 * agreement + 0.5 * margin, 4),
                "reasons": ("router disabled",),
                "escalated": False,
                "effective_fusion": config.fusion_strategy,
                "signals": signals,
            }
        if signals["has_lexical_signal"] and "bm25" in weights:
            weights["bm25"] += config.lexical_route_delta
            reasons.append("identifier or quoted lexical signal")
        if signals["is_long_semantic"]:
            for alias in ("qwen", "bge", "e5"):
                if alias in weights:
                    weights[alias] += 0.05
            reasons.append("long semantic query")
        confidence = round(0.5 * agreement + 0.5 * margin, 4)
        effective_fusion = config.fusion_strategy
        if agreement < config.disagreement_threshold:
            effective_fusion = "rrf"
            reasons.append("dense/sparse top-result disagreement")
        if not reasons:
            reasons.append("stable query and stream signals")
        return weights, {
            "agreement": agreement,
            "margin": margin,
            "confidence": confidence,
            "reasons": tuple(reasons),
            "escalated": False,
            "effective_fusion": effective_fusion,
            "signals": signals,
        }

    def _fuse_streams(
        self,
        streams: dict[str, list[dict[str, Any]]],
        weights: dict[str, float],
        *,
        config: Phase4Config,
        effective_fusion: str,
    ) -> tuple[list[tuple[int, float]], dict[int, dict[str, Any]]]:
        features: dict[int, dict[str, Any]] = {}
        for alias, values in streams.items():
            scores = _normalise([float(item["score"]) for item in values])
            for rank, (item, normalised_score) in enumerate(zip(values, scores, strict=True), start=1):
                index = int(item["index"])
                record = features.setdefault(
                    index,
                    {
                        "content": self.index.records[index]["content"],
                        "metadata": self.index.records[index]["metadata"],
                        "stream_count": 0,
                    },
                )
                record["stream_count"] = int(record["stream_count"]) + 1
                record[f"{alias}_score"] = float(item["score"])
                record[f"{alias}_normalized_score"] = float(normalised_score)
                record[f"{alias}_rank"] = rank
                if alias == "qwen":
                    record["qwen_instruction_modes"] = item.get("instruction_modes", ())
                for key, value in item.items():
                    if key.startswith("bm25_"):
                        record[key] = float(value)

        final_scores: dict[int, float] = {index: 0.0 for index in features}
        for alias, values in streams.items():
            weight = weights.get(alias, 0.0)
            if effective_fusion == "weighted_linear":
                for item in values:
                    record = features[int(item["index"])]
                    final_scores[int(item["index"])] += weight * float(record.get(f"{alias}_normalized_score", 0.0))
            else:
                effective_weight = weight if effective_fusion == "weighted_rrf" else 1.0
                for rank, item in enumerate(values, start=1):
                    final_scores[int(item["index"])] += effective_weight / (config.rrf_k + rank)

        if config.hierarchical_on and final_scores:
            sections: dict[tuple[str, str], float] = {}
            for index, score in final_scores.items():
                metadata = features[index]["metadata"]
                section = (str(metadata.get("file_name", "")), str(metadata.get("heading_path", "")))
                sections[section] = max(sections.get(section, float("-inf")), score)
            section_order = sorted(sections, key=lambda key: (-sections[key], key))
            section_bonus = {
                section: (len(section_order) - rank) / max(1, len(section_order))
                for rank, section in enumerate(section_order)
            }
            for index in final_scores:
                metadata = features[index]["metadata"]
                section = (str(metadata.get("file_name", "")), str(metadata.get("heading_path", "")))
                features[index]["hierarchy_score"] = section_bonus[section]
                final_scores[index] += config.hierarchy_weight * section_bonus[section]

        ranked = sorted(final_scores.items(), key=lambda item: (-item[1], item[0]))
        for rank, (index, score) in enumerate(ranked, start=1):
            features[index]["final_score"] = float(score)
            features[index]["candidate_rank"] = rank
            features[index]["rank_agreement"] = self._candidate_rank_agreement(features[index])
        return ranked, features

    @staticmethod
    def _candidate_rank_agreement(feature: dict[str, Any]) -> float:
        ranks = [float(feature[key]) for key in ("qwen_rank", "bge_rank", "e5_rank", "bm25_rank") if key in feature]
        return 1.0 / (1.0 + max(ranks) - min(ranks)) if ranks else 0.0

    def _retrieve_once(
        self,
        query: str,
        config: Phase4Config,
        *,
        allow_cascade: bool = True,
    ) -> tuple[list[tuple[int, float]], dict[int, dict[str, Any]], dict[str, Any]]:
        started = perf_counter()
        depth = config.candidate_depth
        initial_depth = min(depth, max(config.top_k, config.cascade_initial_depth)) if config.cascade_on else depth
        streams = self._query_streams(query, config, initial_depth)
        weights, routing = self._route_weights(query, config, streams)
        if (
            allow_cascade
            and config.cascade_on
            and routing["confidence"] < config.cascade_confidence_threshold
            and initial_depth < depth
        ):
            streams = self._query_streams(query, config, depth)
            weights, routing = self._route_weights(query, config, streams)
            routing["escalated"] = True
            routing["reasons"] = (*routing["reasons"], "low confidence after cheap first pass")
        results, features = self._fuse_streams(
            streams,
            weights,
            config=config,
            effective_fusion=routing["effective_fusion"],
        )
        routing["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        routing["depth"] = max((len(values) for values in streams.values()), default=0)
        return results, features, routing

    @staticmethod
    def _feedback_query(query: str, features: dict[int, dict[str, Any]], depth: int, max_terms: int) -> str:
        counts: Counter[str] = Counter()
        for index in sorted(features, key=lambda value: int(features[value].get("candidate_rank", 10**9)))[:depth]:
            for token in _tokens(str(features[index].get("content", "")), ngrams=False):
                counts[token] += 1
        terms = [token for token, _ in sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[:max_terms]]
        return f"{query} {' '.join(terms)}".strip()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
        config: Phase4Config | None = None,
    ) -> list[tuple[str, dict[str, Any], float]]:
        del chat_history
        config = config or Phase4Config(top_k=top_k or 5)
        if top_k is not None and top_k != config.top_k:
            config = replace(config, top_k=top_k)
        results, features, routing = self._retrieve_once(query, config)
        prf_applied = False
        prf_terms: tuple[str, ...] = ()
        hyde_status = "disabled"
        if config.prf_on and routing["confidence"] < config.prf_confidence_threshold and results:
            feedback_query = self._feedback_query(query, features, config.prf_depth, config.prf_max_terms)
            prf_terms = tuple(_tokens(feedback_query, ngrams=False)[len(_tokens(query, ngrams=False)) :])
            if prf_terms:
                feedback_results, feedback_features, _ = self._retrieve_once(
                    feedback_query,
                    replace(config, prf_on=False),
                    allow_cascade=False,
                )
                combined: dict[int, float] = {}
                for index, score in results:
                    combined[index] = combined.get(index, 0.0) + float(score)
                for index, score in feedback_results:
                    combined[index] = combined.get(index, 0.0) + config.prf_weight * float(score)
                    features.setdefault(index, feedback_features[index])
                    features[index]["prf_rank"] = next(
                        (rank for rank, (candidate, _) in enumerate(feedback_results, start=1) if candidate == index),
                        None,
                    )
                results = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
                prf_applied = True

        if config.hyde_on:
            if self.hyde_provider is None:
                hyde_status = "blocked: no local hypothetical-answer provider configured"
            else:
                hyde_status = "applied"
                hypothesis = str(self.hyde_provider(query))
                hyde_results, hyde_features, _ = self._retrieve_once(
                    hypothesis,
                    replace(config, hyde_on=False),
                    allow_cascade=False,
                )
                combined = {index: float(score) for index, score in results}
                for index, score in hyde_results:
                    combined[index] = combined.get(index, 0.0) + config.prf_weight * float(score)
                    features.setdefault(index, hyde_features[index])
                results = sorted(combined.items(), key=lambda item: (-item[1], item[0]))

        rerank_applied = False
        if self.ranker is not None:
            feature_records = [LTRFeatureExtractor.extract(query, features[index]) for index, _ in results]
            predictions = self.ranker.predict(LTRFeatureExtractor.matrix(feature_records))
            scored_results = list(zip(predictions, results, strict=True))
            results = [
                item
                for _, item in sorted(
                    scored_results, key=lambda value: (-float(value[0]), value[1][0])
                )
            ]
            rerank_applied = True
            ranker_scores = {item[1][0]: float(item[0]) for item in scored_results}
            for rank, (index, _) in enumerate(results, start=1):
                features[index]["ranker_score"] = ranker_scores[index]
                features[index]["candidate_rank"] = rank

        result_tuples = [
            (
                str(self.index.records[index]["content"]),
                dict(self.index.records[index]["metadata"]),
                float(score),
            )
            for index, score in results[: config.top_k]
        ]
        self.last_trace = {
            "candidate_features": [features[index] for index, _ in results],
            "routing": routing,
            "confidence_score": routing.get("confidence"),
            "confidence_triggered": bool(routing.get("escalated")),
            "escalated": bool(routing.get("escalated")),
            "escalation_rate": 1.0 if routing.get("escalated") else 0.0,
            "router_reasons": routing.get("reasons", ()),
            "instruction_modes": sorted(
                {
                    mode
                    for feature in features.values()
                    for mode in feature.get("qwen_instruction_modes", ())
                }
            ),
            "prf_applied": prf_applied,
            "prf_terms": prf_terms,
            "hyde_status": hyde_status,
            "ranker_applied": rerank_applied,
            "model_errors": dict(self.index.model_errors),
        }
        return result_tuples


def span_or_chunk_gain(case: dict[str, Any], metadata: dict[str, Any]) -> tuple[str | None, float]:
    """Return the relevance gain used for development ranking examples."""
    from app.evals.evaluator import Evaluator

    span_labels = Evaluator._parse_span_relevance(case)
    if span_labels is not None:
        matching = [label for label in span_labels if span_matches_metadata(label, metadata)]
        if matching:
            return str(matching[0]["span_id"]), max(float(label["gain"]) for label in matching)
        return None, 0.0
    labels = Evaluator._parse_relevance(case)
    if labels is not None:
        return Evaluator._match_label(labels, metadata)
    is_source_match = any(
        Evaluator._source_matches(source, metadata) for source in Evaluator._expected_sources(case)
    )
    return ("source", 1.0) if is_source_match else (None, 0.0)
