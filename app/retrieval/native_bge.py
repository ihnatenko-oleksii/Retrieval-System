"""Optional native BGE-M3 dense, learned-sparse, and late-interaction scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class NativeBackendUnavailable(RuntimeError):
    """Raised when the optional native BGE-M3 runtime cannot be used."""


@dataclass(frozen=True)
class NativeCandidateScore:
    content: str
    metadata: dict[str, Any]
    dense_score: float
    sparse_score: float
    late_interaction_score: float
    dense_rank: int | None
    sparse_rank: int | None
    late_interaction_rank: int | None
    final_score: float
    candidate_rank: int


@dataclass(frozen=True)
class NativeSearchResult:
    results: list[tuple[str, dict[str, Any], float]]
    candidate_scores: tuple[NativeCandidateScore, ...]
    trace: tuple[dict[str, Any], ...]


def _dot(left: Any, right: Any) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right, strict=True)))


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def _rank(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    ranks = [0] * len(values)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    return ranks


class NativeBGEBackend:
    """Score a small candidate pool with the BGE-M3 native API.

    The backend intentionally accepts a candidate pool instead of requiring a
    production multi-vector ANN index. This keeps the experiment modular while
    making late interaction measurable on the current small corpus.
    """

    def __init__(self, model: Any | None = None, *, unavailable_reason: str | None = None):
        self.model = model
        self.unavailable_reason = unavailable_reason or (
            "FlagEmbedding BGEM3FlagModel is not loaded"
            if model is None
            else None
        )

    @classmethod
    def from_pretrained(cls, model_name: str = "BAAI/bge-m3", *, use_fp16: bool = False) -> NativeBGEBackend:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as exc:  # pragma: no cover - depends on optional runtime
            return cls(unavailable_reason=f"FlagEmbedding is unavailable: {type(exc).__name__}: {exc}")

        try:  # pragma: no cover - depends on optional model/runtime
            model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        except Exception as exc:
            return cls(unavailable_reason=f"BGE-M3 native model unavailable: {type(exc).__name__}: {exc}")
        return cls(model=model)

    @property
    def available(self) -> bool:
        return self.model is not None

    def _encode(self, texts: list[str]) -> dict[str, Any]:
        if self.model is None:
            raise NativeBackendUnavailable(self.unavailable_reason or "native BGE-M3 backend unavailable")
        try:
            return self.model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
            )
        except Exception as exc:
            raise NativeBackendUnavailable(
                f"BGE-M3 native encode failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _sparse_score(self, query: Any, document: Any) -> float:
        scorer = getattr(self.model, "compute_lexical_matching_score", None)
        if not callable(scorer):
            raise NativeBackendUnavailable("BGE-M3 native model lacks learned-sparse scoring")
        return float(scorer(query, document))

    def _late_score(self, query: Any, document: Any) -> float:
        scorer = getattr(self.model, "colbert_score", None) or getattr(self.model, "compute_colbert_score", None)
        if not callable(scorer):
            raise NativeBackendUnavailable("BGE-M3 native model lacks ColBERT late-interaction scoring")
        return float(scorer(query, document))

    def search(
        self,
        query: str,
        candidates: list[tuple[str, dict[str, Any]]],
        *,
        top_n: int,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.3,
        colbert_weight: float = 0.3,
    ) -> NativeSearchResult:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not candidates:
            return NativeSearchResult(results=[], candidate_scores=(), trace=())
        if any(weight < 0 for weight in (dense_weight, sparse_weight, colbert_weight)):
            raise ValueError("native BGE weights cannot be negative")
        if dense_weight + sparse_weight + colbert_weight <= 0:
            raise ValueError("at least one native BGE weight must be positive")

        encoded = self._encode([query, *[content for content, _ in candidates]])
        dense_vectors = encoded.get("dense_vecs")
        sparse_vectors = encoded.get("lexical_weights")
        colbert_vectors = encoded.get("colbert_vecs")
        if dense_vectors is None or sparse_vectors is None or colbert_vectors is None:
            raise NativeBackendUnavailable("BGE-M3 native output omitted dense, sparse, or ColBERT vectors")

        dense_scores = [_dot(dense_vectors[0], vector) for vector in dense_vectors[1:]]
        sparse_scores = [self._sparse_score(sparse_vectors[0], vector) for vector in sparse_vectors[1:]]
        late_scores = [self._late_score(colbert_vectors[0], vector) for vector in colbert_vectors[1:]]
        dense_ranks = _rank(dense_scores)
        sparse_ranks = _rank(sparse_scores)
        late_ranks = _rank(late_scores)
        dense_normalized = _normalize(dense_scores)
        sparse_normalized = _normalize(sparse_scores)
        late_normalized = _normalize(late_scores)

        scored: list[NativeCandidateScore] = []
        for index, (content, metadata) in enumerate(candidates):
            final_score = (
                dense_weight * dense_normalized[index]
                + sparse_weight * sparse_normalized[index]
                + colbert_weight * late_normalized[index]
            )
            scored.append(
                NativeCandidateScore(
                    content=content,
                    metadata=metadata,
                    dense_score=dense_scores[index],
                    sparse_score=sparse_scores[index],
                    late_interaction_score=late_scores[index],
                    dense_rank=dense_ranks[index],
                    sparse_rank=sparse_ranks[index],
                    late_interaction_rank=late_ranks[index],
                    final_score=final_score,
                    candidate_rank=0,
                )
            )

        ordered = sorted(scored, key=lambda item: (-item.final_score, item.dense_rank or 0))
        ordered = [
            NativeCandidateScore(**{**item.__dict__, "candidate_rank": rank})
            for rank, item in enumerate(ordered, start=1)
        ]
        selected = ordered[:top_n]
        return NativeSearchResult(
            results=[(item.content, item.metadata, item.final_score) for item in selected],
            candidate_scores=tuple(selected),
            trace=tuple(
                {
                    "candidate_rank": item.candidate_rank,
                    "dense_score": item.dense_score,
                    "dense_rank": item.dense_rank,
                    "sparse_score": item.sparse_score,
                    "sparse_rank": item.sparse_rank,
                    "late_interaction_score": item.late_interaction_score,
                    "late_interaction_rank": item.late_interaction_rank,
                    "final_score": item.final_score,
                }
                for item in selected
            ),
        )

