"""Small, query-grouped learning-to-rank utilities for DEV-only fusion."""

from __future__ import annotations

import importlib.util
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = (
    "dense_score",
    "dense_rank",
    "sparse_score",
    "sparse_rank",
    "late_interaction_score",
    "late_interaction_rank",
    "reranker_score",
    "reranker_rank",
    "stream_count",
    "rank_agreement",
    "exact_token_overlap",
    "query_token_count",
    "chunk_token_count",
    "identifier_signal",
    "qwen_score",
    "qwen_rank",
    "bge_score",
    "bge_rank",
    "e5_score",
    "e5_rank",
    "bm25_score",
    "bm25_rank",
    "title_lexical_overlap",
    "heading_lexical_overlap",
    "stream_presence",
    "section_depth",
    "source_char_length",
    "identifier_overlap",
)

_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b|\b[A-Za-z][A-Za-z0-9]*\b")
_STOPWORDS = {
    "a",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "which",
    "with",
}


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text or "") if token.casefold() not in _STOPWORDS]


def grouped_query_folds(
    query_ids: Sequence[str], *, n_splits: int = 5, seed: int = 1729
) -> list[tuple[list[int], list[int]]]:
    """Return deterministic folds whose validation groups are whole queries."""
    groups = sorted(set(str(query_id) for query_id in query_ids))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_splits > len(groups):
        raise ValueError("n_splits cannot exceed the number of unique queries")

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    assignments = {group: index % n_splits for index, group in enumerate(shuffled)}
    folds: list[tuple[list[int], list[int]]] = []
    for fold_index in range(n_splits):
        validation = [index for index, query_id in enumerate(query_ids) if assignments[str(query_id)] == fold_index]
        train = [index for index in range(len(query_ids)) if index not in set(validation)]
        folds.append((train, validation))
    return folds


class LTRFeatureExtractor:
    """Build features only from query text and retrieval evidence."""

    @staticmethod
    def extract(query: str, candidate: dict[str, Any]) -> dict[str, float]:
        query_tokens = set(_tokens(query))
        chunk_tokens = set(_tokens(str(candidate.get("content", ""))))
        overlap = len(query_tokens & chunk_tokens) / len(query_tokens) if query_tokens else 0.0
        metadata = candidate.get("metadata", {}) if isinstance(candidate.get("metadata", {}), dict) else {}
        title_tokens = set(_tokens(str(metadata.get("heading_path", "")).split(" > ", 1)[0]))
        heading_tokens = set(_tokens(str(metadata.get("heading_path", ""))))
        title_overlap = len(query_tokens & title_tokens) / len(query_tokens) if query_tokens else 0.0
        heading_overlap = len(query_tokens & heading_tokens) / len(query_tokens) if query_tokens else 0.0
        query_identifiers = {
            token.casefold()
            for token in _TOKEN_RE.findall(query)
            if "-" in token or "_" in token or token.isupper() or token.isdigit()
        }
        content_identifiers = {
            token.casefold()
            for token in _TOKEN_RE.findall(str(candidate.get("content", "")))
            if "-" in token or "_" in token or token.isupper() or token.isdigit()
        }
        identifier_overlap = (
            len(query_identifiers & content_identifiers) / len(query_identifiers) if query_identifiers else 0.0
        )
        identifier_signal = sum(
            1
            for token in _TOKEN_RE.findall(str(candidate.get("content", "")))
            if "-" in token or "_" in token or token.isupper()
        )
        ranks = [
            float(candidate[key])
            for key in ("dense_rank", "sparse_rank", "late_interaction_rank", "reranker_rank")
            if candidate.get(key) is not None
        ]
        rank_agreement = 1.0 / (1.0 + (max(ranks) - min(ranks))) if ranks else 0.0
        return {
            "dense_score": float(candidate.get("dense_score", 0.0) or 0.0),
            "dense_rank": float(candidate.get("dense_rank", 0.0) or 0.0),
            "sparse_score": float(candidate.get("sparse_score", 0.0) or 0.0),
            "sparse_rank": float(candidate.get("sparse_rank", 0.0) or 0.0),
            "late_interaction_score": float(candidate.get("late_interaction_score", 0.0) or 0.0),
            "late_interaction_rank": float(candidate.get("late_interaction_rank", 0.0) or 0.0),
            "reranker_score": float(candidate.get("reranker_score", 0.0) or 0.0),
            "reranker_rank": float(candidate.get("reranker_rank", 0.0) or 0.0),
            "stream_count": float(candidate.get("stream_count", 0.0) or 0.0),
            "rank_agreement": rank_agreement,
            "exact_token_overlap": overlap,
            "query_token_count": float(len(query_tokens)),
            "chunk_token_count": float(len(chunk_tokens)),
            "identifier_signal": float(identifier_signal),
            "qwen_score": float(candidate.get("qwen_score", 0.0) or 0.0),
            "qwen_rank": float(candidate.get("qwen_rank", 0.0) or 0.0),
            "bge_score": float(candidate.get("bge_score", 0.0) or 0.0),
            "bge_rank": float(candidate.get("bge_rank", 0.0) or 0.0),
            "e5_score": float(candidate.get("e5_score", 0.0) or 0.0),
            "e5_rank": float(candidate.get("e5_rank", 0.0) or 0.0),
            "bm25_score": float(candidate.get("bm25_score", 0.0) or 0.0),
            "bm25_rank": float(candidate.get("bm25_rank", 0.0) or 0.0),
            "title_lexical_overlap": title_overlap,
            "heading_lexical_overlap": heading_overlap,
            "stream_presence": float(candidate.get("stream_count", 0.0) or 0.0),
            "section_depth": float(len(str(metadata.get("heading_path", "")).split(" > ")) if metadata.get("heading_path") else 0),
            "source_char_length": float(len(str(candidate.get("content", "")))),
            "identifier_overlap": identifier_overlap,
        }

    @staticmethod
    def matrix(records: Sequence[dict[str, float]]) -> np.ndarray:
        return np.asarray([[float(record.get(name, 0.0)) for name in FEATURE_NAMES] for record in records], dtype=float)


@dataclass(frozen=True)
class GroupedCVSummary:
    model_name: str
    folds: tuple[dict[str, Any], ...]


class _PairwiseLinearModel:
    def __init__(self, random_state: int):
        self.random_state = random_state
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, random_state=random_state),
        )

    def fit(self, features: np.ndarray, labels: np.ndarray, query_ids: Sequence[str]) -> None:
        differences: list[np.ndarray] = []
        targets: list[int] = []
        for query_id in sorted(set(str(value) for value in query_ids)):
            indexes = [index for index, value in enumerate(query_ids) if str(value) == query_id]
            for left_position, left_index in enumerate(indexes):
                for right_index in indexes[left_position + 1 :]:
                    if labels[left_index] == labels[right_index]:
                        continue
                    left_wins = labels[left_index] > labels[right_index]
                    difference = features[left_index] - features[right_index]
                    differences.extend([difference, -difference])
                    targets.extend([int(left_wins), int(not left_wins)])
        if not differences:
            raise ValueError("pairwise LTR needs at least one unequal within-query label pair")
        self.model.fit(np.asarray(differences), np.asarray(targets))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.decision_function(features), dtype=float)


class GroupedLTR:
    """Use XGBoost LambdaMART when installed, otherwise a recorded sklearn fallback."""

    def __init__(self, *, model_name: str = "auto", n_estimators: int = 80, random_state: int = 1729):
        self.requested_model_name = model_name
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Any | None = None
        self.backend_name = "unfitted"

    def _make_model(self) -> Any:
        if self.requested_model_name == "pairwise-linear":
            self.backend_name = "pairwise-linear"
            return _PairwiseLinearModel(self.random_state)
        wants_xgboost = self.requested_model_name in {"auto", "xgboost", "xgboost-lambdamart"}
        if wants_xgboost and importlib.util.find_spec("xgboost") is not None:  # pragma: no cover - optional dependency
            from xgboost import XGBRanker

            self.backend_name = "xgboost-lambdamart"
            return XGBRanker(
                objective="rank:ndcg",
                eval_metric="ndcg@5",
                n_estimators=self.n_estimators,
                max_depth=4,
                learning_rate=0.08,
                random_state=self.random_state,
                n_jobs=1,
                tree_method="hist",
            )
        self.backend_name = "sklearn-random-forest-regression-fallback"
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=8,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=1,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray, query_ids: Sequence[str]) -> GroupedLTR:
        if len(features) != len(labels) or len(labels) != len(query_ids):
            raise ValueError("features, labels, and query_ids must have equal length")
        if len(set(str(query_id) for query_id in query_ids)) < 2:
            raise ValueError("learning-to-rank requires at least two query groups")
        self.model = self._make_model()
        if self.backend_name == "xgboost-lambdamart":  # pragma: no cover - optional dependency
            order = sorted(range(len(query_ids)), key=lambda index: str(query_ids[index]))
            ordered_features = features[order]
            ordered_labels = labels[order]
            ordered_groups = [str(query_ids[index]) for index in order]
            group_sizes: list[int] = []
            for query_id in dict.fromkeys(ordered_groups):
                group_sizes.append(ordered_groups.count(query_id))
            self.model.fit(ordered_features, ordered_labels, group=group_sizes)
        elif self.backend_name == "pairwise-linear":
            self.model.fit(features, labels, query_ids)
        else:
            self.model.fit(features, labels)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GroupedLTR must be fitted before predict")
        return np.asarray(self.model.predict(features), dtype=float)
