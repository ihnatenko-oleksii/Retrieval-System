"""Controlled pseudo-relevance feedback for low-confidence retrieval queries."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-_][A-Za-z0-9]+)+\b|\b[A-Za-z][A-Za-z0-9]{2,}\b")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "into",
    "jest",
    "jak",
    "jaki",
    "który",
    "na",
    "oraz",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "work",
}


@dataclass(frozen=True)
class FeedbackQuery:
    """Original query plus a bounded, inspectable feedback representation."""

    original_query: str
    query: str
    terms: tuple[str, ...]
    depth: int


class PseudoRelevanceFeedback:
    """Extract useful lexical signals without copying whole passages."""

    def __init__(self, *, max_terms: int = 8):
        if max_terms < 1:
            raise ValueError("max_terms must be at least 1")
        self.max_terms = max_terms

    @staticmethod
    def _key(token: str) -> str:
        return token.casefold()

    def extract_terms(self, passages: list[str], *, depth: int = 1) -> tuple[str, ...]:
        if depth < 1:
            raise ValueError("depth must be at least 1")

        counts: Counter[str] = Counter()
        display: dict[str, str] = {}
        for passage in passages[:depth]:
            for token in _TOKEN_RE.findall(passage or ""):
                key = self._key(token)
                if key in _STOPWORDS or len(key) < 3:
                    continue
                # Identifiers/acronyms are intentionally stronger than common
                # words because they are useful ambiguity disambiguators.
                weight = 3 if ("-" in token or "_" in token or token.isupper()) else 1
                counts[key] += weight
                display.setdefault(key, token)

        ranked = sorted(counts, key=lambda key: (-counts[key], -len(key), key))
        return tuple(display[key] for key in ranked[: self.max_terms])

    def build_query(self, original_query: str, passages: list[str], *, depth: int = 1) -> FeedbackQuery:
        terms = self.extract_terms(passages, depth=depth)
        suffix = " ".join(terms)
        query = f"{original_query} {suffix}".strip() if suffix else original_query
        return FeedbackQuery(
            original_query=original_query,
            query=query,
            terms=terms,
            depth=depth,
        )

    @staticmethod
    def should_apply(
        query: str,
        *,
        confidence: float,
        threshold: float,
        has_results: bool,
        protected_signals: tuple[str, ...] = (),
    ) -> bool:
        """Gate extra retrieval work using deterministic, label-free signals."""
        if not has_results or confidence >= threshold or protected_signals:
            return False
        normalized = (query or "").casefold()
        ambiguous_markers = ("what", "how", "which", "why", "it", "this", "that")
        return len(normalized.split()) <= 8 or any(marker in normalized.split() for marker in ambiguous_markers)

