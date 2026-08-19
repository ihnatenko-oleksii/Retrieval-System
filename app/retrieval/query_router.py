"""Deterministic, query-only routing decisions for optional query variants."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QuerySignals:
    """Observable lexical signals extracted from one raw query."""

    token_count: int
    acronyms: tuple[str, ...] = ()
    quoted_terms: tuple[str, ...] = ()
    numeric_codes: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    precision_markers: tuple[str, ...] = ()
    ambiguity_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryDecision:
    """A deterministic routing decision made without benchmark metadata."""

    should_rewrite: bool
    should_expand: bool
    protected_signals: tuple[str, ...]
    reasons: tuple[str, ...]
    signals: QuerySignals


class QueryGate:
    """Choose whether a raw query is suitable for semantic variants.

    The gate intentionally uses only text features. Exact technical queries
    are protected because an LLM can remove identifiers or narrow terms that
    lexical retrieval needs. Natural-language or underspecified queries are
    allowed to use variants when their text contains no precision signal.
    """

    _precision_phrases = (
        "exact",
        "exactly",
        "immediately",
        "already issued",
        "automatically",
        "same operation",
        "same as",
        "difference between",
        "differ",
        "rather than",
        "instead of",
        "end date",
        "reset",
        "timeout",
        "error code",
        "status code",
        "quoted",
        "included",
        "distinguish",
    )
    _ambiguity_phrases = (
        "what happens",
        "how are",
        "how does",
        "what does",
        "something",
        "someone",
        "anything",
        "no longer usable",
        "remain available",
        "repeated",
        "fails",
        "failed",
        "after an operation",
        "across the platform",
    )
    _pronouns = {"someone", "something", "anything", "they", "it", "this", "that"}

    def signals(self, query: str) -> QuerySignals:
        text = query or ""
        tokens = re.findall(r"\b[\w-]+\b", text, flags=re.UNICODE)
        acronyms = tuple(re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", text))
        quoted_terms = tuple(match.group(1) for match in re.finditer(r"[`\"']([^`\"']+)[`\"']", text))
        numeric_codes = tuple(re.findall(r"\b[1-5]\d{2}\b", text))
        identifiers = tuple(
            re.findall(r"\b[A-Za-z][A-Za-z0-9]*(?:[_-][A-Za-z0-9]+)+\b", text)
        )
        lowered = text.lower()
        precision_markers = tuple(
            marker for marker in self._precision_phrases if marker in lowered
        )
        ambiguity_markers = tuple(
            marker for marker in self._ambiguity_phrases if marker in lowered
        )
        if any(token.lower() in self._pronouns for token in tokens):
            ambiguity_markers = (*ambiguity_markers, "pronoun")
        return QuerySignals(
            token_count=len(tokens),
            acronyms=acronyms,
            quoted_terms=quoted_terms,
            numeric_codes=numeric_codes,
            identifiers=identifiers,
            precision_markers=precision_markers,
            ambiguity_markers=tuple(dict.fromkeys(ambiguity_markers)),
        )

    def decide(self, query: str) -> QueryDecision:
        query_signals = self.signals(query)
        protected: list[str] = []
        if query_signals.acronyms:
            protected.append("acronym")
        if query_signals.quoted_terms:
            protected.append("quoted_term")
        if query_signals.numeric_codes:
            protected.append("numeric_code")
        if query_signals.identifiers:
            protected.append("identifier")
        protected.extend(query_signals.precision_markers)
        protected = list(dict.fromkeys(protected))

        question_like = bool(re.match(r"\s*(?:what|which|how|why|when|where|who|does|is|are)\b", query or "", re.I))
        semantic_question = question_like and query_signals.token_count >= 7
        ambiguous_question = bool(query_signals.ambiguity_markers)
        should_route = not protected and (ambiguous_question or semantic_question)
        reasons: list[str] = []
        if protected:
            reasons.append("precision-preserving query")
        if ambiguous_question:
            reasons.append("underspecified natural-language signal")
        elif semantic_question and not protected:
            reasons.append("long semantic question")
        if not reasons:
            reasons.append("short or non-question query")
        return QueryDecision(
            should_rewrite=should_route,
            should_expand=should_route,
            protected_signals=tuple(protected),
            reasons=tuple(reasons),
            signals=query_signals,
        )
