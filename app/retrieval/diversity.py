"""Optional diversity-aware final selection for multi-relevant queries."""

from __future__ import annotations

import re


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in re.findall(r"\b\w+\b", text or "") if len(token) > 2}


def _similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def mmr_select(
    results: list[tuple[str, dict, float]],
    *,
    top_k: int,
    relevance_weight: float = 0.7,
) -> list[tuple[str, dict, float]]:
    """Greedily select relevant and complementary chunks, keeping rank one."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between 0 and 1")
    if len(results) <= top_k:
        return list(results)

    scores = [float(result[2]) for result in results]
    low, high = min(scores), max(scores)
    relevance = [1.0 if high == low else (score - low) / (high - low) for score in scores]
    selected_indexes = [0]
    remaining = set(range(1, len(results)))
    while remaining and len(selected_indexes) < top_k:
        best_index = max(
            remaining,
            key=lambda index: (
                relevance_weight * relevance[index]
                - (1.0 - relevance_weight)
                * max(_similarity(results[index][0], results[selected][0]) for selected in selected_indexes),
                -index,
            ),
        )
        selected_indexes.append(best_index)
        remaining.remove(best_index)
    return [results[index] for index in selected_indexes]

