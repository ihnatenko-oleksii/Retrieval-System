from app.retrieval.prf import PseudoRelevanceFeedback


def test_feedback_query_preserves_original_and_extracts_bounded_terms():
    prf = PseudoRelevanceFeedback(max_terms=4)

    result = prf.build_query(
        "How do retries work?",
        [
            "Retries use exponential backoff for API-Retry-After and transient failures.",
            "The retry policy limits attempts and records retry metadata.",
        ],
        depth=2,
    )

    assert result.original_query == "How do retries work?"
    assert result.query.startswith("How do retries work?")
    assert result.terms
    assert len(result.terms) <= 4
    assert "API-Retry-After" in result.terms

