from inspect import signature

from app.retrieval.query_router import QueryGate


def test_gate_skips_llm_variants_for_exact_technical_queries():
    decision = QueryGate().decide('Which header gives "X-Atlas-Event-Id" its stable identity?')

    assert decision.should_rewrite is False
    assert decision.should_expand is False
    assert "quoted_term" in decision.protected_signals


def test_gate_routes_underspecified_natural_language_using_query_text_only():
    decision = QueryGate().decide("What happens when a request or event is repeated?")

    assert decision.should_rewrite is True
    assert decision.should_expand is True
    assert "underspecified natural-language signal" in decision.reasons


def test_gate_public_interface_has_no_benchmark_metadata_argument():
    assert list(signature(QueryGate.decide).parameters) == ["self", "query"]
