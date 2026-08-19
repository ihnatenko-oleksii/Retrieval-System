from app.retrieval.diversity import mmr_select
from tests.conftest import make_meta


def test_mmr_preserves_first_rank_and_can_select_a_complementary_candidate():
    results = [
        ("retry backoff details", make_meta("a.md", 0), 1.0),
        ("retry backoff details duplicated", make_meta("a.md", 1), 0.9),
        ("idempotency key and replay protection", make_meta("b.md", 0), 0.8),
    ]

    selected = mmr_select(results, top_k=2, relevance_weight=0.5)

    assert selected[0][0] == "retry backoff details"
    assert selected[1][0] == "idempotency key and replay protection"
