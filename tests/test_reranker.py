from app.retrieval.reranker import Reranker
from tests.conftest import make_meta


def make_disabled_reranker() -> Reranker:
    # settings.reranker_on defaults to False and force_load=False, so this
    # never touches sentence-transformers / downloads a model.
    return Reranker(force_load=False)


class FakeCrossEncoder:
    """Deterministic stand-in for sentence_transformers.CrossEncoder."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return self.scores


def test_disabled_returns_input_unchanged():
    reranker = make_disabled_reranker()
    chunks = [("a", make_meta("a.md", 0), 0.1), ("b", make_meta("b.md", 0), 0.2)]

    result = reranker.rerank("query", chunks, top_n=5, enabled=False)

    assert result == chunks


def test_disabled_still_truncates_to_top_n():
    reranker = make_disabled_reranker()
    chunks = [("a", make_meta("a.md", 0), 0.1), ("b", make_meta("b.md", 0), 0.2)]

    result = reranker.rerank("query", chunks, top_n=1, enabled=False)

    assert result == [("a", make_meta("a.md", 0), 0.1)]


def test_empty_chunks_returns_empty():
    reranker = make_disabled_reranker()
    assert reranker.rerank("query", [], top_n=5, enabled=True) == []


def test_enabled_reorders_by_cross_encoder_score():
    reranker = make_disabled_reranker()
    meta_a = make_meta("a.md", 0)
    meta_b = make_meta("b.md", 0)
    chunks = [("low relevance", meta_a, 0.9), ("high relevance", meta_b, 0.1)]
    # Cross-encoder disagrees with the incoming (fusion) order.
    reranker.model = FakeCrossEncoder(scores=[0.1, 9.0])

    result = reranker.rerank("query", chunks, top_n=2, enabled=True)

    assert [content for content, _, _ in result] == ["high relevance", "low relevance"]
    assert result[0][2] == 9.0


def test_enabled_truncates_to_top_n():
    reranker = make_disabled_reranker()
    meta = make_meta("a.md", 0)
    chunks = [("a", meta, 0.0), ("b", meta, 0.0), ("c", meta, 0.0)]
    reranker.model = FakeCrossEncoder(scores=[1.0, 3.0, 2.0])

    result = reranker.rerank("query", chunks, top_n=2, enabled=True)

    assert len(result) == 2
    assert [content for content, _, _ in result] == ["b", "c"]


def test_model_prediction_failure_falls_back_to_original_order():
    reranker = make_disabled_reranker()
    meta = make_meta("a.md", 0)
    chunks = [("a", meta, 0.0), ("b", meta, 0.0)]

    class ExplodingCrossEncoder:
        def predict(self, pairs):
            raise RuntimeError("model backend unavailable")

    reranker.model = ExplodingCrossEncoder()

    result = reranker.rerank("query", chunks, top_n=5, enabled=True)

    assert result == chunks


def test_no_model_available_falls_back_without_loading(monkeypatch):
    reranker = make_disabled_reranker()
    # Simulate a model load failure (e.g. no network access to download it).
    monkeypatch.setattr(reranker, "_load_model", lambda: None)
    chunks = [("a", make_meta("a.md", 0), 0.0)]

    result = reranker.rerank("query", chunks, top_n=5, enabled=True)

    assert result == chunks
