import pytest

from app.retrieval.native_bge import NativeBackendUnavailable, NativeBGEBackend
from tests.conftest import make_meta


class FakeBGEModel:
    def encode(self, texts, **kwargs):
        return {
            "dense_vecs": [[1.0, 0.0] if "first" in text else [0.0, 1.0] for text in texts],
            "lexical_weights": [{"first": 1.0} if "first" in text else {"second": 1.0} for text in texts],
            "colbert_vecs": [[[1.0]] if "first" in text else [[0.2]] for text in texts],
        }

    @staticmethod
    def compute_lexical_matching_score(query, document):
        return 2.0 if "first" in document else 0.1

    @staticmethod
    def colbert_score(query, document):
        return float(document[0][0])


def test_native_backend_records_all_component_scores_and_ranks():
    backend = NativeBGEBackend(model=FakeBGEModel())

    result = backend.search(
        "first query",
        [("first passage", make_meta("first.md", 0)), ("second passage", make_meta("second.md", 0))],
        top_n=2,
        dense_weight=0.4,
        sparse_weight=0.3,
        colbert_weight=0.3,
    )

    assert result.results[0][0] == "first passage"
    assert result.candidate_scores[0].dense_score == pytest.approx(1.0)
    assert result.candidate_scores[0].sparse_score == pytest.approx(2.0)
    assert result.candidate_scores[0].late_interaction_score == pytest.approx(1.0)
    assert result.candidate_scores[0].dense_rank == 1
    assert {"dense_score", "sparse_score", "late_interaction_score", "candidate_rank"} <= result.trace[0].keys()


def test_unavailable_native_runtime_has_explicit_reason():
    backend = NativeBGEBackend(model=None, unavailable_reason="FlagEmbedding is not installed")

    with pytest.raises(NativeBackendUnavailable, match="FlagEmbedding is not installed"):
        backend.search("query", [("passage", make_meta("a.md", 0))], top_n=1)

