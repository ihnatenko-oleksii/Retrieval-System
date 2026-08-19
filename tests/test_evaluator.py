import math

import pytest

from app.evals.evaluator import Evaluator
from tests.conftest import make_meta


def make_evaluator(top_k=3, skip_generation=False) -> Evaluator:
    """Bare Evaluator with no real stores/retriever/generator constructed."""
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = None
    evaluator.top_k = top_k
    evaluator.skip_generation = skip_generation
    return evaluator


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_k=None, config=None):
        return self.chunks


class FakeGenerator:
    def __init__(self, answer_text):
        self.answer_text = answer_text
        self.calls = []

    def generate_answer(self, question, chunks, chat_history=None):
        self.calls.append((question, chunks))
        return {"final_answer": self.answer_text}


def write_jsonl(tmp_path, cases):
    import json

    path = tmp_path / "evals.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in cases), encoding="utf-8")
    return str(path)


class TestSourceMatching:
    def test_empty_expected_source_never_matches(self):
        evaluator = make_evaluator()
        assert evaluator._source_matches("", make_meta("docs/a.md", 0)) is False

    def test_exact_relative_path_matches(self):
        evaluator = make_evaluator()
        meta = make_meta("docs/architecture/a.md", 0)
        assert evaluator._source_matches("docs/architecture/a.md", meta) is True

    def test_basename_only_matches_legacy_ingestion(self):
        evaluator = make_evaluator()
        meta = make_meta("a.md", 0, file_name="a.md")
        assert evaluator._source_matches("docs/architecture/a.md", meta) is True

    def test_case_and_slash_insensitive(self):
        evaluator = make_evaluator()
        meta = make_meta("Docs\\A.MD", 0, file_name="Docs\\A.MD")
        assert evaluator._source_matches("docs/a.md", meta) is True

    def test_unrelated_source_does_not_match(self):
        evaluator = make_evaluator()
        meta = make_meta("docs/b.md", 0, file_name="b.md")
        assert evaluator._source_matches("docs/a.md", meta) is False


class TestNdcgCorrectness:
    """
    Regression tests for the nDCG bug: idcg was hardcoded to 1.0, so any case
    with more than one relevant retrieved chunk produced ndcg > 1, which is
    mathematically impossible for a normalized metric.
    """

    def test_single_relevant_chunk_at_rank_one_is_perfect(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        meta_hit = make_meta("a.md", 0, file_name="a.md")
        meta_miss = make_meta("b.md", 0, file_name="b.md")
        evaluator.retriever = FakeRetriever([("t1", meta_hit, 0.9), ("t2", meta_miss, 0.5), ("t3", meta_miss, 0.4)])
        evaluator.generator = FakeGenerator("some answer")

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": []}])
        metrics, _ = evaluator.evaluate_cases(jsonl)

        assert metrics["ndcg"] == pytest.approx(1.0)

    def test_ndcg_never_exceeds_one_with_multiple_relevant_hits(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        meta_hit = make_meta("a.md", 0, file_name="a.md")
        meta_miss = make_meta("b.md", 0, file_name="b.md")
        # relevant at rank 1 and rank 3
        evaluator.retriever = FakeRetriever([("t1", meta_hit, 0.9), ("t2", meta_miss, 0.5), ("t3", meta_hit, 0.4)])
        evaluator.generator = FakeGenerator("some answer")

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": []}])
        metrics, _ = evaluator.evaluate_cases(jsonl)

        dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        expected_ndcg = dcg / idcg
        assert metrics["ndcg"] == pytest.approx(round(expected_ndcg, 4))
        assert metrics["ndcg"] <= 1.0

    def test_no_relevant_hit_gives_zero_ndcg(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        meta_miss = make_meta("b.md", 0, file_name="b.md")
        evaluator.retriever = FakeRetriever([("t1", meta_miss, 0.9)])
        evaluator.generator = FakeGenerator("some answer")

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": []}])
        metrics, _ = evaluator.evaluate_cases(jsonl)

        assert metrics["ndcg"] == 0.0


class TestRetrievalMetrics:
    def test_recall_precision_mrr_for_single_hit_at_rank_two(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        meta_hit = make_meta("a.md", 0, file_name="a.md")
        meta_miss = make_meta("b.md", 0, file_name="b.md")
        evaluator.retriever = FakeRetriever([("t1", meta_miss, 0.9), ("t2", meta_hit, 0.5), ("t3", meta_miss, 0.4)])
        evaluator.generator = FakeGenerator("some answer")

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": []}])
        metrics, details = evaluator.evaluate_cases(jsonl)

        assert metrics["recall@3"] == 1.0
        assert metrics["precision@3"] == pytest.approx(1 / 3, abs=1e-4)
        assert metrics["mrr"] == pytest.approx(0.5)
        assert details[0]["first_relevant_rank"] == 2

    def test_miss_contributes_zero_to_all_retrieval_metrics(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        meta_miss = make_meta("b.md", 0, file_name="b.md")
        evaluator.retriever = FakeRetriever([("t1", meta_miss, 0.9)])
        evaluator.generator = FakeGenerator("some answer")

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": []}])
        metrics, details = evaluator.evaluate_cases(jsonl)

        assert metrics["recall@3"] == 0.0
        assert metrics["precision@3"] == 0.0
        assert metrics["mrr"] == 0.0
        assert details[0]["source_hit"] is False
        assert details[0]["first_relevant_rank"] is None

    def test_chunk_level_graded_labels_use_total_relevance_for_recall_and_ndcg(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        meta_miss = make_meta("c.md", 0, file_name="c.md")
        meta_relevant = make_meta("b.md", 0, file_name="b.md")
        evaluator.retriever = FakeRetriever(
            [
                ("unrelated", meta_miss, 0.9),
                ("related", meta_relevant, 0.8),
            ]
        )
        evaluator.generator = FakeGenerator("some answer")

        jsonl = write_jsonl(
            tmp_path,
            [
                {
                    "question": "q",
                    "relevance": {"a.md::0": 3, "b.md::0": 1},
                    "expected_keywords": [],
                }
            ],
        )
        metrics, details = evaluator.evaluate_cases(jsonl)

        assert metrics["recall@3"] == pytest.approx(0.5)
        assert metrics["precision@3"] == pytest.approx(1 / 3, abs=1e-4)
        assert metrics["mrr"] == pytest.approx(0.5)
        assert 0.0 < metrics["ndcg"] < 1.0
        assert details[0]["relevant_chunk_ids"] == "a.md::0 | b.md::0"


class TestKeywordHitRate:
    def test_all_keywords_present_scores_one(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        evaluator.retriever = FakeRetriever([])
        evaluator.generator = FakeGenerator("dependency injection uses inversion of control")

        jsonl = write_jsonl(
            tmp_path,
            [
                {
                    "question": "q",
                    "expected_source": "",
                    "expected_keywords": ["dependency injection", "inversion of control"],
                }
            ],
        )
        metrics, _ = evaluator.evaluate_cases(jsonl)

        assert metrics["keyword_hit_rate"] == 1.0

    def test_partial_keyword_match_scores_fraction(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        evaluator.retriever = FakeRetriever([])
        evaluator.generator = FakeGenerator("dependency injection is a pattern")

        jsonl = write_jsonl(
            tmp_path,
            [
                {
                    "question": "q",
                    "expected_source": "",
                    "expected_keywords": ["dependency injection", "inversion of control"],
                }
            ],
        )
        metrics, _ = evaluator.evaluate_cases(jsonl)

        assert metrics["keyword_hit_rate"] == pytest.approx(0.5)

    def test_no_expected_keywords_defaults_to_full_score(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        evaluator.retriever = FakeRetriever([])
        evaluator.generator = FakeGenerator("anything")

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "", "expected_keywords": []}])
        metrics, _ = evaluator.evaluate_cases(jsonl)

        assert metrics["keyword_hit_rate"] == 1.0


class TestSkipGeneration:
    def test_skip_generation_never_calls_generator(self, tmp_path):
        evaluator = make_evaluator(top_k=3, skip_generation=True)
        meta_hit = make_meta("a.md", 0, file_name="a.md")
        evaluator.retriever = FakeRetriever([("t1", meta_hit, 0.9)])
        generator = FakeGenerator("answer")
        evaluator.generator = generator

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": ["foo"]}])
        metrics, details = evaluator.evaluate_cases(jsonl)

        assert generator.calls == []
        assert "keyword_hit_rate" not in metrics
        assert details[0]["keyword_hit_score"] is None
        # Retrieval metrics are still computed without an LLM.
        assert metrics["recall@3"] == 1.0

    def test_default_still_calls_generator(self, tmp_path):
        evaluator = make_evaluator(top_k=3, skip_generation=False)
        meta_hit = make_meta("a.md", 0, file_name="a.md")
        evaluator.retriever = FakeRetriever([("t1", meta_hit, 0.9)])
        generator = FakeGenerator("answer")
        evaluator.generator = generator

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": []}])
        metrics, _ = evaluator.evaluate_cases(jsonl)

        assert len(generator.calls) == 1
        assert "keyword_hit_rate" in metrics


class TestEvaluatorFailureBehavior:
    def test_missing_file_returns_empty_metrics(self):
        evaluator = make_evaluator()
        metrics, details = evaluator.evaluate_cases("/no/such/file.jsonl")
        assert metrics == {}
        assert details == []

    def test_empty_file_returns_empty_metrics(self, tmp_path):
        evaluator = make_evaluator()
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        metrics, details = evaluator.evaluate_cases(str(path))
        assert metrics == {}
        assert details == []

    def test_evaluate_file_returns_only_metrics(self, tmp_path):
        evaluator = make_evaluator(top_k=3)
        meta_hit = make_meta("a.md", 0, file_name="a.md")
        evaluator.retriever = FakeRetriever([("t1", meta_hit, 0.9)])
        evaluator.generator = FakeGenerator("answer")

        jsonl = write_jsonl(tmp_path, [{"question": "q", "expected_source": "a.md", "expected_keywords": []}])
        metrics = evaluator.evaluate_file(jsonl)

        assert "ndcg" in metrics
        assert isinstance(metrics, dict)
