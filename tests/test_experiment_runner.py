import json

import pytest

from app.evals.benchmark_protocol import BenchmarkSplit
from app.evals.experiment_runner import (
    ExperimentResult,
    ExperimentSpec,
    InvalidChunkMapping,
    _category_metrics,
    _routing_metrics,
    _validate_chunk_labels,
    baseline_spec,
    default_retrieval_specs,
    llm_variants,
    rank_dev_results,
    relative_improvement,
    write_final_comparison,
    write_phase_artifacts,
)
from tests.conftest import make_chunk


def make_split() -> BenchmarkSplit:
    dev = tuple({"id": f"dev-{index}", "category": "cat-a"} for index in range(2))
    test = tuple({"id": f"test-{index}", "category": "cat-b"} for index in range(1))
    return BenchmarkSplit(
        dev_cases=dev,
        test_cases=test,
        manifest={"test_frozen": True, "dev_size": 2, "test_size": 1},
    )


def make_result(spec: ExperimentSpec, *, ndcg: float, mrr: float, phase: str = "dev") -> ExperimentResult:
    return ExperimentResult(
        phase=phase,
        spec=spec,
        status="ok",
        metrics={"recall@5": 0.5, "precision@5": 0.2, "mrr": mrr, "ndcg": ndcg},
        category_metrics={"cat-a": {"ndcg": ndcg, "mrr": mrr, "cases": 2}},
        elapsed_seconds=1.0,
        seconds_per_case=0.5,
    )


class TestExperimentMatrix:
    def test_default_matrix_covers_requested_experiment_families(self):
        specs = default_retrieval_specs()

        assert {spec.fusion_strategy for spec in specs} >= {"weighted_linear", "rrf", "weighted_rrf"}
        assert {spec.candidate_depth for spec in specs} >= {20, 30, 50, 75, 100}
        assert {spec.embedding_model for spec in specs} >= {"intfloat/multilingual-e5-base", "BAAI/bge-m3"}
        assert {spec.rerank_candidate_pool for spec in specs if spec.reranker_on} >= {10, 20, 30, 50}
        assert {(spec.chunk_size, spec.chunk_overlap) for spec in specs} >= {
            (400, 80),
            (600, 100),
            (800, 150),
            (1000, 200),
            (1200, 200),
        }
        assert baseline_spec().reranker_on is False
        assert baseline_spec().query_rewriting_on is False
        assert baseline_spec().query_expansion_on is False

    def test_llm_matrix_contains_selective_original_preserving_and_confidence_variants(self):
        variants = llm_variants(ExperimentSpec("bge-base", embedding_model="BAAI/bge-m3"))

        assert any(spec.rewrite_policy == "selective" and spec.include_original_query for spec in variants)
        assert any(spec.query_expansion_on and spec.expansion_policy == "selective" for spec in variants)
        assert any(spec.confidence_routing for spec in variants)
        assert {spec.multi_query_fusion_strategy for spec in variants} >= {"weighted_linear", "rrf", "weighted_rrf"}

    def test_chunk_sweep_rejects_same_id_with_changed_content(self):
        cases = [{"id": "case-1", "category": "exact", "relevance": {"a.md::0": 3}}]
        candidate = [make_chunk("new chunk boundaries", "a.md", 0)]
        reference = [make_chunk("canonical chunk boundaries", "a.md", 0)]

        with pytest.raises(InvalidChunkMapping, match="content changed"):
            _validate_chunk_labels(cases, candidate, reference_chunks=reference)

    def test_dev_ranking_uses_ndcg_then_mrr_and_ignores_failures(self):
        low = make_result(ExperimentSpec("low"), ndcg=0.8, mrr=0.99)
        high = make_result(ExperimentSpec("high"), ndcg=0.9, mrr=0.5)
        failed = ExperimentResult(
            phase="dev",
            spec=ExperimentSpec("failed"),
            status="error",
            metrics={},
            category_metrics={},
            elapsed_seconds=0.1,
            seconds_per_case=None,
            error="model unavailable",
        )

        assert [result.spec.name for result in rank_dev_results([low, failed, high])] == ["high", "low"]

    def test_relative_improvement_uses_required_formula(self):
        assert relative_improvement({"ndcg": 0.8}, {"ndcg": 0.9}, "ndcg") == 12.5
        assert relative_improvement({"ndcg": 0.0}, {"ndcg": 0.9}, "ndcg") is None


class TestExperimentArtifacts:
    def test_category_aggregation_uses_per_case_metrics(self):
        details = [
            {"category": "exact", "recall": 1.0, "precision": 0.2, "mrr": 1.0, "ndcg": 1.0},
            {"category": "exact", "recall": 0.5, "precision": 0.4, "mrr": 0.5, "ndcg": 0.5},
        ]

        assert _category_metrics(details, 5)["exact"] == {
            "recall@5": 0.75,
            "precision@5": 0.3,
            "mrr": 0.75,
            "ndcg": 0.75,
            "cases": 2,
        }

    def test_routing_metrics_report_rewrite_rate_and_latency_inputs(self):
        metrics = _routing_metrics(
            [
                {"rewrite_applied": True, "expansion_applied": False, "confidence_score": 0.2, "confidence_triggered": True, "query_variant_count": 2},
                {"rewrite_applied": False, "expansion_applied": True, "confidence_score": 0.8, "confidence_triggered": False, "query_variant_count": 3},
            ]
        )

        assert metrics["rewrite_rate_percent"] == 50.0
        assert metrics["expansion_rate_percent"] == 50.0
        assert metrics["confidence_trigger_count"] == 1.0
        assert metrics["average_query_variant_count"] == 2.5

    def test_writers_emit_machine_and_human_readable_outputs(self, tmp_path):
        split = make_split()
        baseline = make_result(baseline_spec(), ndcg=0.8, mrr=0.8)
        final = make_result(ExperimentSpec("final"), ndcg=0.9, mrr=0.9, phase="test")

        write_phase_artifacts([baseline], tmp_path, "dev", command="uv run command", split=split)
        payload = write_final_comparison(
            replace_result_phase(baseline, "test"),
            final,
            tmp_path,
            command="uv run command",
            split=split,
        )

        assert (tmp_path / "dev_results.csv").exists()
        assert (tmp_path / "dev_results.json").exists()
        assert (tmp_path / "dev_results.md").exists()
        assert (tmp_path / "final_test_results.csv").exists()
        assert (tmp_path / "final_test_results.json").exists()
        assert (tmp_path / "final_test_report.md").exists()
        assert payload["relative_improvement_percent"]["ndcg"] == 12.5
        assert "Improved retrieval nDCG" in payload["recommended_cv_bullet"]
        assert json.loads((tmp_path / "final_test_results.json").read_text())["test_frozen"] is True

    def test_final_writer_emits_three_way_frozen_test_comparison(self, tmp_path):
        split = make_split()
        baseline = make_result(baseline_spec(), ndcg=0.8, mrr=0.8)
        previous = make_result(ExperimentSpec("previous"), ndcg=0.85, mrr=0.85, phase="test")
        new = make_result(ExperimentSpec("new"), ndcg=0.9, mrr=0.9, phase="test")

        payload = write_final_comparison(
            replace_result_phase(baseline, "test"),
            new,
            tmp_path,
            command="uv run command",
            split=split,
            previous_final=previous,
        )

        assert payload["previous_final"]["name"] == "previous"
        assert payload["new_final"]["name"] == "new"
        csv_text = (tmp_path / "final_test_results.csv").read_text()
        assert csv_text.count("\n") == 4
        report = (tmp_path / "final_test_report.md").read_text()
        assert "| Previous final |" in report
        assert "| New final |" in report
        assert "Query routing and generalized failure modes" in report


def replace_result_phase(result: ExperimentResult, phase: str) -> ExperimentResult:
    return ExperimentResult(
        phase=phase,
        spec=result.spec,
        status=result.status,
        metrics=result.metrics,
        category_metrics=result.category_metrics,
        elapsed_seconds=result.elapsed_seconds,
        seconds_per_case=result.seconds_per_case,
        error=result.error,
        index_reused=result.index_reused,
    )
