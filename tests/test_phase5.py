import json
from pathlib import Path

import pytest

from app.chunking.phase5 import build_phase5_chunks
from app.evals.span_relevance import load_markdown_spans, span_catalog_sha256
from app.retrieval.phase5 import (
    Phase5Config,
    _case_metrics,
    bootstrap_mean_delta,
    convert_cases_to_source_spans,
    corpus_sha256,
    query_instruction_mode,
    tokenize,
)
from scripts.create_benchmark_v4 import OUTPUT_ROOT, _build_cases, _question_sha256, _validate


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_benchmark_v4_is_sealed_balanced_and_disjoint_from_observed_cases():
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    cases = _jsonl(OUTPUT_ROOT / "eval.jsonl")
    corpus_root = OUTPUT_ROOT / "corpus"
    spans = load_markdown_spans(corpus_root)

    assert manifest["selection_eligible"] is False
    assert manifest["sealed_before_final_evaluation"] is True
    assert manifest["tuning_allowed"] is False
    assert manifest["document_count"] >= 50
    assert len(list((corpus_root / "python_stdlib").glob("*.md"))) == manifest["document_count"]
    assert len(cases) == 125
    assert manifest["category_counts"] == {
        "ambiguous": 25,
        "fine_grained": 25,
        "lexical": 25,
        "multiple_relevant": 25,
        "semantic": 25,
    }
    assert manifest["corpus_sha256"] == corpus_sha256(corpus_root)
    assert manifest["atlas_corpus_sha256"] != manifest["corpus_sha256"]
    assert manifest["span_catalog_sha256"] == span_catalog_sha256(spans)
    assert manifest["question_sha256"] == _question_sha256(cases)
    assert all("chunk_id" not in label for case in cases for label in case["relevance_spans"])
    assert all("::" not in label["span_id"] for case in cases for label in case["relevance_spans"])
    assert all("atlas" not in str(case["question"]).casefold() for case in cases)

    observed_paths = {
        path.name
        for path in (
            Path("docs/benchmark_corpus"),
            Path("docs/benchmark_v2/corpus"),
            Path("docs/benchmark_v3/corpus"),
        )
        if path.exists()
        for path in path.rglob("*.md")
    }
    v4_paths = {path.name for path in (corpus_root / "python_stdlib").glob("*.md")}
    assert observed_paths.isdisjoint(v4_paths)


def test_benchmark_v4_rebuild_is_deterministic_and_has_valid_span_labels():
    cases = _jsonl(OUTPUT_ROOT / "eval.jsonl")
    spans = load_markdown_spans(OUTPUT_ROOT / "corpus")

    _validate(cases)
    assert cases == _build_cases()
    assert all(label["start"] < label["end"] for case in cases for label in case["relevance_spans"])
    assert {label["span_id"] for case in cases for label in case["relevance_spans"]}.issubset(
        {span.span_id for span in spans}
    )


def test_phase5_chunking_strategies_return_source_intervals_and_context():
    text = (
        "# Root\n\n"
        "Root introduction with enough text to make a useful retrieval passage.\n\n"
        "## Child\n\n"
        "Child details include a stable identifier and an implementation constraint.\n\n"
        "## Sibling\n\n"
        "Sibling details explain the neighboring behavior and its boundary.\n"
    )

    for strategy in ("character", "paragraph", "heading", "section", "heading_context"):
        chunks = build_phase5_chunks(text, strategy, chunk_size=80, chunk_overlap=20)
        assert chunks
        assert all(0 <= chunk.start < chunk.end <= len(text) for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)

    context_chunks = build_phase5_chunks(text, "heading_context", chunk_size=80, chunk_overlap=20)
    assert any("Root" in chunk.text or "Child" in chunk.text for chunk in context_chunks)
    assert any(" > " in chunk.heading_path or chunk.heading_path == "Root" for chunk in context_chunks)


def test_technical_tokenizer_preserves_identifier_forms_and_parts():
    tokens = tokenize("snake_case kebab-case API.v1 status:404", technical=True)

    assert "snake_case" in tokens
    assert "snake" in tokens and "case" in tokens
    assert "kebab-case" in tokens
    assert "api.v1" in tokens
    assert "status:404" in tokens


def test_instruction_routing_uses_query_text_only():
    assert query_instruction_mode("What does timeout_ms=500 mean?") == "precise"
    assert query_instruction_mode("How does the retry concept work?") == "semantic"
    assert query_instruction_mode("What is the difference between the two modes?") == "context"
    assert query_instruction_mode("How do A and B fit together?") == "multi"
    assert query_instruction_mode("Where is the setting described?") == "generic"
    assert Phase5Config(instruction_mode="routed").instruction_mode == "routed"


def test_source_span_metrics_do_not_double_count_overlapping_chunks():
    case = {
        "id": "q1",
        "relevance_spans": [
            {"document_id": "doc.md", "span_id": "doc.md#section", "start": 10, "end": 30, "gain": 3}
        ],
    }
    records = [
        {
            "id": "doc.md::0",
            "metadata": {
                "file_name": "doc.md",
                "chunk_id": "doc.md::0",
                "source_char_start": 10,
                "source_char_end": 20,
            },
        },
        {
            "id": "doc.md::1",
            "metadata": {
                "file_name": "doc.md",
                "chunk_id": "doc.md::1",
                "source_char_start": 15,
                "source_char_end": 30,
            },
        },
    ]

    metrics = _case_metrics(case, records, top_k=5)

    assert metrics["recall@5"] == pytest.approx(1.0)
    assert metrics["ndcg"] == pytest.approx(1.0)
    assert metrics["ndcg"] <= 1.0
    assert metrics["precision@5"] == pytest.approx(0.2)


def test_legacy_chunk_labels_are_converted_to_source_intervals():
    canonical_records = [
        {
            "id": "doc.md::0",
            "metadata": {
                "chunk_id": "doc.md::0",
                "file_name": "doc.md",
                "heading_path": "Root",
                "source_char_start": 0,
                "source_char_end": 40,
            },
        }
    ]
    cases = [{"id": "q1", "question": "Where?", "relevance": {"doc.md::0": 3}}]

    converted = convert_cases_to_source_spans(cases, canonical_records)

    assert converted[0]["ground_truth_origin"] == "legacy_chunk_id_mapped_to_canonical_source_interval"
    assert "relevance" not in converted[0]
    assert converted[0]["relevance_spans"][0]["start"] == 0
    assert converted[0]["relevance_spans"][0]["end"] == 40


def test_bootstrap_delta_is_deterministic_and_reports_ci():
    first = bootstrap_mean_delta([0.5, 0.6, 0.7], [0.6, 0.7, 0.8], seed=1729, samples=100)
    second = bootstrap_mean_delta([0.5, 0.6, 0.7], [0.6, 0.7, 0.8], seed=1729, samples=100)

    assert first == second
    assert first["observed"] == pytest.approx(0.1)
    assert first["lower_95"] <= first["observed"] <= first["upper_95"]


def test_final_artifact_records_one_completed_comparison_and_uncertainty():
    result_path = Path("docs/retrieval_results/phase5_final_results.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["final_score_call_count"] == 1
    assert result["benchmark_v4"]["question_count"] == 125
    assert set(result["systems"]) == {"original-e5-dense", "phase3-qwen3-hybrid", "phase5-winner"}
    assert result["selection"]["benchmark_v4_used_for_selection"] is False
    assert result["selection"]["phase5_winner_equals_phase3"] is True
    for name in result["systems"]:
        assert all(0.0 <= float(value) <= 1.0 for value in result["systems"][name]["metrics"].values())
    for name in ("phase3-qwen3-hybrid", "phase5-winner"):
        for metric in ("mrr", "ndcg"):
            interval = result["comparisons"][name]["bootstrap_delta_vs_original_e5"][metric]
            assert interval["samples"] == 5000
            assert interval["lower_95"] <= interval["upper_95"]
