import json
from pathlib import Path

from app.evals.span_relevance import load_markdown_spans, span_catalog_sha256, span_matches_metadata
from scripts.create_benchmark_v3 import (
    CORPUS_ROOT,
    OUTPUT_ROOT,
    _corpus_sha256,
    _question_sha256,
    build_cases,
    validate_cases,
)


def load_cases() -> list[dict]:
    return [json.loads(line) for line in (OUTPUT_ROOT / "eval.jsonl").read_text().splitlines() if line.strip()]


def test_benchmark_v3_is_balanced_sealed_and_corpus_bound():
    cases = load_cases()
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text())
    spans = load_markdown_spans(CORPUS_ROOT)

    assert len(cases) == 100
    assert manifest["selection_eligible"] is False
    assert manifest["sealed_before_phase4_selection"] is True
    assert manifest["corpus_sha256"] == _corpus_sha256(CORPUS_ROOT)
    assert manifest["span_catalog_sha256"] == span_catalog_sha256(spans)
    assert manifest["question_sha256"] == _question_sha256(cases)
    assert manifest["category_counts"] == {
        "ambiguous": 20,
        "fine_grained": 20,
        "lexical": 20,
        "multiple_relevant": 20,
        "semantic": 20,
    }


def test_benchmark_v3_rebuild_is_deterministic_and_span_labeled():
    cases = load_cases()
    spans = load_markdown_spans(CORPUS_ROOT)
    validate_cases(cases, spans)

    assert cases == build_cases(spans)
    assert all("chunk_id" not in label for case in cases for label in case["relevance_spans"])
    assert all("::" not in label["span_id"] for case in cases for label in case["relevance_spans"])


def test_source_span_matching_requires_same_document_and_interval_overlap():
    label = {
        "document_id": "atlas/api-retries.md",
        "start": 100,
        "end": 200,
    }
    assert span_matches_metadata(
        label,
        {"file_name": "atlas/api-retries.md", "source_char_start": 150, "source_char_end": 240},
    )
    assert not span_matches_metadata(
        label,
        {"file_name": "atlas/api-errors.md", "source_char_start": 150, "source_char_end": 240},
    )
    assert not span_matches_metadata(
        label,
        {"file_name": "atlas/api-retries.md", "source_char_start": 200, "source_char_end": 240},
    )


def test_v3_artifact_does_not_reuse_observed_questions():
    def questions(path: Path) -> set[str]:
        return {
            " ".join(str(json.loads(line)["question"]).casefold().split())
            for line in path.read_text().splitlines()
            if line.strip()
        }

    v3_questions = questions(OUTPUT_ROOT / "eval.jsonl")
    observed = questions(Path("docs/benchmark_eval.jsonl")) | questions(Path("docs/benchmark_v2/eval.jsonl"))
    assert v3_questions.isdisjoint(observed)
