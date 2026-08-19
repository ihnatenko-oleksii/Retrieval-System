import json

import pytest

from app.evals.benchmark_protocol import FrozenSplitError, load_or_create_split, stratified_split


def make_cases() -> list[dict]:
    return [
        {
            "id": f"{category}-{index:02d}",
            "category": category,
            "question": f"Question {category} {index}",
            "relevance": {f"{category}.md::{index}": 3},
        }
        for category in ("ambiguous", "exact", "fine", "multiple", "semantic")
        for index in range(12)
    ]


class TestStratifiedSplit:
    def test_is_deterministic_disjoint_and_proportional(self):
        cases = make_cases()

        dev_one, test_one = stratified_split(cases, seed=123)
        dev_two, test_two = stratified_split(cases, seed=123)

        assert dev_one == dev_two
        assert test_one == test_two
        assert len(dev_one) == 40
        assert len(test_one) == 20
        assert {case["id"] for case in dev_one}.isdisjoint({case["id"] for case in test_one})
        assert {case["category"] for case in dev_one} == {case["category"] for case in test_one}
        assert all(sum(case["category"] == category for case in dev_one) == 8 for category in {case["category"] for case in cases})
        assert all(sum(case["category"] == category for case in test_one) == 4 for category in {case["category"] for case in cases})

    def test_rejects_duplicate_ids(self):
        cases = make_cases()
        cases[1]["id"] = cases[0]["id"]

        with pytest.raises(FrozenSplitError, match="unique"):
            stratified_split(cases)


class TestPersistedSplit:
    def test_first_run_persists_and_second_run_validates_same_split(self, tmp_path):
        source = tmp_path / "eval.jsonl"
        source.write_text("\n".join(json.dumps(case) for case in make_cases()) + "\n", encoding="utf-8")
        split_dir = tmp_path / "split"

        first = load_or_create_split(source, split_dir, seed=99)
        second = load_or_create_split(source, split_dir, seed=99)

        assert first.dev_ids == second.dev_ids
        assert first.test_ids == second.test_ids
        assert json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))["test_frozen"] is True

    def test_refuses_changed_source_after_test_is_frozen(self, tmp_path):
        source = tmp_path / "eval.jsonl"
        source.write_text("\n".join(json.dumps(case) for case in make_cases()) + "\n", encoding="utf-8")
        split_dir = tmp_path / "split"
        load_or_create_split(source, split_dir)

        changed = make_cases()
        changed[0]["question"] = "changed after freeze"
        source.write_text("\n".join(json.dumps(case) for case in changed) + "\n", encoding="utf-8")

        with pytest.raises(FrozenSplitError, match="source hash"):
            load_or_create_split(source, split_dir)

    def test_refuses_partial_split_directory(self, tmp_path):
        source = tmp_path / "eval.jsonl"
        source.write_text("\n".join(json.dumps(case) for case in make_cases()) + "\n", encoding="utf-8")
        split_dir = tmp_path / "split"
        split_dir.mkdir()
        (split_dir / "test.jsonl").write_text("", encoding="utf-8")

        with pytest.raises(FrozenSplitError, match="incomplete"):
            load_or_create_split(source, split_dir)
