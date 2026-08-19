"""Integrity-preserving utilities for the retrieval benchmark protocol."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SPLIT_SEED = 1729
DEFAULT_DEV_SIZE = 40
DEFAULT_TEST_SIZE = 20


class FrozenSplitError(ValueError):
    """Raised when a persisted split cannot be trusted as the frozen split."""


@dataclass(frozen=True)
class BenchmarkSplit:
    """A validated, deterministic benchmark split."""

    dev_cases: tuple[dict[str, Any], ...]
    test_cases: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    @property
    def dev_ids(self) -> tuple[str, ...]:
        return tuple(str(case["id"]) for case in self.dev_cases)

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(str(case["id"]) for case in self.test_cases)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_id(case: dict[str, Any], index: int) -> str:
    value = case.get("id")
    if not isinstance(value, str) or not value.strip():
        raise FrozenSplitError(f"Benchmark case at position {index} has no non-empty string id")
    return value


def _allocate_counts(category_counts: dict[str, int], target: int) -> dict[str, int]:
    """Allocate ``target`` cases proportionally using largest remainders."""
    total = sum(category_counts.values())
    if target < 0 or target > total:
        raise FrozenSplitError(f"Requested {target} cases from {total} available cases")
    if total == 0:
        return {category: 0 for category in category_counts}

    raw = {category: count * target / total for category, count in category_counts.items()}
    allocation = {category: int(value) for category, value in raw.items()}
    remaining = target - sum(allocation.values())
    ranked = sorted(
        category_counts,
        key=lambda category: (raw[category] - allocation[category], category),
        reverse=True,
    )
    for category in ranked[:remaining]:
        allocation[category] += 1
    return allocation


def _category_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for index, case in enumerate(cases):
        _case_id(case, index)
        category = case.get("category")
        if not isinstance(category, str) or not category.strip():
            raise FrozenSplitError(f"Benchmark case {_case_id(case, index)} has no non-empty category")
        counts[category] += 1
    return dict(sorted(counts.items()))


def stratified_split(
    cases: list[dict[str, Any]],
    *,
    dev_size: int = DEFAULT_DEV_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a deterministic stratified split while preserving source order."""
    if len(cases) != dev_size + test_size:
        raise FrozenSplitError(
            f"Split sizes must cover the source exactly: {dev_size} DEV + {test_size} TEST != {len(cases)} cases"
        )

    ids = [_case_id(case, index) for index, case in enumerate(cases)]
    if len(set(ids)) != len(ids):
        raise FrozenSplitError("Benchmark case ids must be unique")

    category_counts = _category_counts(cases)
    dev_by_category = _allocate_counts(category_counts, dev_size)
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, case in enumerate(cases):
        grouped[str(case["category"])].append((index, case))

    rng = random.Random(seed)
    dev_indexes: set[int] = set()
    test_indexes: set[int] = set()
    for category in sorted(grouped):
        members = list(grouped[category])
        rng.shuffle(members)
        dev_count = dev_by_category[category]
        dev_indexes.update(index for index, _ in members[:dev_count])
        test_indexes.update(index for index, _ in members[dev_count:])

    if dev_indexes & test_indexes or len(dev_indexes) != dev_size or len(test_indexes) != test_size:
        raise FrozenSplitError("Internal split allocation error: DEV and TEST are not disjoint and exhaustive")

    dev_cases = [cases[index] for index in range(len(cases)) if index in dev_indexes]
    test_cases = [cases[index] for index in range(len(cases)) if index in test_indexes]
    return dev_cases, test_cases


def _distribution(cases: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(case["category"]) for case in cases).items()))


def _build_manifest(
    *,
    source_path: Path,
    source_sha256: str,
    dev_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    total_distribution = Counter(str(case["category"]) for case in [*dev_cases, *test_cases])
    return {
        "protocol_version": 1,
        "source_eval": str(source_path),
        "source_sha256": source_sha256,
        "seed": seed,
        "dev_size": len(dev_cases),
        "test_size": len(test_cases),
        "test_frozen": True,
        "categories": {
            category: {
                "total": total_distribution[category],
                "dev": _distribution(dev_cases).get(category, 0),
                "test": _distribution(test_cases).get(category, 0),
            }
            for category in sorted(total_distribution)
        },
        "dev_case_ids": [str(case["id"]) for case in dev_cases],
        "test_case_ids": [str(case["id"]) for case in test_cases],
    }


def _write_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases),
        encoding="utf-8",
    )


def _validate_persisted_split(
    *,
    source_path: Path,
    dev_cases: list[dict[str, Any]],
    test_cases: list[dict[str, Any]],
    manifest: dict[str, Any],
    seed: int,
    dev_size: int,
    test_size: int,
) -> None:
    source_hash = _source_sha256(source_path)
    if manifest.get("source_sha256") != source_hash:
        raise FrozenSplitError(
            "Frozen split source hash does not match the current evaluation file; refusing to reuse TEST"
        )
    if manifest.get("seed") != seed or manifest.get("dev_size") != dev_size or manifest.get("test_size") != test_size:
        raise FrozenSplitError("Frozen split protocol parameters changed; refusing to reuse TEST")
    if manifest.get("test_frozen") is not True:
        raise FrozenSplitError("Persisted split is not marked as frozen")

    actual_dev_ids = [str(case.get("id")) for case in dev_cases]
    actual_test_ids = [str(case.get("id")) for case in test_cases]
    if actual_dev_ids != manifest.get("dev_case_ids") or actual_test_ids != manifest.get("test_case_ids"):
        raise FrozenSplitError("Persisted split case IDs do not match its manifest")
    if len(dev_cases) != dev_size or len(test_cases) != test_size:
        raise FrozenSplitError("Persisted split sizes do not match its manifest")

    source_cases = load_jsonl(source_path)
    source_by_id = {str(case["id"]): case for case in source_cases}
    persisted_cases = [*dev_cases, *test_cases]
    if set(actual_dev_ids) & set(actual_test_ids) or set(actual_dev_ids + actual_test_ids) != set(source_by_id):
        raise FrozenSplitError("Persisted DEV/TEST IDs are not a disjoint, exhaustive source split")
    if any(source_by_id.get(str(case["id"])) != case for case in persisted_cases):
        raise FrozenSplitError("Persisted split changed a question or relevance label")


def load_or_create_split(
    source_path: Path,
    split_dir: Path,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    dev_size: int = DEFAULT_DEV_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
) -> BenchmarkSplit:
    """Create the split once, then validate it on every later invocation."""
    source_path = source_path.resolve()
    split_dir = split_dir.resolve()
    if not source_path.exists():
        raise FrozenSplitError(f"Evaluation file not found: {source_path}")

    dev_path = split_dir / "dev.jsonl"
    test_path = split_dir / "test.jsonl"
    manifest_path = split_dir / "manifest.json"
    existing = [path.exists() for path in (dev_path, test_path, manifest_path)]
    if any(existing) and not all(existing):
        raise FrozenSplitError(f"Split directory is incomplete: {split_dir}")

    if not any(existing):
        cases = load_jsonl(source_path)
        dev_cases, test_cases = stratified_split(
            cases,
            dev_size=dev_size,
            test_size=test_size,
            seed=seed,
        )
        manifest = _build_manifest(
            source_path=source_path,
            source_sha256=_source_sha256(source_path),
            dev_cases=dev_cases,
            test_cases=test_cases,
            seed=seed,
        )
        split_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(dev_path, dev_cases)
        _write_jsonl(test_path, test_cases)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        dev_cases = load_jsonl(dev_path)
        test_cases = load_jsonl(test_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_persisted_split(
            source_path=source_path,
            dev_cases=dev_cases,
            test_cases=test_cases,
            manifest=manifest,
            seed=seed,
            dev_size=dev_size,
            test_size=test_size,
        )

    return BenchmarkSplit(tuple(dev_cases), tuple(test_cases), manifest)
