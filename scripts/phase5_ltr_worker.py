"""Fit one isolated XGBoost LambdaMART model for the Phase 5 harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.ltr import GroupedLTR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = np.load(args.input, allow_pickle=False)
    train_features = np.asarray(payload["train_features"], dtype=float)
    train_labels = np.asarray(payload["train_labels"], dtype=float)
    train_query_ids = [str(value) for value in payload["train_query_ids"].tolist()]
    validation_features = np.asarray(payload["validation_features"], dtype=float)
    ranker = GroupedLTR(model_name="xgboost-lambdamart", n_estimators=80, random_state=1729)
    ranker.fit(train_features, train_labels, train_query_ids)
    if ranker.backend_name != "xgboost-lambdamart":
        raise RuntimeError(f"worker did not use real LambdaMART: {ranker.backend_name}")
    np.save(args.output, ranker.predict(validation_features), allow_pickle=False)
    print(ranker.backend_name)


if __name__ == "__main__":
    main()
