import numpy as np

from app.retrieval.ltr import GroupedLTR, LTRFeatureExtractor, grouped_query_folds


def test_ltr_features_include_component_scores_and_query_safe_signals():
    features = LTRFeatureExtractor.extract(
        "How does API-Retry work?",
        {
            "content": "API-Retry uses exponential backoff.",
            "dense_score": 0.8,
            "dense_rank": 2,
            "sparse_score": 4.0,
            "sparse_rank": 1,
            "late_interaction_score": 0.7,
            "late_interaction_rank": 3,
            "stream_count": 2,
        },
    )

    assert features["dense_score"] == 0.8
    assert features["sparse_rank"] == 1.0
    assert features["exact_token_overlap"] > 0.0
    assert features["stream_count"] == 2.0
    assert "case_id" not in features


def test_grouped_query_folds_never_split_a_query_between_train_and_validation():
    query_ids = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4", "q5", "q5"]

    folds = grouped_query_folds(query_ids, n_splits=5, seed=1729)

    assert len(folds) == 5
    for train_indexes, validation_indexes in folds:
        assert set(train_indexes).isdisjoint(validation_indexes)
        assert {query_ids[index] for index in train_indexes}.isdisjoint(
            {query_ids[index] for index in validation_indexes}
        )


def test_grouped_ltr_fits_and_predicts_with_query_groups():
    ranker = GroupedLTR(n_estimators=8, random_state=1729)
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    labels = np.asarray([3.0, 0.0, 3.0, 0.0])
    query_ids = ["q1", "q1", "q2", "q2"]

    ranker.fit(features, labels, query_ids)

    predictions = ranker.predict(features)

    assert predictions.shape == (4,)
    assert predictions[0] > predictions[1]


def test_pairwise_linear_ltr_uses_within_query_preferences():
    ranker = GroupedLTR(model_name="pairwise-linear", random_state=1729)
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    labels = np.asarray([3.0, 0.0, 3.0, 0.0])
    query_ids = ["q1", "q1", "q2", "q2"]

    ranker.fit(features, labels, query_ids)
    predictions = ranker.predict(features)

    assert ranker.backend_name == "pairwise-linear"
    assert predictions[0] > predictions[1]
