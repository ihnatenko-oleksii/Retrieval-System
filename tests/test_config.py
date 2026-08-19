import pytest

from app.core.config import Settings, settings
from app.core.runtime_config import RetrievalConfig


class TestSettingsDefaults:
    def test_default_hybrid_weights_sum_to_one(self):
        assert settings.hybrid_search_weights_dense + settings.hybrid_search_weights_sparse == 1.0

    def test_reranker_and_query_features_default_off(self):
        # Sensible baseline: no LLM-driven query preprocessing or reranking
        # unless a user opts in, since those add latency and require Ollama.
        assert settings.reranker_on is False
        assert settings.query_rewriting_on is False
        assert settings.query_expansion_on is False

    def test_experiment_defaults_are_explicit(self):
        assert settings.retrieval_candidate_depth == 20
        assert settings.fusion_strategy == "weighted_linear"
        assert settings.rerank_candidate_pool == 20

    def test_overrides_via_constructor(self):
        custom = Settings(retrieval_top_k=10, hybrid_search_weights_dense=0.5, hybrid_search_weights_sparse=0.5)
        assert custom.retrieval_top_k == 10
        assert custom.hybrid_search_weights_dense == 0.5


class TestRetrievalConfigFromSettings:
    def test_mirrors_global_settings(self):
        cfg = RetrievalConfig.from_settings()

        assert cfg.top_k == settings.retrieval_top_k
        assert cfg.dense_weight == settings.hybrid_search_weights_dense
        assert cfg.sparse_weight == settings.hybrid_search_weights_sparse
        assert cfg.reranker_on == settings.reranker_on
        assert cfg.rerank_top_n == settings.rerank_top_n
        assert cfg.query_rewriting_on == settings.query_rewriting_on
        assert cfg.query_expansion_on == settings.query_expansion_on
        assert cfg.llm_model == settings.llm_model
        assert cfg.candidate_depth == settings.retrieval_candidate_depth
        assert cfg.fusion_strategy == settings.fusion_strategy
        assert cfg.rerank_candidate_pool == settings.rerank_candidate_pool

    def test_is_independent_of_later_settings_mutation(self, monkeypatch):
        cfg = RetrievalConfig.from_settings()
        monkeypatch.setattr(settings, "retrieval_top_k", 999)
        assert cfg.top_k != 999

    def test_rejects_unknown_fusion_strategy(self):
        with pytest.raises(ValueError, match="Unsupported fusion strategy"):
            RetrievalConfig(
                top_k=5,
                dense_weight=1.0,
                sparse_weight=0.0,
                reranker_on=False,
                rerank_top_n=5,
                query_rewriting_on=False,
                query_expansion_on=False,
                fusion_strategy="not-a-strategy",
            )
