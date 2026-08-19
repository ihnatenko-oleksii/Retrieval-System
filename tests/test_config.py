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

    def test_is_independent_of_later_settings_mutation(self, monkeypatch):
        cfg = RetrievalConfig.from_settings()
        monkeypatch.setattr(settings, "retrieval_top_k", 999)
        assert cfg.top_k != 999
