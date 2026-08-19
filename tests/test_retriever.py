from unittest.mock import MagicMock

import pytest

from app.core.runtime_config import RetrievalConfig
from app.retrieval.retriever import Retriever
from tests.conftest import make_meta


def make_retriever(vector_store=None, bm25_store=None, reranker=None) -> Retriever:
    return Retriever(
        vector_store=vector_store or MagicMock(),
        bm25_store=bm25_store or MagicMock(),
        reranker=reranker or MagicMock(),
    )


def base_config(**overrides) -> RetrievalConfig:
    defaults = dict(
        top_k=3,
        dense_weight=0.7,
        sparse_weight=0.3,
        reranker_on=False,
        rerank_top_n=3,
        query_rewriting_on=False,
        query_expansion_on=False,
        llm_model="test-model",
    )
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


class TestNormalizeScores:
    def test_empty_input(self):
        retriever = make_retriever()
        assert retriever._normalize_scores([]) == []

    def test_min_max_normalization(self):
        retriever = make_retriever()
        results = [("a", {}, 10.0), ("b", {}, 0.0), ("c", {}, 5.0)]
        normalized = retriever._normalize_scores(results)
        scores = {content: score for content, _, score in normalized}
        assert scores["b"] == pytest.approx(0.0)
        assert scores["a"] == pytest.approx(1.0)
        assert scores["c"] == pytest.approx(0.5)

    def test_invert_flips_ranking(self):
        # Dense results come back as *distances* (lower = better), so inverting
        # must turn the smallest distance into the highest normalized score.
        retriever = make_retriever()
        results = [("close", {}, 0.1), ("far", {}, 0.9)]
        normalized = retriever._normalize_scores(results, invert=True)
        scores = {content: score for content, _, score in normalized}
        assert scores["close"] > scores["far"]

    def test_all_equal_scores_normalize_to_one(self):
        retriever = make_retriever()
        results = [("a", {}, 3.0), ("b", {}, 3.0)]
        normalized = retriever._normalize_scores(results)
        assert all(score == 1.0 for _, _, score in normalized)


class TestFusionAndRouting:
    def test_acronym_extraction_does_not_treat_sentence_initial_word_as_acronym(self):
        retriever = make_retriever()

        assert retriever._extract_acronyms("What is CI?") == ["CI"]

    def test_explicit_candidate_depth_is_independent_of_final_top_k(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        vector_store.query.return_value = {"documents": [[]]}
        bm25_store.query.return_value = []
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)

        retriever.retrieve("q", config=base_config(top_k=5, candidate_depth=75))

        assert vector_store.query.call_args.kwargs["n_results"] == 75

    def test_rrf_fusion_uses_rank_agreement(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        meta_a = make_meta("docs/a.md", 0)
        meta_b = make_meta("docs/b.md", 0)
        meta_c = make_meta("docs/c.md", 0)
        vector_store.query.return_value = {
            "documents": [["dense a", "dense b"]],
            "metadatas": [[meta_a, meta_b]],
            "distances": [[0.1, 0.2]],
        }
        bm25_store.query.return_value = [
            ("sparse b", meta_b, 10.0),
            ("sparse c", meta_c, 9.0),
        ]
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)

        results = retriever.retrieve(
            "query",
            config=base_config(
                top_k=3,
                fusion_strategy="rrf",
                adaptive_routing=False,
                dense_weight=1.0,
                sparse_weight=1.0,
            ),
        )

        assert results[0][1]["file_path"] == "docs/b.md"

    def test_adaptive_weights_uses_raw_query_signals(self):
        retriever = make_retriever()

        assert retriever._adaptive_weights("What is CI?", 0.7, 0.3) == (0.35, 0.65)
        assert retriever._adaptive_weights("How does the service handle pagination cursors?", 0.7, 0.3) == (0.8, 0.2)

    def test_selective_policy_skips_rewrite_for_protected_query(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        rewriter = MagicMock()
        meta = make_meta("docs/events.md", 0)
        vector_store.query.return_value = {
            "documents": [["event identity"]],
            "metadatas": [[meta]],
            "distances": [[0.1]],
        }
        bm25_store.query.return_value = []
        retriever = Retriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            reranker=MagicMock(),
            rewriter=rewriter,
        )

        retriever.retrieve(
            'What is "X-Atlas-Event-Id"?',
            config=base_config(
                query_rewriting_on=True,
                rewrite_policy="selective",
                include_original_query=True,
                dense_weight=1.0,
                sparse_weight=0.0,
            ),
        )

        rewriter.rewrite_query.assert_not_called()
        assert retriever.last_trace["rewrite_applied"] is False

    def test_original_and_rewrite_are_fused_in_a_second_rank_stage(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        rewriter = MagicMock()
        meta_a = make_meta("docs/a.md", 0)
        meta_b = make_meta("docs/b.md", 0)
        meta_c = make_meta("docs/c.md", 0)

        def dense_query(*, query_text, n_results):
            if query_text == "original query":
                docs = ["a", "b"]
                metas = [meta_a, meta_b]
            else:
                docs = ["b", "c"]
                metas = [meta_b, meta_c]
            return {
                "documents": [docs],
                "metadatas": [metas],
                "distances": [[0.1, 0.2]],
            }

        vector_store.query.side_effect = dense_query
        bm25_store.query.return_value = []
        rewriter.rewrite_query.return_value = "rewritten query"
        retriever = Retriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            reranker=MagicMock(),
            rewriter=rewriter,
        )

        results = retriever.retrieve(
            "original query",
            config=base_config(
                query_rewriting_on=True,
                rewrite_policy="always",
                include_original_query=True,
                multi_query_fusion_strategy="weighted_rrf",
                dense_weight=1.0,
                sparse_weight=0.0,
                candidate_depth=2,
            ),
        )

        assert [meta["file_path"] for _, meta, _ in results] == ["docs/b.md", "docs/a.md", "docs/c.md"]
        assert [call.kwargs["query_text"] for call in vector_store.query.call_args_list] == [
            "original query",
            "rewritten query",
        ]
        assert retriever.last_trace["query_variant_labels"] == ["original", "rewrite"]

    def test_low_confidence_can_trigger_selective_rewrite_after_cheap_retrieval(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        rewriter = MagicMock()
        meta = make_meta("docs/uncertain.md", 0)
        vector_store.query.return_value = {
            "documents": [["weak match"]],
            "metadatas": [[meta]],
            "distances": [[0.7]],
        }
        bm25_store.query.return_value = []
        rewriter.rewrite_query.return_value = "more specific query"
        retriever = Retriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            reranker=MagicMock(),
            rewriter=rewriter,
        )

        retriever.retrieve(
            "short query",
            config=base_config(
                query_rewriting_on=True,
                rewrite_policy="selective",
                include_original_query=True,
                confidence_routing=True,
                confidence_threshold=0.9,
                dense_weight=1.0,
                sparse_weight=0.0,
            ),
        )

        rewriter.rewrite_query.assert_called_once()
        assert retriever.last_trace["confidence_triggered"] is True

    def test_empty_or_unchanged_rewrite_falls_back_to_original_results(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        rewriter = MagicMock()
        meta = make_meta("docs/original.md", 0)
        vector_store.query.return_value = {
            "documents": [["original hit"]],
            "metadatas": [[meta]],
            "distances": [[0.1]],
        }
        bm25_store.query.return_value = []
        rewriter.rewrite_query.return_value = "original query"
        retriever = Retriever(
            vector_store=vector_store,
            bm25_store=bm25_store,
            reranker=MagicMock(),
            rewriter=rewriter,
        )

        results = retriever.retrieve(
            "original query",
            config=base_config(query_rewriting_on=True, rewrite_policy="always", include_original_query=False, dense_weight=1.0, sparse_weight=0.0),
        )

        assert results[0][0] == "original hit"


class TestChunkIdentity:
    def test_same_metadata_same_identity(self):
        retriever = make_retriever()
        meta = make_meta("docs/a.md", 0)
        assert retriever._chunk_identity(meta, "some text") == retriever._chunk_identity(meta, "some text")

    def test_identical_content_different_source_is_distinct(self):
        # This is the dedup bug: two different chunks that happen to share
        # identical text must not collapse into a single retrieval result.
        retriever = make_retriever()
        meta_a = make_meta("docs/a.md", 0)
        meta_b = make_meta("docs/b.md", 0)
        assert retriever._chunk_identity(meta_a, "Introduction") != retriever._chunk_identity(meta_b, "Introduction")

    def test_falls_back_to_content_when_metadata_missing(self):
        retriever = make_retriever()
        assert retriever._chunk_identity({}, "hello") == ("hello",)
        assert retriever._chunk_identity({}, "hello") != retriever._chunk_identity({}, "world")


class TestHybridFusionDedup:
    def test_duplicate_content_from_different_sources_both_survive(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        meta_a = make_meta("docs/a.md", 0)
        meta_b = make_meta("docs/b.md", 0)
        vector_store.query.return_value = {
            "documents": [["Introduction", "Introduction"]],
            "metadatas": [[meta_a, meta_b]],
            "distances": [[0.1, 0.2]],
        }
        bm25_store.query.return_value = []

        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)
        results = retriever.retrieve("intro", config=base_config(top_k=5))

        sources = {meta.get("file_path") for _, meta, _ in results}
        assert sources == {"docs/a.md", "docs/b.md"}
        assert len(results) == 2

    def test_same_chunk_from_dense_and_sparse_merges_scores(self):
        # Two candidates per side so min-max normalization isn't degenerate
        # (a single result always normalizes to a tie).
        vector_store = MagicMock()
        bm25_store = MagicMock()
        meta = make_meta("docs/a.md", 0)
        meta_other_dense = make_meta("docs/other-dense.md", 0)
        meta_other_sparse = make_meta("docs/other-sparse.md", 0)
        vector_store.query.return_value = {
            "documents": [["shared text", "other dense"]],
            "metadatas": [[meta, meta_other_dense]],
            "distances": [[0.0, 1.0]],
        }
        bm25_store.query.return_value = [
            ("shared text", meta, 5.0),
            ("other sparse", meta_other_sparse, 1.0),
        ]

        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)
        results = retriever.retrieve("query", config=base_config(top_k=5))

        scores = {content: score for content, _, score in results}
        # Best dense distance (inverted -> 1.0) * 0.7 + best sparse score
        # (normalized -> 1.0) * 0.3 = 1.0, and it must rank first.
        assert scores["shared text"] == pytest.approx(1.0)
        assert scores["shared text"] == max(scores.values())


class TestRetrieveConfigResolution:
    def test_uses_config_top_k_when_top_k_arg_omitted(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        vector_store.query.return_value = {"documents": [[]]}
        bm25_store.query.return_value = []
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)

        retriever.retrieve("q", config=base_config(top_k=7))

        called_n_results = vector_store.query.call_args.kwargs["n_results"]
        assert called_n_results == max(7 * 6, 20)

    def test_explicit_top_k_overrides_config(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        vector_store.query.return_value = {"documents": [[]]}
        bm25_store.query.return_value = []
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)

        retriever.retrieve("q", top_k=2, config=base_config(top_k=7))

        called_n_results = vector_store.query.call_args.kwargs["n_results"]
        assert called_n_results == max(2 * 6, 20)

    def test_falls_back_to_global_settings_without_config(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        vector_store.query.return_value = {"documents": [[]]}
        bm25_store.query.return_value = []
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)

        # No config passed -> falls back to app.core.config.settings.
        results = retriever.retrieve("q")
        assert results == []

    def test_dense_only_does_not_query_bm25(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        meta = make_meta("docs/dense.md", 0)
        vector_store.query.return_value = {
            "documents": [["dense result"]],
            "metadatas": [[meta]],
            "distances": [[0.1]],
        }
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)

        results = retriever.retrieve("exact acronym CI", config=base_config(dense_weight=1.0, sparse_weight=0.0))

        assert results[0][0] == "dense result"
        assert results[0][2] == pytest.approx(1.0)
        bm25_store.query.assert_not_called()


class TestRerankingSwitch:
    def test_reranker_invoked_when_enabled(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        reranker = MagicMock()
        meta = make_meta("docs/a.md", 0)
        vector_store.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[meta]],
            "distances": [[0.1]],
        }
        bm25_store.query.return_value = []
        reranker.rerank.return_value = [("text", meta, 9.9)]

        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store, reranker=reranker)
        results = retriever.retrieve("q", config=base_config(reranker_on=True, rerank_top_n=3, top_k=3))

        reranker.rerank.assert_called_once()
        assert results == [("text", meta, 9.9)]

    def test_reranker_skipped_when_disabled(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        reranker = MagicMock()
        meta = make_meta("docs/a.md", 0)
        vector_store.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[meta]],
            "distances": [[0.1]],
        }
        bm25_store.query.return_value = []

        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store, reranker=reranker)
        retriever.retrieve("q", config=base_config(reranker_on=False, top_k=3))

        reranker.rerank.assert_not_called()

    def test_reranker_receives_explicit_candidate_pool(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        reranker = MagicMock()
        metas = [make_meta(f"docs/{letter}.md", 0) for letter in "abcd"]
        vector_store.query.return_value = {
            "documents": [["a", "b", "c", "d"]],
            "metadatas": [metas],
            "distances": [[0.1, 0.2, 0.3, 0.4]],
        }
        bm25_store.query.return_value = []
        reranker.rerank.return_value = [("a", metas[0], 1.0), ("b", metas[1], 0.9)]

        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store, reranker=reranker)
        retriever.retrieve(
            "q",
            config=base_config(
                top_k=2,
                reranker_on=True,
                rerank_candidate_pool=4,
                candidate_depth=4,
            ),
        )

        assert len(reranker.rerank.call_args.args[1]) == 4


class TestQueryRewritingExpansionSwitches:
    def test_rewriting_and_expansion_disabled_uses_single_original_query(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        vector_store.query.return_value = {"documents": [[]]}
        bm25_store.query.return_value = []
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)

        retriever.retrieve("original query", config=base_config(query_rewriting_on=False, query_expansion_on=False))

        assert vector_store.query.call_count == 1
        assert vector_store.query.call_args.kwargs["query_text"] == "original query"

    def test_expansion_enabled_queries_multiple_variants(self, monkeypatch):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        vector_store.query.return_value = {"documents": [[]]}
        bm25_store.query.return_value = []
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)
        retriever.rewriter.expand_query = MagicMock(return_value=["original query", "variant one", "variant two"])

        retriever.retrieve("original query", config=base_config(query_rewriting_on=False, query_expansion_on=True))

        assert vector_store.query.call_count == 3

    def test_rewriting_enabled_calls_rewriter(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        vector_store.query.return_value = {"documents": [[]]}
        bm25_store.query.return_value = []
        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)
        retriever.rewriter.rewrite_query = MagicMock(return_value="rewritten query")

        retriever.retrieve("original query", config=base_config(query_rewriting_on=True))

        retriever.rewriter.rewrite_query.assert_called_once()
        assert vector_store.query.call_args.kwargs["query_text"] == "rewritten query"


class TestAcronymBoost:
    def test_acronym_query_shifts_fusion_weight_toward_sparse(self):
        vector_store = MagicMock()
        bm25_store = MagicMock()
        meta = make_meta("docs/a.md", 0)
        vector_store.query.return_value = {
            "documents": [["some text about CI"]],
            "metadatas": [[meta]],
            "distances": [[0.0]],
        }
        bm25_store.query.return_value = [("some text about CI", meta, 1.0)]

        retriever = make_retriever(vector_store=vector_store, bm25_store=bm25_store)
        # dense-only weighting would give ~0.7 for a lone dense hit; the
        # acronym path re-weights to 0.35/0.65 before adding the sparse hit.
        results = retriever.retrieve("What is CI?", config=base_config(dense_weight=0.7, sparse_weight=0.3))

        assert len(results) == 1

    def test_glossary_hit_boosts_score_above_plain_normalized_range(self):
        retriever = make_retriever()
        meta_glossary = make_meta("docs/glossary.md", 0, file_name="glossary.md")
        combined = {("g", 0): ["CI stands for continuous integration", meta_glossary, 0.5]}

        retriever._apply_query_intent_boost("What is CI?", combined)

        # glossary file + acronym hit + definition-style query -> +2.0 boost.
        assert combined[("g", 0)][2] == pytest.approx(2.5)


class TestFormatContext:
    def test_formats_sources_with_index_and_chunk(self):
        retriever = make_retriever()
        meta = make_meta("docs/a.md", 2, file_name="a.md")
        context = retriever.format_context([("chunk text", meta, 0.9)])
        assert "[1]: a.md" in context
        assert "Chunk 2" in context
        assert "chunk text" in context
