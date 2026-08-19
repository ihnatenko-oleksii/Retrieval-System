from app.retrieval.rewriter import QueryRewriter


def make_rewriter() -> QueryRewriter:
    return QueryRewriter(model_name="test-model")


class TestRewriteQuery:
    def test_disabled_without_history_returns_original(self, monkeypatch):
        rewriter = make_rewriter()
        calls = []
        monkeypatch.setattr("app.retrieval.rewriter.ollama.chat", lambda **kw: calls.append(kw))

        result = rewriter.rewrite_query("original query", enabled=False)

        assert result == "original query"
        assert calls == []

    def test_enabled_calls_llm_and_strips_quotes(self, monkeypatch):
        rewriter = make_rewriter()
        monkeypatch.setattr(
            "app.retrieval.rewriter.ollama.chat",
            lambda **kw: {"message": {"content": '"a clearer query"'}},
        )

        result = rewriter.rewrite_query("orig", enabled=True)

        assert result == "a clearer query"

    def test_disabled_but_with_history_still_rewrites(self, monkeypatch):
        # Follow-up questions must still be made standalone even when the
        # generic "rewriting" toggle is off.
        rewriter = make_rewriter()
        captured = {}

        def fake_chat(**kw):
            captured.update(kw)
            return {"message": {"content": "standalone version"}}

        monkeypatch.setattr("app.retrieval.rewriter.ollama.chat", fake_chat)
        history = [{"role": "user", "content": "What is RAG?"}, {"role": "assistant", "content": "..."}]

        result = rewriter.rewrite_query("What about hybrid search?", chat_history=history, enabled=False)

        assert result == "standalone version"
        assert "What is RAG?" in captured["messages"][0]["content"]

    def test_llm_failure_falls_back_to_original_query(self, monkeypatch):
        rewriter = make_rewriter()

        def failing_chat(**kw):
            raise ConnectionError("ollama not running")

        monkeypatch.setattr("app.retrieval.rewriter.ollama.chat", failing_chat)

        result = rewriter.rewrite_query("original query", enabled=True)

        assert result == "original query"

    def test_empty_llm_response_falls_back_to_original_query(self, monkeypatch):
        rewriter = make_rewriter()
        monkeypatch.setattr("app.retrieval.rewriter.ollama.chat", lambda **kw: {"message": {"content": "   "}})

        result = rewriter.rewrite_query("original query", enabled=True)

        assert result == "original query"


class TestExpandQuery:
    def test_disabled_returns_only_original_query(self, monkeypatch):
        rewriter = make_rewriter()
        calls = []
        monkeypatch.setattr("app.retrieval.rewriter.ollama.chat", lambda **kw: calls.append(kw))

        result = rewriter.expand_query("original query", enabled=False)

        assert result == ["original query"]
        assert calls == []

    def test_enabled_returns_original_plus_expansions(self, monkeypatch):
        rewriter = make_rewriter()
        monkeypatch.setattr(
            "app.retrieval.rewriter.ollama.chat",
            lambda **kw: {"message": {"content": "variant one\nvariant two"}},
        )

        result = rewriter.expand_query("original query", enabled=True)

        assert result == ["original query", "variant one", "variant two"]

    def test_respects_num_expansions_cap(self, monkeypatch):
        rewriter = make_rewriter()
        monkeypatch.setattr(
            "app.retrieval.rewriter.ollama.chat",
            lambda **kw: {"message": {"content": "one\ntwo\nthree\nfour"}},
        )

        result = rewriter.expand_query("q", num_expansions=2, enabled=True)

        assert result == ["q", "one", "two"]

    def test_llm_failure_falls_back_to_original_only(self, monkeypatch):
        rewriter = make_rewriter()

        def failing_chat(**kw):
            raise ConnectionError("ollama not running")

        monkeypatch.setattr("app.retrieval.rewriter.ollama.chat", failing_chat)

        result = rewriter.expand_query("original query", enabled=True)

        assert result == ["original query"]
