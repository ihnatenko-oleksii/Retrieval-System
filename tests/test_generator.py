from app.generation.generator import Generator
from tests.conftest import make_meta


def make_generator() -> Generator:
    return Generator(model_name="test-model")


class TestNoContext:
    def test_empty_chunks_short_circuits_without_calling_llm(self, monkeypatch):
        calls = []
        monkeypatch.setattr("app.generation.generator.ollama.chat", lambda **kw: calls.append(kw))
        generator = make_generator()

        result = generator.generate_answer("question", [])

        assert result["final_answer"] == "No relevant context found to answer the question."
        assert result["retrieved_sources"] == []
        assert calls == []


class TestSourceCitations:
    def test_sources_are_numbered_in_retrieval_order(self, monkeypatch):
        monkeypatch.setattr(
            "app.generation.generator.ollama.chat", lambda **kw: {"message": {"content": "answer [1][2]"}}
        )
        generator = make_generator()
        chunks = [
            ("first chunk", make_meta("a.md", 0, file_name="a.md"), 0.9),
            ("second chunk", make_meta("b.md", 3, file_name="b.md"), 0.5),
        ]

        result = generator.generate_answer("question", chunks)

        sources = result["retrieved_sources"]
        assert sources[0] == {"source_id": 1, "file_name": "a.md", "distance": 0.9, "chunk_index": 0}
        assert sources[1] == {"source_id": 2, "file_name": "b.md", "distance": 0.5, "chunk_index": 3}

    def test_prompt_includes_source_markers_for_every_chunk(self, monkeypatch):
        captured = {}

        def fake_chat(**kw):
            captured.update(kw)
            return {"message": {"content": "answer"}}

        monkeypatch.setattr("app.generation.generator.ollama.chat", fake_chat)
        generator = make_generator()
        chunks = [
            ("first chunk text", make_meta("a.md", 0, file_name="a.md"), 0.9),
            ("second chunk text", make_meta("b.md", 0, file_name="b.md"), 0.5),
        ]

        generator.generate_answer("question", chunks)

        prompt = captured["messages"][-1]["content"]
        assert "Source [1]: a.md" in prompt
        assert "Source [2]: b.md" in prompt
        assert "first chunk text" in prompt
        assert "second chunk text" in prompt

    def test_missing_file_name_falls_back_to_unknown_source(self, monkeypatch):
        monkeypatch.setattr("app.generation.generator.ollama.chat", lambda **kw: {"message": {"content": "answer"}})
        generator = make_generator()

        result = generator.generate_answer("question", [("text", {}, 0.5)])

        assert result["retrieved_sources"][0]["file_name"] == "Unknown Source"


class TestLlmFailureBehavior:
    def test_llm_error_is_surfaced_in_answer_not_raised(self, monkeypatch):
        def failing_chat(**kw):
            raise ConnectionError("ollama not running")

        monkeypatch.setattr("app.generation.generator.ollama.chat", failing_chat)
        generator = make_generator()

        result = generator.generate_answer("question", [("text", make_meta("a.md", 0), 0.5)])

        assert "Error generating answer" in result["final_answer"]
        # Sources are still reported even though generation failed.
        assert result["retrieved_sources"][0]["file_name"] == "a.md"


class TestChatHistory:
    def test_history_is_capped_to_last_four_messages(self, monkeypatch):
        captured = {}

        def fake_chat(**kw):
            captured.update(kw)
            return {"message": {"content": "answer"}}

        monkeypatch.setattr("app.generation.generator.ollama.chat", fake_chat)
        generator = make_generator()
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]

        generator.generate_answer("question", [("text", make_meta("a.md", 0), 0.5)], chat_history=history)

        # 4 history messages + the final user prompt message.
        assert len(captured["messages"]) == 5
        assert captured["messages"][0]["content"] == "msg 6"

    def test_non_string_history_content_is_normalized(self, monkeypatch):
        captured = {}

        def fake_chat(**kw):
            captured.update(kw)
            return {"message": {"content": "answer"}}

        monkeypatch.setattr("app.generation.generator.ollama.chat", fake_chat)
        generator = make_generator()
        history = [{"role": "user", "content": {"text": "structured content"}}]

        generator.generate_answer("question", [("text", make_meta("a.md", 0), 0.5)], chat_history=history)

        assert captured["messages"][0]["content"] == "structured content"


class TestSafeText:
    def test_none_becomes_empty_string(self):
        assert make_generator()._safe_text(None) == ""

    def test_list_of_parts_is_joined(self):
        assert make_generator()._safe_text(["hello", {"text": "world"}]) == "hello world"

    def test_dict_without_text_key_is_stringified(self):
        assert make_generator()._safe_text({"foo": "bar"}) == "{'foo': 'bar'}"
