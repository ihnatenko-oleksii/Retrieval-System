import app.embeddings.embedder as embedder_module
from app.embeddings.embedder import (
    DEFAULT_QWEN3_QUERY_INSTRUCTION,
    PrefixedSentenceTransformerEmbeddingFunction,
    _is_e5_model,
    _is_qwen3_model,
    get_embedding_function,
)


class FakeInnerEmbeddingFunction:
    """Stand-in for chromadb's SentenceTransformerEmbeddingFunction (no model download)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


class TestIsE5Model:
    def test_multilingual_e5_variants_are_detected(self):
        assert _is_e5_model("intfloat/multilingual-e5-base") is True
        assert _is_e5_model("intfloat/multilingual-e5-small") is True

    def test_plain_e5_variants_are_detected(self):
        assert _is_e5_model("intfloat/e5-large-v2") is True

    def test_non_e5_model_is_not_detected(self):
        assert _is_e5_model("BAAI/bge-m3") is False
        assert _is_e5_model("sentence-transformers/all-MiniLM-L6-v2") is False

    def test_none_or_empty_is_safe(self):
        assert _is_e5_model(None) is False
        assert _is_e5_model("") is False


def test_qwen3_models_are_detected_for_instruction_aware_queries():
    assert _is_qwen3_model("Qwen/Qwen3-Embedding-0.6B") is True
    assert _is_qwen3_model("BAAI/bge-m3") is False


class TestPrefixing:
    def _make(self, monkeypatch, model_name):
        monkeypatch.setattr(embedder_module, "SentenceTransformerEmbeddingFunction", FakeInnerEmbeddingFunction)
        return PrefixedSentenceTransformerEmbeddingFunction(model_name)

    def test_e5_model_prefixes_documents_with_passage(self, monkeypatch):
        ef = self._make(monkeypatch, "intfloat/multilingual-e5-base")

        ef(["hello", "world"])

        assert ef._inner.calls == [["passage: hello", "passage: world"]]

    def test_non_e5_model_does_not_prefix_documents(self, monkeypatch):
        ef = self._make(monkeypatch, "BAAI/bge-m3")

        ef(["hello", "world"])

        assert ef._inner.calls == [["hello", "world"]]

    def test_e5_query_gets_query_prefix(self, monkeypatch):
        ef = self._make(monkeypatch, "intfloat/multilingual-e5-base")

        ef.embed_query("search text")

        assert ef._inner.calls == [["query: search text"]]

    def test_non_e5_query_has_no_prefix(self, monkeypatch):
        ef = self._make(monkeypatch, "BAAI/bge-m3")

        ef.embed_query("search text")

        assert ef._inner.calls == [["search text"]]

    def test_embed_query_returns_plain_python_floats(self, monkeypatch):
        ef = self._make(monkeypatch, "BAAI/bge-m3")

        vector = ef.embed_query("abc")

        assert vector == [3.0]
        assert all(isinstance(x, float) for x in vector)

    def test_name_includes_model_name(self, monkeypatch):
        ef = self._make(monkeypatch, "BAAI/bge-m3")
        assert ef.name() == "prefixed::BAAI/bge-m3"

    def test_qwen3_instruction_is_query_only(self, monkeypatch):
        ef = self._make(monkeypatch, "Qwen/Qwen3-Embedding-0.6B")

        ef(["document passage"])
        ef.embed_query("technical question")

        assert ef._inner.calls[0] == ["document passage"]
        assert ef._inner.calls[1] == [
            f"Instruct: {DEFAULT_QWEN3_QUERY_INSTRUCTION}\nQuery: technical question"
        ]


def test_get_embedding_function_uses_configured_model(monkeypatch):
    monkeypatch.setattr(embedder_module, "SentenceTransformerEmbeddingFunction", FakeInnerEmbeddingFunction)
    monkeypatch.setattr(embedder_module.settings, "embedding_model", "BAAI/bge-m3")

    ef = get_embedding_function()

    assert ef.model_name == "BAAI/bge-m3"
