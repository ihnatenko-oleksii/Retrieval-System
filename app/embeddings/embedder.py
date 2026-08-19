from collections.abc import Sequence

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.core.config import settings

DEFAULT_QWEN3_QUERY_INSTRUCTION = (
    "Given a technical support or engineering question, retrieve passages that directly answer the question "
    "or contain the necessary implementation details."
)


def _is_e5_model(model_name: str | None) -> bool:
    """E5-family models require 'passage: '/'query: ' prefixes."""
    name = (model_name or "").lower()
    return name.startswith("intfloat/multilingual-e5") or name.startswith("intfloat/e5-")


def _is_qwen3_model(model_name: str | None) -> bool:
    name = (model_name or "").lower()
    return name.startswith("qwen/qwen3-embedding")


class PrefixedSentenceTransformerEmbeddingFunction:
    """
    Chroma-compatible embedding function that transparently applies
    model-specific prefixes. E5 models need 'passage: ' for documents and
    'query: ' for queries; skipping these degrades retrieval quality noticeably.

    For the query path, use `embed_query(text)` to get a vector that was
    encoded with the 'query: ' prefix; pass it to Chroma via
    `query_embeddings=[...]` to avoid double-prefixing.
    """

    def __init__(self, model_name: str, query_instruction: str | None = None):
        self.model_name = model_name
        self._inner = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._needs_e5_prefix = _is_e5_model(model_name)
        self._needs_qwen3_instruction = _is_qwen3_model(model_name)
        self.query_instruction = (
            DEFAULT_QWEN3_QUERY_INSTRUCTION if query_instruction is None else query_instruction
        )

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        prepared = [f"passage: {text}" for text in input] if self._needs_e5_prefix else list(input)
        return self._inner(prepared)

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string with the model-specific query prefix."""
        if self._needs_e5_prefix:
            prepared = f"query: {text}"
        elif self._needs_qwen3_instruction and self.query_instruction:
            prepared = f"Instruct: {self.query_instruction}\nQuery: {text}"
        else:
            prepared = text
        vectors = self._inner([prepared])
        # Chroma `query_embeddings` expects plain Python numeric types.
        # SentenceTransformers may return numpy scalars (np.float32), so cast.
        return [float(x) for x in vectors[0]]

    def name(self) -> str:
        return f"prefixed::{self.model_name}"


def get_embedding_function(
    model_name: str | None = None, query_instruction: str | None = None
) -> PrefixedSentenceTransformerEmbeddingFunction:
    """Returns the configured embedding function with model-aware prefixing."""
    return PrefixedSentenceTransformerEmbeddingFunction(
        model_name=model_name or settings.embedding_model,
        query_instruction=query_instruction,
    )
