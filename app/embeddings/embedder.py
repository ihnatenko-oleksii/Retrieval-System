from collections.abc import Sequence

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.core.config import settings


def _is_e5_model(model_name: str) -> bool:
    """E5-family models require 'passage: '/'query: ' prefixes."""
    name = (model_name or "").lower()
    return name.startswith("intfloat/multilingual-e5") or name.startswith("intfloat/e5-")


class PrefixedSentenceTransformerEmbeddingFunction:
    """
    Chroma-compatible embedding function that transparently applies
    model-specific prefixes. E5 models need 'passage: ' for documents and
    'query: ' for queries; skipping these degrades retrieval quality noticeably.

    For the query path, use `embed_query(text)` to get a vector that was
    encoded with the 'query: ' prefix; pass it to Chroma via
    `query_embeddings=[...]` to avoid double-prefixing.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._inner = SentenceTransformerEmbeddingFunction(model_name=model_name)
        self._needs_e5_prefix = _is_e5_model(model_name)

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        prepared = [f"passage: {text}" for text in input] if self._needs_e5_prefix else list(input)
        return self._inner(prepared)

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string with the model-specific query prefix."""
        prepared = f"query: {text}" if self._needs_e5_prefix else text
        vectors = self._inner([prepared])
        # Chroma `query_embeddings` expects plain Python numeric types.
        # SentenceTransformers may return numpy scalars (np.float32), so cast.
        return [float(x) for x in vectors[0]]

    def name(self) -> str:
        return f"prefixed::{self.model_name}"


def get_embedding_function() -> PrefixedSentenceTransformerEmbeddingFunction:
    """Returns the configured embedding function with model-aware prefixing."""
    return PrefixedSentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
