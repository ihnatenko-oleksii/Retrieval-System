from unittest.mock import MagicMock

import pytest

from app.core.models import Chunk, ChunkMetadata
from app.vector_store.chroma_store import IncompatibleEmbeddingIndexError, VectorStore
from tests.conftest import make_chunk


def make_store(collection=None, embedding_function=None) -> VectorStore:
    """Bypass __init__ so no real Chroma client / embedding model is built."""
    store = VectorStore.__new__(VectorStore)
    store.collection = collection or MagicMock()
    store.embedding_function = embedding_function
    store.collection_name = "documents"
    return store


class TestAddChunks:
    def test_empty_chunks_is_a_noop(self):
        collection = MagicMock()
        store = make_store(collection=collection)

        store.add_chunks([])

        collection.upsert.assert_not_called()

    def test_none_metadata_values_are_stripped(self):
        # ChromaDB rejects None in metadata values, so optional fields left
        # unset (page_number here, via the model's extra="allow" config)
        # must be dropped before upsert.
        collection = MagicMock()
        store = make_store(collection=collection)
        chunk = Chunk(
            document_id="doc-1",
            content="content",
            metadata=ChunkMetadata(
                document_id="doc-1",
                file_path="a.md",
                file_name="a.md",
                extension=".md",
                loader_type="TextLoader",
                chunk_index=0,
                page_number=None,
            ),
        )

        store.add_chunks([chunk])

        _, kwargs = collection.upsert.call_args
        assert "page_number" not in kwargs["metadatas"][0]
        assert None not in kwargs["metadatas"][0].values()

    def test_upsert_receives_matching_ids_documents_and_metadata(self):
        collection = MagicMock()
        store = make_store(collection=collection)
        chunk_a = make_chunk("first", file_path="a.md", chunk_index=0)
        chunk_b = make_chunk("second", file_path="b.md", chunk_index=1)

        store.add_chunks([chunk_a, chunk_b])

        _, kwargs = collection.upsert.call_args
        assert kwargs["ids"] == [chunk_a.id, chunk_b.id]
        assert kwargs["documents"] == ["first", "second"]
        assert len(kwargs["metadatas"]) == 2


class TestQuery:
    def test_uses_precomputed_query_embedding_when_available(self):
        collection = MagicMock()
        embedding_function = MagicMock()
        embedding_function.embed_query.return_value = [0.1, 0.2, 0.3]
        store = make_store(collection=collection, embedding_function=embedding_function)

        store.query("hello", n_results=5)

        embedding_function.embed_query.assert_called_once_with("hello")
        _, kwargs = collection.query.call_args
        assert kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]
        assert kwargs["n_results"] == 5

    def test_falls_back_to_query_texts_without_embed_query(self):
        collection = MagicMock()
        embedding_function = object()  # no embed_query attribute
        store = make_store(collection=collection, embedding_function=embedding_function)

        store.query("hello", n_results=5)

        _, kwargs = collection.query.call_args
        assert kwargs["query_texts"] == ["hello"]

    def test_defaults_n_results_to_settings_top_k(self, monkeypatch):
        collection = MagicMock()
        embedding_function = object()
        store = make_store(collection=collection, embedding_function=embedding_function)
        monkeypatch.setattr("app.vector_store.chroma_store.settings.retrieval_top_k", 7)

        store.query("hello")

        _, kwargs = collection.query.call_args
        assert kwargs["n_results"] == 7


def make_compatibility_store(collection, *, model="Qwen/Qwen3-Embedding-0.6B", dimension=1024) -> VectorStore:
    store = make_store(collection=collection)
    store.embedding_model = model
    store.embedding_function = MagicMock()
    store.embedding_function.embedding_dimension.return_value = dimension
    return store


def test_rejects_existing_index_with_a_different_embedding_dimension():
    collection = MagicMock()
    collection.metadata = {}
    collection.count.return_value = 1
    collection.peek.return_value = {"embeddings": [[0.0] * 768]}
    store = make_compatibility_store(collection, dimension=1024)

    with pytest.raises(IncompatibleEmbeddingIndexError, match="Re-ingest the corpus"):
        store._validate_existing_index()

    collection.query.assert_not_called()


def test_accepts_existing_unmarked_index_when_configured_model_dimension_matches():
    collection = MagicMock()
    collection.metadata = {}
    collection.count.return_value = 1
    collection.peek.return_value = {"embeddings": [[0.0] * 768]}
    store = make_compatibility_store(collection, model="intfloat/multilingual-e5-base", dimension=768)

    store._validate_existing_index()
