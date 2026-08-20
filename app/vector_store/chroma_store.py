import logging

import chromadb
from chromadb.config import Settings

from app.core.config import settings
from app.core.models import Chunk
from app.embeddings.embedder import get_embedding_function

logger = logging.getLogger(__name__)


class IncompatibleEmbeddingIndexError(RuntimeError):
    """Raised before querying vectors created by a different embedding model."""


class VectorStore:
    def __init__(
        self,
        persist_dir: str | None = None,
        embedding_model: str | None = None,
        query_instruction: str | None = None,
        collection_name: str = "documents",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir or settings.vector_db_path,
            settings=Settings(allow_reset=True),
        )
        self.embedding_model = embedding_model or settings.embedding_model
        effective_query_instruction = (
            settings.embedding_query_instruction if query_instruction is None else query_instruction
        )
        self.embedding_function = get_embedding_function(
            self.embedding_model,
            query_instruction=effective_query_instruction,
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model},
        )
        self._validate_existing_index()

    def _configured_embedding_dimension(self) -> int | None:
        dimension_getter = getattr(self.embedding_function, "embedding_dimension", None)
        if callable(dimension_getter):
            dimension = dimension_getter()
            if dimension is not None:
                return int(dimension)

        # The embedding model is already loaded by VectorStore construction.
        # Encoding an empty probe does not query or mutate the persisted index.
        embed_query = getattr(self.embedding_function, "embed_query", None)
        if callable(embed_query):
            return len(embed_query(""))
        return None

    def _stored_embedding_dimension(self) -> int | None:
        if self.collection.count() == 0:
            return None
        try:
            embeddings = self.collection.peek(limit=1).get("embeddings")
        except Exception as exc:
            raise IncompatibleEmbeddingIndexError(
                "Existing index could not be inspected safely. Re-ingest the corpus or use the original model; "
                "the persisted vectors were not queried or modified."
            ) from exc
        if embeddings is None or len(embeddings) == 0:
            raise IncompatibleEmbeddingIndexError(
                "Existing index has data but no inspectable embedding dimension. Re-ingest the corpus or use the "
                "original model; the persisted vectors were not queried or modified."
            )
        return len(embeddings[0])

    def _validate_existing_index(self) -> None:
        """Fail closed when persisted vectors cannot be verified for this model."""
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != self.embedding_model:
            raise IncompatibleEmbeddingIndexError(
                f"Existing index was created with embedding model '{stored_model}', but this process is configured "
                f"for '{self.embedding_model}'. Re-ingest the corpus or set EMBEDDING_MODEL to the original model."
            )

        stored_dimension = self._stored_embedding_dimension()
        if stored_dimension is None:
            return

        configured_dimension = self._configured_embedding_dimension()
        if configured_dimension is None:
            raise IncompatibleEmbeddingIndexError(
                f"Existing index has {stored_dimension}-dimensional vectors, but the configured model "
                f"'{self.embedding_model}' could not be verified. Re-ingest the corpus or use the original model."
            )
        if stored_dimension != configured_dimension:
            raise IncompatibleEmbeddingIndexError(
                f"Existing index was created with a different embedding model (stored dimension "
                f"{stored_dimension}, configured '{self.embedding_model}' dimension {configured_dimension}). "
                "Re-ingest the corpus or use the original model. The persisted vectors were not queried or modified."
            )

        if not stored_model:
            logger.warning(
                "Existing embedding index has no model metadata; dimension %s matches configured model '%s'.",
                stored_dimension,
                self.embedding_model,
            )

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        ids = [chunk.id for chunk in chunks]
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata.model_dump() for chunk in chunks]

        # Filter out None values from metadatas (ChromaDB does not support None in metadata)
        clean_metadatas = []
        for metadata in metadatas:
            clean_metadata = {k: v for k, v in metadata.items() if v is not None}
            clean_metadatas.append(clean_metadata)

        batch_size = 5461  # ChromaDB recommended batch size
        for i in range(0, len(ids), batch_size):
            try:
                self.collection.upsert(
                    ids=ids[i : i + batch_size],
                    documents=texts[i : i + batch_size],
                    metadatas=clean_metadatas[i : i + batch_size],
                )
                logger.info(f"Inserted batch {i // batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to insert batch {i // batch_size + 1}: {e}")

    def query(self, query_text: str, n_results: int = None) -> dict:
        if n_results is None:
            n_results = settings.retrieval_top_k

        # Use pre-computed query embedding so model-specific query prefixes
        # (e.g. E5 'query: ') are applied instead of the document 'passage: ' one.
        embed_query = getattr(self.embedding_function, "embed_query", None)
        if callable(embed_query):
            query_vector = embed_query(query_text)
            return self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
            )

        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
