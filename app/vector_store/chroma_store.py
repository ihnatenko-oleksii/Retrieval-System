import logging

import chromadb
from chromadb.config import Settings

from app.core.config import settings
from app.core.models import Chunk
from app.embeddings.embedder import get_embedding_function

logger = logging.getLogger(__name__)


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
        effective_query_instruction = (
            settings.embedding_query_instruction if query_instruction is None else query_instruction
        )
        self.embedding_function = get_embedding_function(
            embedding_model,
            query_instruction=effective_query_instruction,
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, embedding_function=self.embedding_function, metadata={"hnsw:space": "cosine"}
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
