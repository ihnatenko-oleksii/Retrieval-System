import chromadb
from chromadb.config import Settings
from typing import List
from app.core.config import settings
from app.core.models import Chunk
from app.embeddings.embedder import get_embedding_function
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.vector_db_path,
            settings=Settings(allow_reset=True)
        )
        self.embedding_function = get_embedding_function()
        self.collection_name = "documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk]):
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

        batch_size = 5461 # ChromaDB recommended batch size
        for i in range(0, len(ids), batch_size):
            try:
                self.collection.upsert(
                    ids=ids[i:i+batch_size],
                    documents=texts[i:i+batch_size],
                    metadatas=clean_metadatas[i:i+batch_size]
                )
                logger.info(f"Inserted batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")

    def query(self, query_text: str, n_results: int = None) -> dict:
        if n_results is None:
            n_results = settings.retrieval_top_k
            
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
