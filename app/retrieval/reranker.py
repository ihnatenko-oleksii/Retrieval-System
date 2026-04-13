from sentence_transformers import CrossEncoder
from typing import List, Tuple
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        if settings.reranker_on:
            try:
                self.model = CrossEncoder(self.model_name, max_length=512)
                logger.info(f"Loaded CrossEncoder reranker: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker {self.model_name}: {e}")

    def rerank(self, query: str, retrieved_chunks: List[Tuple[str, dict, float]], top_n: int = None) -> List[Tuple[str, dict, float]]:
        if not settings.reranker_on or self.model is None or not retrieved_chunks:
            return retrieved_chunks[:top_n] if top_n else retrieved_chunks

        if top_n is None:
            top_n = settings.rerank_top_n

        contents = [chunk[0] for chunk in retrieved_chunks]
        query_document_pairs = [[query, content] for content in contents]
        
        try:
            scores = self.model.predict(query_document_pairs)
            
            # Combine scores with original chunks
            scored_chunks = []
            for i, chunk in enumerate(retrieved_chunks):
                content, meta, _ = chunk
                scored_chunks.append((content, meta, float(scores[i])))
                
            # Sort by new cross-encoder scores descending
            reranked = sorted(scored_chunks, key=lambda x: x[2], reverse=True)
            return reranked[:top_n]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return retrieved_chunks[:top_n] if top_n else retrieved_chunks
