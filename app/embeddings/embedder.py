from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.core.config import settings

def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Returns the configured sentence-transformers embedding function."""
    return SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
