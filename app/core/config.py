from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    # Vector Store
    vector_db_path: str = "./storage/chroma"

    # Embedding Model
    embedding_model: str = "all-MiniLM-L6-v2"

    # LLM Model
    llm_model: str = "llama3.2"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retrieval_top_k: int = 3
    
    # Reranking (Phase 2)
    rerank_top_n: int = 3
    reranker_on: bool = False
    
    # Query Expansion/Rewriting (Phase 2)
    query_rewriting_on: bool = False
    query_expansion_on: bool = False

    # Hybrid Search (Phase 2)
    hybrid_search_weights_dense: float = 0.7
    hybrid_search_weights_sparse: float = 0.3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
