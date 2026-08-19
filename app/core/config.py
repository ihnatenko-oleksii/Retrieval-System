from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Vector Store
    vector_db_path: str = "./storage/chroma"

    # Embedding Model
    # Default: multilingual E5 base (~280 MB, 768-d) - strong Polish+English,
    # good quality/speed tradeoff on Apple Silicon.
    # Alternatives:
    #   "intfloat/multilingual-e5-small" -> fastest, lighter (~118 MB, 384-d)
    #   "BAAI/bge-m3"                    -> best multilingual quality, heavier (~2.3 GB)
    # IMPORTANT: after changing this value you MUST re-ingest your corpus,
    # otherwise stored embeddings won't match the new model.
    embedding_model: str = "intfloat/multilingual-e5-base"

    # LLM Model
    llm_model: str = "qwen3.5:4b-mlx"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retrieval_top_k: int = 3

    # Reranking (Phase 2)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 3
    reranker_on: bool = False

    # Query Expansion/Rewriting (Phase 2)
    query_rewriting_on: bool = False
    query_expansion_on: bool = False

    # Hybrid Search (Phase 2)
    hybrid_search_weights_dense: float = 0.7
    hybrid_search_weights_sparse: float = 0.3

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
