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
    rerank_candidate_pool: int = 20
    reranker_on: bool = False

    # Query Expansion/Rewriting (Phase 2)
    query_rewriting_on: bool = False
    query_expansion_on: bool = False
    rewrite_policy: str = "always"
    expansion_policy: str = "always"
    include_original_query: bool = False
    multi_query_fusion_strategy: str = "weighted_rrf"
    original_query_weight: float = 1.0
    rewrite_query_weight: float = 0.7
    expansion_query_weight: float = 0.5
    confidence_routing: bool = False
    confidence_threshold: float = 0.35

    # Hybrid Search (Phase 2)
    hybrid_search_weights_dense: float = 0.7
    hybrid_search_weights_sparse: float = 0.3
    retrieval_candidate_depth: int = 20
    fusion_strategy: str = "weighted_linear"
    adaptive_routing: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
