from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_QWEN3_QUERY_INSTRUCTION = (
    "Given a technical support or engineering question, retrieve passages that directly answer the question "
    "or contain the necessary implementation details."
)


class Settings(BaseSettings):
    # Vector Store
    vector_db_path: str = "./storage/chroma"

    # Validated retrieval default. Changing the embedding model requires a
    # complete re-ingestion because stored vectors are model-specific.
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    # Qwen3 applies this instruction to queries only; document embeddings stay
    # unprompted. Keep this exact text aligned with the validated benchmark.
    embedding_query_instruction: str | None = DEFAULT_QWEN3_QUERY_INSTRUCTION

    # LLM Model
    llm_model: str = "qwen3.5:4b-mlx"

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval
    retrieval_top_k: int = 3

    # Optional reranking experiment (disabled by default).
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 3
    rerank_candidate_pool: int = 20
    reranker_on: bool = False

    # Optional query expansion/rewriting experiments (disabled by default).
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

    # Optional experimental retrieval components (disabled by default).
    prf_on: bool = False
    prf_depth: int = 1
    prf_min_confidence: float = 0.35
    prf_max_terms: int = 8
    prf_weight: float = 0.35
    native_bge_on: bool = False
    native_bge_dense_weight: float = 0.4
    native_bge_sparse_weight: float = 0.3
    native_bge_colbert_weight: float = 0.3
    ltr_on: bool = False
    ltr_model: str = "auto"
    ltr_candidate_depth: int = 50
    diversity_on: bool = False
    diversity_relevance_weight: float = 0.7
    lexical_overlap_weight: float = 0.0

    # Validated hybrid retrieval defaults.
    hybrid_search_weights_dense: float = 0.7
    hybrid_search_weights_sparse: float = 0.3
    retrieval_candidate_depth: int = 20
    fusion_strategy: str = "weighted_linear"
    # The validated winner uses fixed 0.7/0.3 fusion. Query-adaptive weights
    # remain available as an explicit experiment, but are not the default.
    adaptive_routing: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
