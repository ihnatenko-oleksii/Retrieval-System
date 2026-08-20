from dataclasses import dataclass

from app.core.config import DEFAULT_QWEN3_QUERY_INSTRUCTION, settings


@dataclass
class RetrievalConfig:
    """
    Per-request / per-trial runtime knobs for the retrieval + generation
    pipeline. Lets the UI and the tuning sweep override what would otherwise
    be read from the global `settings` singleton.
    """

    top_k: int
    dense_weight: float
    sparse_weight: float
    reranker_on: bool
    rerank_top_n: int
    query_rewriting_on: bool
    query_expansion_on: bool
    llm_model: str | None = None
    fusion_strategy: str = "weighted_linear"
    rewrite_policy: str = "always"
    expansion_policy: str = "always"
    include_original_query: bool = False
    multi_query_fusion_strategy: str = "weighted_rrf"
    original_query_weight: float = 1.0
    rewrite_query_weight: float = 0.7
    expansion_query_weight: float = 0.5
    confidence_routing: bool = False
    confidence_threshold: float = 0.35
    candidate_depth: int | None = None
    rerank_candidate_pool: int | None = None
    adaptive_routing: bool = False
    rrf_k: int = 60
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
    embedding_query_instruction: str | None = DEFAULT_QWEN3_QUERY_INSTRUCTION
    diversity_on: bool = False
    diversity_relevance_weight: float = 0.7
    lexical_overlap_weight: float = 0.0
    embedding_model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.candidate_depth is not None and self.candidate_depth < 1:
            raise ValueError("candidate_depth must be at least 1")
        if self.rerank_candidate_pool is not None and self.rerank_candidate_pool < 1:
            raise ValueError("rerank_candidate_pool must be at least 1 when provided")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be at least 1")
        if self.prf_depth < 1:
            raise ValueError("prf_depth must be at least 1")
        if self.prf_max_terms < 1:
            raise ValueError("prf_max_terms must be at least 1")
        if self.ltr_candidate_depth < self.top_k:
            raise ValueError("ltr_candidate_depth must be at least top_k")
        if not 0.0 <= self.diversity_relevance_weight <= 1.0:
            raise ValueError("diversity_relevance_weight must be between 0 and 1")
        if self.lexical_overlap_weight < 0:
            raise ValueError("lexical_overlap_weight cannot be negative")
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if any(
            weight < 0
            for weight in (
                self.prf_weight,
                self.native_bge_dense_weight,
                self.native_bge_sparse_weight,
                self.native_bge_colbert_weight,
            )
        ):
            raise ValueError("retrieval component weights cannot be negative")
        if self.fusion_strategy not in {"weighted_linear", "rrf", "weighted_rrf"}:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")
        if self.rewrite_policy not in {"never", "always", "selective"}:
            raise ValueError(f"Unsupported rewrite policy: {self.rewrite_policy}")
        if self.expansion_policy not in {"never", "always", "selective"}:
            raise ValueError(f"Unsupported expansion policy: {self.expansion_policy}")
        if self.multi_query_fusion_strategy not in {"weighted_linear", "rrf", "weighted_rrf"}:
            raise ValueError(
                f"Unsupported multi-query fusion strategy: {self.multi_query_fusion_strategy}"
            )
        if any(
            weight < 0
            for weight in (
                self.original_query_weight,
                self.rewrite_query_weight,
                self.expansion_query_weight,
            )
        ):
            raise ValueError("query variant weights cannot be negative")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

    @classmethod
    def from_settings(cls) -> "RetrievalConfig":
        return cls(
            top_k=settings.retrieval_top_k,
            dense_weight=settings.hybrid_search_weights_dense,
            sparse_weight=settings.hybrid_search_weights_sparse,
            reranker_on=settings.reranker_on,
            rerank_top_n=settings.rerank_top_n,
            query_rewriting_on=settings.query_rewriting_on,
            query_expansion_on=settings.query_expansion_on,
            llm_model=settings.llm_model,
            fusion_strategy=settings.fusion_strategy,
            rewrite_policy=settings.rewrite_policy,
            expansion_policy=settings.expansion_policy,
            include_original_query=settings.include_original_query,
            multi_query_fusion_strategy=settings.multi_query_fusion_strategy,
            original_query_weight=settings.original_query_weight,
            rewrite_query_weight=settings.rewrite_query_weight,
            expansion_query_weight=settings.expansion_query_weight,
            confidence_routing=settings.confidence_routing,
            confidence_threshold=settings.confidence_threshold,
            candidate_depth=settings.retrieval_candidate_depth,
            rerank_candidate_pool=settings.rerank_candidate_pool,
            adaptive_routing=settings.adaptive_routing,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            prf_on=settings.prf_on,
            prf_depth=settings.prf_depth,
            prf_min_confidence=settings.prf_min_confidence,
            prf_max_terms=settings.prf_max_terms,
            prf_weight=settings.prf_weight,
            native_bge_on=settings.native_bge_on,
            native_bge_dense_weight=settings.native_bge_dense_weight,
            native_bge_sparse_weight=settings.native_bge_sparse_weight,
            native_bge_colbert_weight=settings.native_bge_colbert_weight,
            ltr_on=settings.ltr_on,
            ltr_model=settings.ltr_model,
            ltr_candidate_depth=settings.ltr_candidate_depth,
            embedding_query_instruction=settings.embedding_query_instruction,
            diversity_on=settings.diversity_on,
            diversity_relevance_weight=settings.diversity_relevance_weight,
            lexical_overlap_weight=settings.lexical_overlap_weight,
        )
