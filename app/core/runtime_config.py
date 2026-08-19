from dataclasses import dataclass

from app.core.config import settings


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
    candidate_depth: int | None = None
    rerank_candidate_pool: int | None = None
    adaptive_routing: bool = True
    rrf_k: int = 60
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
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("retrieval weights cannot be negative")
        if self.fusion_strategy not in {"weighted_linear", "rrf", "weighted_rrf"}:
            raise ValueError(f"Unsupported fusion strategy: {self.fusion_strategy}")

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
            candidate_depth=settings.retrieval_candidate_depth,
            rerank_candidate_pool=settings.rerank_candidate_pool,
            adaptive_routing=settings.adaptive_routing,
        )
