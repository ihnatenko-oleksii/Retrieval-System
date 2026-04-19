from dataclasses import dataclass
from typing import Optional

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
    llm_model: Optional[str] = None

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
        )
