import logging
import re

from app.core.config import settings
from app.core.runtime_config import RetrievalConfig
from app.retrieval.reranker import Reranker
from app.retrieval.rewriter import QueryRewriter
from app.vector_store.bm25_store import BM25Store
from app.vector_store.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store | None = None,
        reranker: Reranker | None = None,
        rewriter: QueryRewriter | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store or BM25Store()
        self.rewriter = rewriter or QueryRewriter()
        # Allow the caller (e.g. tuning UI) to supply a preloaded reranker so
        # we don't pay the CrossEncoder load cost on every trial.
        self.reranker = reranker or Reranker()

    def _normalize_scores(
        self, results: list[tuple[str, dict, float]], invert: bool = False
    ) -> list[tuple[str, dict, float]]:
        if not results:
            return []

        scores = [dist for _, _, dist in results]
        min_score, max_score = min(scores), max(scores)

        normalized = []
        for content, meta, score in results:
            norm_score = 1.0 if max_score == min_score else (score - min_score) / (max_score - min_score)

            if invert and max_score != min_score:
                norm_score = 1.0 - norm_score

            normalized.append((content, meta, norm_score))

        return normalized

    def _chunk_identity(self, meta: dict, content: str) -> tuple:
        """
        Stable dedup key for a retrieved chunk. Prefer (file_path, chunk_index)
        from chunk metadata over raw content: two distinct chunks (different
        source, different position) can share identical or near-identical text
        (headers, boilerplate, short code snippets), and keying on content
        would silently collapse them into one result.
        """
        explicit_chunk_id = meta.get("chunk_id") if meta else None
        if explicit_chunk_id:
            return (explicit_chunk_id,)
        file_path = meta.get("file_path") if meta else None
        chunk_index = meta.get("chunk_index") if meta else None
        if file_path is not None and chunk_index is not None:
            return (file_path, chunk_index)
        return (content,)

    def _extract_acronyms(self, query: str) -> list[str]:
        """Return all-uppercase technical tokens without treating `What` as an acronym."""
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{1,9}\b", query or "")
        return [token for token in tokens if token.isupper()]

    def _has_lexical_signal(self, query: str) -> bool:
        query = query or ""
        return bool(
            self._extract_acronyms(query)
            or re.search(r"[`\"'][^`\"']+[`\"']", query)
            or re.search(r"\b\d{3}\b", query)
            or re.search(r"\b[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+\b", query)
        )

    def _adaptive_weights(self, query: str, dense_weight: float, sparse_weight: float) -> tuple[float, float]:
        """Route only from raw query text; benchmark labels never reach this method."""
        if dense_weight <= 0 or sparse_weight <= 0:
            return dense_weight, sparse_weight
        if self._has_lexical_signal(query):
            return 0.35, 0.65

        # Longer natural-language questions are more likely to benefit from
        # semantic matching; keep short/ambiguous queries at the configured mix.
        if len((query or "").split()) >= 7:
            return 0.8, 0.2
        return dense_weight, sparse_weight

    def _apply_query_intent_boost(
        self,
        query: str,
        combined_scores: dict[tuple, list],
    ) -> None:
        acronyms = self._extract_acronyms(query)
        if not acronyms:
            return
        is_definition_query = bool(re.search(r"\b(co to jest|what is|definition|definicja)\b", (query or "").lower()))

        for acronym in acronyms:
            pattern = re.compile(rf"\b{re.escape(acronym)}\b", re.IGNORECASE)
            for value in combined_scores.values():
                content, meta, score = value
                file_name = str(meta.get("file_name", "")).lower()
                glossary_hint = ("słownik" in file_name) or ("slownik" in file_name) or ("glossary" in file_name)
                acronym_hit = bool(pattern.search(content or ""))
                if acronym_hit:
                    if glossary_hint:
                        value[2] = score + (2.0 if is_definition_query else 1.0)
                    else:
                        value[2] = score + 0.15

    def _fuse_results(
        self,
        dense_results: list[tuple[str, dict, float]],
        sparse_results: list[tuple[str, dict, float]],
        *,
        dense_weight: float,
        sparse_weight: float,
        strategy: str,
        rrf_k: int,
    ) -> list[list]:
        """Fuse candidate lists while preserving stable chunk identity."""
        if strategy == "weighted_linear":
            streams = (
                (self._normalize_scores(dense_results, invert=True), dense_weight),
                (self._normalize_scores(sparse_results), sparse_weight),
            )
            combined_scores: dict[tuple, list] = {}
            for results, weight in streams:
                for content, meta, score in results:
                    key = self._chunk_identity(meta, content)
                    if key not in combined_scores:
                        combined_scores[key] = [content, meta, 0.0]
                    combined_scores[key][2] += score * weight
        else:
            weighted = strategy == "weighted_rrf"
            combined_scores = {}
            streams = (
                (dense_results, dense_weight if weighted else 1.0),
                (sparse_results, sparse_weight if weighted else 1.0),
            )
            for results, weight in streams:
                for rank, (content, meta, _) in enumerate(results, start=1):
                    key = self._chunk_identity(meta, content)
                    if key not in combined_scores:
                        combined_scores[key] = [content, meta, 0.0]
                    combined_scores[key][2] += weight / (rrf_k + rank)

        return sorted(combined_scores.values(), key=lambda value: value[2], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        chat_history: list[dict[str, str]] | None = None,
        config: RetrievalConfig | None = None,
    ) -> list[tuple[str, dict, float]]:
        # Resolve runtime knobs from `config`, falling back to global settings.
        if config is None:
            resolved_top_k = top_k if top_k is not None else settings.retrieval_top_k
            dense_weight = settings.hybrid_search_weights_dense
            sparse_weight = settings.hybrid_search_weights_sparse
            reranker_on = settings.reranker_on
            rerank_top_n = settings.rerank_top_n
            rerank_candidate_pool = settings.rerank_candidate_pool
            query_rewriting_on = settings.query_rewriting_on
            query_expansion_on = settings.query_expansion_on
            fusion_strategy = settings.fusion_strategy
            candidate_depth = settings.retrieval_candidate_depth
            adaptive_routing = settings.adaptive_routing
            rrf_k = 60
        else:
            resolved_top_k = top_k if top_k is not None else config.top_k
            dense_weight = config.dense_weight
            sparse_weight = config.sparse_weight
            reranker_on = config.reranker_on
            rerank_top_n = config.rerank_top_n
            rerank_candidate_pool = config.rerank_candidate_pool
            query_rewriting_on = config.query_rewriting_on
            query_expansion_on = config.query_expansion_on
            fusion_strategy = config.fusion_strategy
            candidate_depth = config.candidate_depth or max(resolved_top_k * 6, 20)
            adaptive_routing = config.adaptive_routing
            rrf_k = config.rrf_k

        if resolved_top_k < 1:
            raise ValueError("top_k must be at least 1")
        initial_k = max(resolved_top_k, candidate_depth)

        # 1. Rewrite and expand. The adaptive router below only sees the raw
        # query, so it cannot accidentally use benchmark category or case IDs.
        base_query = self.rewriter.rewrite_query(
            query,
            chat_history=chat_history,
            enabled=query_rewriting_on,
        )
        queries = self.rewriter.expand_query(base_query, enabled=query_expansion_on)

        # 2. Retrieve for all queries.
        dense_results = []
        sparse_results = []
        if adaptive_routing:
            dense_weight, sparse_weight = self._adaptive_weights(query, dense_weight, sparse_weight)

        for q in queries:
            if dense_weight > 0:
                chroma_res = self.vector_store.query(query_text=q, n_results=initial_k)
                if chroma_res and chroma_res.get("documents") and len(chroma_res["documents"]) > 0:
                    docs = chroma_res["documents"][0]
                    metas = chroma_res["metadatas"][0] if chroma_res.get("metadatas") else [{}] * len(docs)
                    dists = chroma_res["distances"][0] if chroma_res.get("distances") else [0.0] * len(docs)
                    for d, m, c in zip(docs, metas, dists, strict=True):
                        dense_results.append((d, m, c))

            if sparse_weight > 0:
                bm25_res = self.bm25_store.query(query_text=q, n_results=initial_k)
                sparse_results.extend(bm25_res)

        # 3. Fuse and optionally apply the existing acronym intent boost only
        # for adaptive routing; static hybrid remains a true static control.
        final_results = self._fuse_results(
            dense_results,
            sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            strategy=fusion_strategy,
            rrf_k=rrf_k,
        )
        if adaptive_routing and sparse_weight > 0:
            combined_scores = {
                self._chunk_identity(meta, content): [content, meta, score]
                for content, meta, score in final_results
            }
            self._apply_query_intent_boost(query, combined_scores)
            final_results = sorted(combined_scores.values(), key=lambda value: value[2], reverse=True)

        # 4. Rerank only the requested candidate pool, always returning the
        # final top_k. Legacy configs without the new field retain the old
        # `rerank_top_n * 2` candidate behavior.
        if reranker_on:
            candidate_pool = rerank_candidate_pool
            if candidate_pool is None:
                candidate_pool = max(rerank_top_n * 2, resolved_top_k)
            candidate_count = max(resolved_top_k, candidate_pool)
            candidates = [(content, meta, score) for content, meta, score in final_results[:candidate_count]]
            return self.reranker.rerank(
                query,
                candidates,
                top_n=resolved_top_k,
                enabled=True,
            )

        return [tuple(result) for result in final_results[:resolved_top_k]]

    def format_context(self, retrieved_chunks: list[tuple[str, dict, float]]) -> str:
        context_parts = []
        for i, (content, meta, _) in enumerate(retrieved_chunks, start=1):
            source = meta.get("file_name", "Unknown Source")
            chunk_index = meta.get("chunk_index", "?")
            context_parts.append(f"--- Source [{i}]: {source} (Chunk {chunk_index}) ---\n{content}\n")
        return "\n".join(context_parts)
