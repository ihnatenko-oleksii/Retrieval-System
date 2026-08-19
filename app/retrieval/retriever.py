import logging
import re

from app.core.config import settings
from app.core.runtime_config import RetrievalConfig
from app.retrieval.query_router import QueryDecision, QueryGate
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
        query_gate: QueryGate | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store or BM25Store()
        self.rewriter = rewriter or QueryRewriter()
        self.query_gate = query_gate or QueryGate()
        self.last_trace: dict[str, object] = {}
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

    def _fuse_query_variants(
        self,
        streams: list[tuple[str, list[tuple[str, dict, float]], float]],
        *,
        strategy: str,
        rrf_k: int,
    ) -> list[list]:
        """Fuse already-fused query rankings as a separate second stage."""
        if not streams:
            return []
        if len(streams) == 1:
            return [list(result) for result in streams[0][1]]

        combined_scores: dict[tuple, list] = {}
        if strategy == "weighted_linear":
            normalized_streams = [
                (label, self._normalize_scores(results), weight)
                for label, results, weight in streams
            ]
            for _, results, weight in normalized_streams:
                for content, meta, score in results:
                    key = self._chunk_identity(meta, content)
                    if key not in combined_scores:
                        combined_scores[key] = [content, meta, 0.0]
                    combined_scores[key][2] += score * weight
        else:
            weighted = strategy == "weighted_rrf"
            for _, results, weight in streams:
                effective_weight = weight if weighted else 1.0
                for rank, (content, meta, _) in enumerate(results, start=1):
                    key = self._chunk_identity(meta, content)
                    if key not in combined_scores:
                        combined_scores[key] = [content, meta, 0.0]
                    combined_scores[key][2] += effective_weight / (rrf_k + rank)

        return sorted(combined_scores.values(), key=lambda value: value[2], reverse=True)

    def _retrieve_single_query(
        self,
        query: str,
        *,
        resolved_top_k: int,
        dense_weight: float,
        sparse_weight: float,
        fusion_strategy: str,
        candidate_depth: int,
        adaptive_routing: bool,
        raw_query: str,
        rrf_k: int,
    ) -> tuple[list[list], list[tuple[str, dict, float]], list[tuple[str, dict, float]]]:
        """Run stage 1 dense/sparse retrieval for one query variant."""
        initial_k = max(resolved_top_k, candidate_depth)
        effective_dense_weight, effective_sparse_weight = dense_weight, sparse_weight
        if adaptive_routing:
            effective_dense_weight, effective_sparse_weight = self._adaptive_weights(
                raw_query,
                dense_weight,
                sparse_weight,
            )

        dense_results: list[tuple[str, dict, float]] = []
        sparse_results: list[tuple[str, dict, float]] = []
        if effective_dense_weight > 0:
            chroma_res = self.vector_store.query(query_text=query, n_results=initial_k)
            if chroma_res and chroma_res.get("documents") and len(chroma_res["documents"]) > 0:
                docs = chroma_res["documents"][0]
                metas = chroma_res["metadatas"][0] if chroma_res.get("metadatas") else [{}] * len(docs)
                dists = chroma_res["distances"][0] if chroma_res.get("distances") else [0.0] * len(docs)
                for content, meta, distance in zip(docs, metas, dists, strict=True):
                    dense_results.append((content, meta, distance))

        if effective_sparse_weight > 0:
            sparse_results = self.bm25_store.query(query_text=query, n_results=initial_k)

        final_results = self._fuse_results(
            dense_results,
            sparse_results,
            dense_weight=effective_dense_weight,
            sparse_weight=effective_sparse_weight,
            strategy=fusion_strategy,
            rrf_k=rrf_k,
        )
        if adaptive_routing and effective_sparse_weight > 0:
            combined_scores = {
                self._chunk_identity(meta, content): [content, meta, score]
                for content, meta, score in final_results
            }
            self._apply_query_intent_boost(raw_query, combined_scores)
            final_results = sorted(combined_scores.values(), key=lambda value: value[2], reverse=True)
        return final_results, dense_results, sparse_results

    @staticmethod
    def _confidence_score(
        results: list[list],
        dense_results: list[tuple[str, dict, float]],
        sparse_results: list[tuple[str, dict, float]],
        chunk_identity,
    ) -> float:
        """Estimate confidence from score separation and backend agreement."""
        if not results:
            return 0.0
        if len(results) == 1:
            separation = 0.25
        else:
            top_score = float(results[0][2])
            second_score = float(results[1][2])
            separation = min(1.0, max(0.0, top_score - second_score) * 5.0)

        if dense_results and sparse_results:
            dense_top = chunk_identity(dense_results[0][1], dense_results[0][0])
            sparse_top = chunk_identity(sparse_results[0][1], sparse_results[0][0])
            agreement = 1.0 if dense_top == sparse_top else 0.0
        else:
            agreement = 0.25
        return round(0.5 * separation + 0.5 * agreement, 4)

    @staticmethod
    def _policy_enabled(enabled: bool, policy: str, decision: QueryDecision, variant: str) -> bool:
        if not enabled or policy == "never":
            return False
        if policy == "always":
            return True
        return decision.should_rewrite if variant == "rewrite" else decision.should_expand

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
            rewrite_policy = settings.rewrite_policy
            expansion_policy = settings.expansion_policy
            include_original_query = settings.include_original_query
            multi_query_fusion_strategy = settings.multi_query_fusion_strategy
            original_query_weight = settings.original_query_weight
            rewrite_query_weight = settings.rewrite_query_weight
            expansion_query_weight = settings.expansion_query_weight
            confidence_routing = settings.confidence_routing
            confidence_threshold = settings.confidence_threshold
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
            rewrite_policy = config.rewrite_policy
            expansion_policy = config.expansion_policy
            include_original_query = config.include_original_query
            multi_query_fusion_strategy = config.multi_query_fusion_strategy
            original_query_weight = config.original_query_weight
            rewrite_query_weight = config.rewrite_query_weight
            expansion_query_weight = config.expansion_query_weight
            confidence_routing = config.confidence_routing
            confidence_threshold = config.confidence_threshold
            candidate_depth = config.candidate_depth or max(resolved_top_k * 6, 20)
            adaptive_routing = config.adaptive_routing
            rrf_k = config.rrf_k

        if resolved_top_k < 1:
            raise ValueError("top_k must be at least 1")

        decision = self.query_gate.decide(query)
        rewrite_requested = self._policy_enabled(
            query_rewriting_on,
            rewrite_policy,
            decision,
            "rewrite",
        )
        expansion_requested = self._policy_enabled(
            query_expansion_on,
            expansion_policy,
            decision,
            "expansion",
        )
        potential_variant = rewrite_requested or expansion_requested
        # Legacy expansion already returned the original query alongside its
        # generated variants; retain that behavior unless a rewrite is the
        # explicit replacement-only control.
        include_original_effective = include_original_query or (expansion_requested and not rewrite_requested)
        need_original = include_original_effective or confidence_routing or not potential_variant
        original_results: list[list] = []
        original_dense: list[tuple[str, dict, float]] = []
        original_sparse: list[tuple[str, dict, float]] = []
        if need_original:
            original_results, original_dense, original_sparse = self._retrieve_single_query(
                query,
                resolved_top_k=resolved_top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                fusion_strategy=fusion_strategy,
                candidate_depth=candidate_depth,
                adaptive_routing=adaptive_routing,
                raw_query=query,
                rrf_k=rrf_k,
            )

        confidence_score = None
        confidence_triggered = False
        if confidence_routing and need_original:
            confidence_score = self._confidence_score(
                original_results,
                original_dense,
                original_sparse,
                self._chunk_identity,
            )
            if confidence_score < confidence_threshold and not decision.protected_signals:
                if query_rewriting_on and rewrite_policy == "selective":
                    rewrite_requested = True
                    confidence_triggered = True
                if query_expansion_on and expansion_policy == "selective":
                    expansion_requested = True
                    confidence_triggered = True

        streams: list[tuple[str, list[tuple[str, dict, float]], float]] = []
        if include_original_effective or not (rewrite_requested or expansion_requested):
            streams.append(("original", original_results, original_query_weight))

        rewrite_applied = False
        expansion_applied = False
        if rewrite_requested:
            rewritten_query = self.rewriter.rewrite_query(
                query,
                chat_history=chat_history,
                enabled=True,
            )
            rewrite_applied = True
            if rewritten_query and rewritten_query != query:
                rewritten_results, _, _ = self._retrieve_single_query(
                    rewritten_query,
                    resolved_top_k=resolved_top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    fusion_strategy=fusion_strategy,
                    candidate_depth=candidate_depth,
                    adaptive_routing=adaptive_routing,
                    raw_query=query,
                    rrf_k=rrf_k,
                )
                streams.append(("rewrite", rewritten_results, rewrite_query_weight))

        if expansion_requested:
            expansion_queries = self.rewriter.expand_query(query, enabled=True)
            expansion_applied = True
            for expansion_query in expansion_queries:
                if not expansion_query or expansion_query == query:
                    continue
                expanded_results, _, _ = self._retrieve_single_query(
                    expansion_query,
                    resolved_top_k=resolved_top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    fusion_strategy=fusion_strategy,
                    candidate_depth=candidate_depth,
                    adaptive_routing=adaptive_routing,
                    raw_query=query,
                    rrf_k=rrf_k,
                )
                streams.append(("expansion", expanded_results, expansion_query_weight))

        if not streams:
            # A failed/empty LLM response must never erase the original result.
            if not original_results:
                original_results, _, _ = self._retrieve_single_query(
                    query,
                    resolved_top_k=resolved_top_k,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    fusion_strategy=fusion_strategy,
                    candidate_depth=candidate_depth,
                    adaptive_routing=adaptive_routing,
                    raw_query=query,
                    rrf_k=rrf_k,
                )
            streams.append(("original", original_results, original_query_weight))
        final_results = self._fuse_query_variants(
            streams,
            strategy=multi_query_fusion_strategy,
            rrf_k=rrf_k,
        )
        self.last_trace = {
            "rewrite_applied": rewrite_applied,
            "expansion_applied": expansion_applied,
            "query_variant_count": len(streams),
            "query_variant_labels": [label for label, _, _ in streams],
            "gate_should_rewrite": decision.should_rewrite,
            "gate_should_expand": decision.should_expand,
            "gate_protected_signals": decision.protected_signals,
            "gate_reasons": decision.reasons,
            "confidence_score": confidence_score,
            "confidence_triggered": confidence_triggered,
        }

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
