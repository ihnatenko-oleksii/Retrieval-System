from typing import List, Tuple, Dict, Optional
from app.vector_store.chroma_store import VectorStore
from app.vector_store.bm25_store import BM25Store
from app.retrieval.rewriter import QueryRewriter
from app.retrieval.reranker import Reranker
from app.core.config import settings
from app.core.runtime_config import RetrievalConfig
import logging
import re

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store = None,
        reranker: Optional[Reranker] = None,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store or BM25Store()
        self.rewriter = QueryRewriter()
        # Allow the caller (e.g. tuning UI) to supply a preloaded reranker so
        # we don't pay the CrossEncoder load cost on every trial.
        self.reranker = reranker or Reranker()

    def _normalize_scores(self, results: List[Tuple[str, dict, float]], invert: bool = False) -> List[Tuple[str, dict, float]]:
        if not results:
            return []

        scores = [dist for _, _, dist in results]
        min_score, max_score = min(scores), max(scores)

        normalized = []
        for content, meta, score in results:
            if max_score == min_score:
                norm_score = 1.0
            else:
                norm_score = (score - min_score) / (max_score - min_score)

            if invert:
                norm_score = 1.0 - norm_score

            normalized.append((content, meta, norm_score))

        return normalized

    def _extract_acronyms(self, query: str) -> List[str]:
        return re.findall(r"\b[A-Z]{2,10}\b", query or "")

    def _apply_query_intent_boost(
        self,
        query: str,
        combined_scores: Dict[str, List],
    ) -> None:
        acronyms = self._extract_acronyms(query)
        if not acronyms:
            return
        is_definition_query = bool(re.search(r"\b(co to jest|what is|definition|definicja)\b", (query or "").lower()))

        for acronym in acronyms:
            pattern = re.compile(rf"\b{re.escape(acronym)}\b", re.IGNORECASE)
            for key, value in combined_scores.items():
                content, meta, score = value
                file_name = str(meta.get("file_name", "")).lower()
                glossary_hint = ("słownik" in file_name) or ("slownik" in file_name) or ("glossary" in file_name)
                acronym_hit = bool(pattern.search(content or ""))
                if acronym_hit:
                    if glossary_hint:
                        value[2] = score + (2.0 if is_definition_query else 1.0)
                    else:
                        value[2] = score + 0.15

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[RetrievalConfig] = None,
    ) -> List[Tuple[str, dict, float]]:
        # Resolve runtime knobs from `config`, falling back to global settings.
        if config is None:
            resolved_top_k = top_k if top_k is not None else settings.retrieval_top_k
            dense_weight = settings.hybrid_search_weights_dense
            sparse_weight = settings.hybrid_search_weights_sparse
            reranker_on = settings.reranker_on
            rerank_top_n = settings.rerank_top_n
            query_rewriting_on = settings.query_rewriting_on
            query_expansion_on = settings.query_expansion_on
        else:
            resolved_top_k = top_k if top_k is not None else config.top_k
            dense_weight = config.dense_weight
            sparse_weight = config.sparse_weight
            reranker_on = config.reranker_on
            rerank_top_n = config.rerank_top_n
            query_rewriting_on = config.query_rewriting_on
            query_expansion_on = config.query_expansion_on

        initial_k = max(resolved_top_k * 6, 20)

        # 1. Rewrite and expand
        base_query = self.rewriter.rewrite_query(
            query,
            chat_history=chat_history,
            enabled=query_rewriting_on,
        )
        queries = self.rewriter.expand_query(base_query, enabled=query_expansion_on)

        # 2. Retrieve for all queries
        dense_results = []
        sparse_results = []

        for q in queries:
            chroma_res = self.vector_store.query(query_text=q, n_results=initial_k)
            if chroma_res and chroma_res.get("documents") and len(chroma_res["documents"]) > 0:
                docs = chroma_res["documents"][0]
                metas = chroma_res["metadatas"][0] if chroma_res.get("metadatas") else [{}] * len(docs)
                dists = chroma_res["distances"][0] if chroma_res.get("distances") else [0.0] * len(docs)
                for d, m, c in zip(docs, metas, dists):
                    dense_results.append((d, m, c))

            bm25_res = self.bm25_store.query(query_text=q, n_results=initial_k)
            sparse_results.extend(bm25_res)

        # 3. Normalize
        norm_dense = self._normalize_scores(dense_results, invert=True)
        norm_sparse = self._normalize_scores(sparse_results, invert=False)

        # 4. Hybrid fusion (weights are overridable).
        wd = dense_weight
        ws = sparse_weight
        if self._extract_acronyms(query):
            # Acronym lookups work better with lexical emphasis.
            wd, ws = 0.35, 0.65

        combined_scores: Dict[str, List] = {}
        for content, meta, score in norm_dense:
            key = content
            if key not in combined_scores:
                combined_scores[key] = [content, meta, 0.0]
            combined_scores[key][2] += score * wd

        for content, meta, score in norm_sparse:
            key = content
            if key not in combined_scores:
                combined_scores[key] = [content, meta, 0.0]
            combined_scores[key][2] += score * ws

        self._apply_query_intent_boost(query, combined_scores)

        final_results = sorted(combined_scores.values(), key=lambda x: x[2], reverse=True)

        candidate_count = rerank_top_n * 2 if reranker_on else resolved_top_k
        candidates = [(c, m, s) for c, m, s in final_results][:candidate_count]

        # 5. Optional reranking
        if reranker_on:
            reranked = self.reranker.rerank(
                query,
                candidates,
                top_n=resolved_top_k,
                enabled=True,
            )
            return reranked

        return candidates[:resolved_top_k]

    def format_context(self, retrieved_chunks: List[Tuple[str, dict, float]]) -> str:
        context_parts = []
        for i, (content, meta, _) in enumerate(retrieved_chunks, start=1):
            source = meta.get("file_name", "Unknown Source")
            chunk_index = meta.get("chunk_index", "?")
            context_parts.append(f"--- Source [{i}]: {source} (Chunk {chunk_index}) ---\n{content}\n")
        return "\n".join(context_parts)
