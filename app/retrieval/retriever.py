from typing import List, Tuple, Dict, Optional
from app.vector_store.chroma_store import VectorStore
from app.vector_store.bm25_store import BM25Store
from app.retrieval.rewriter import QueryRewriter
from app.retrieval.reranker import Reranker
from app.core.config import settings
import logging
import re

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store = None):
        self.vector_store = vector_store
        self.bm25_store = bm25_store or BM25Store()
        self.rewriter = QueryRewriter()
        self.reranker = Reranker()

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
        # Prefer acronym-style matches like "SOW", "CRD", etc.
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

    def retrieve(self, query: str, top_k: int = None, chat_history: Optional[List[Dict[str, str]]] = None) -> List[Tuple[str, dict, float]]:
        if top_k is None:
            top_k = settings.retrieval_top_k
        initial_k = max(top_k * 6, 20)

        # 1. Rewrite and Expand
        base_query = self.rewriter.rewrite_query(query, chat_history=chat_history)
        queries = self.rewriter.expand_query(base_query)

        # 2. Retrieve for all queries
        dense_results = []
        sparse_results = []
        
        for q in queries:
            # Dense retrieval
            chroma_res = self.vector_store.query(query_text=q, n_results=initial_k)
            if chroma_res and chroma_res.get("documents") and len(chroma_res["documents"]) > 0:
                docs = chroma_res["documents"][0]
                metas = chroma_res["metadatas"][0] if chroma_res.get("metadatas") else [{}] * len(docs)
                dists = chroma_res["distances"][0] if chroma_res.get("distances") else [0.0] * len(docs)
                for d, m, c in zip(docs, metas, dists):
                    dense_results.append((d, m, c))

            # Sparse retrieval
            bm25_res = self.bm25_store.query(query_text=q, n_results=initial_k)
            sparse_results.extend(bm25_res)

        # 3. Deduplicate and normalize
        # Chroma distances are often smaller=better (e.g. cosine distance), so invert=True
        norm_dense = self._normalize_scores(dense_results, invert=True)
        
        # BM25 scores are larger=better, invert=False
        norm_sparse = self._normalize_scores(sparse_results, invert=False)

        # 4. Reciprocal Rank Fusion / Weighted Sum
        combined_scores: Dict[str, Tuple[str, dict, float]] = {}
        
        wd = settings.hybrid_search_weights_dense
        ws = settings.hybrid_search_weights_sparse
        if self._extract_acronyms(query):
            # Acronym lookups (e.g., "SOW") tend to work better with lexical emphasis.
            wd, ws = 0.35, 0.65

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

        # 4b. Intent-aware boost (helps acronym definition lookups).
        self._apply_query_intent_boost(query, combined_scores)

        # Sort by combined score
        final_results = sorted(combined_scores.values(), key=lambda x: x[2], reverse=True)
        
        # Take a wider net for reranking if reranker is enabled
        candidate_count = settings.rerank_top_n * 2 if settings.reranker_on else top_k
        candidates = [(c, m, s) for c, m, s in final_results][:candidate_count]

        # 5. Reranking
        if settings.reranker_on:
            reranked = self.reranker.rerank(query, candidates, top_n=top_k)
            return reranked
        
        return candidates[:top_k]

    def format_context(self, retrieved_chunks: List[Tuple[str, dict, float]]) -> str:
        context_parts = []
        for i, (content, meta, _) in enumerate(retrieved_chunks, start=1):
            source = meta.get("file_name", "Unknown Source")
            chunk_index = meta.get("chunk_index", "?")
            context_parts.append(f"--- Source [{i}]: {source} (Chunk {chunk_index}) ---\n{content}\n")
        return "\n".join(context_parts)
