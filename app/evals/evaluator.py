import json
import logging
import math
from typing import List, Dict, Any
from app.vector_store.chroma_store import VectorStore
from app.vector_store.bm25_store import BM25Store
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.bm25_store = BM25Store()
        self.retriever = Retriever(self.vector_store, self.bm25_store)
        self.generator = Generator()

    def evaluate_file(self, jsonl_path: str) -> Dict[str, float]:
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                cases = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            logger.error(f"Failed to read eval file: {e}")
            return {}
            
        if not cases:
            logger.warning("No evaluation cases found.")
            return {}

        metrics = {
            f"recall@{self.top_k}": 0.0,
            f"precision@{self.top_k}": 0.0,
            "mrr": 0.0,
            "ndcg": 0.0,
            "keyword_hit_rate": 0.0
        }
        
        for case in cases:
            question = case.get("question", "")
            expected_source = case.get("expected_source", "")
            expected_keywords = case.get("expected_keywords", [])
            
            # 1. Retrieval
            chunks = self.retriever.retrieve(question, top_k=self.top_k)
            retrieved_sources = [chunk[1].get("file_name", "") for chunk in chunks]
            
            # Metrics computation
            relevant_ranks = []
            for rank, src in enumerate(retrieved_sources, start=1):
                if expected_source and expected_source in src:
                    relevant_ranks.append(rank)
                    
            if relevant_ranks:
                first_rank = relevant_ranks[0]
                metrics[f"recall@{self.top_k}"] += 1.0  # At least one relevant found
                metrics[f"precision@{self.top_k}"] += len(relevant_ranks) / self.top_k
                metrics["mrr"] += 1.0 / first_rank
                
                # Simple nDCG (relevance = 1 for expected_source, 0 otherwise)
                dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
                idcg = 1.0 # Max 1 relevant doc assumed for ideal
                metrics["ndcg"] += dcg / idcg

            # 2. Generation
            answer_dict = self.generator.generate_answer(question, chunks)
            answer_text = answer_dict.get("final_answer", "").lower()
            
            if expected_keywords:
                hits = sum(1 for kw in expected_keywords if str(kw).lower() in answer_text)
                if hits == len(expected_keywords):
                    metrics["keyword_hit_rate"] += 1.0
                elif hits > 0:
                    metrics["keyword_hit_rate"] += hits / len(expected_keywords)
            else:
                # If no expected keywords provided, treat as hit for this case
                metrics["keyword_hit_rate"] += 1.0
                    
        # Average metrics
        num_cases = len(cases)
        for k in metrics:
            metrics[k] = round(metrics[k] / num_cases, 4)
            
        return metrics
