import logging
import os
import pickle
import re

from rank_bm25 import BM25Okapi

from app.core.models import Chunk

logger = logging.getLogger(__name__)


class BM25Store:
    def __init__(self, persist_dir: str = "./storage"):
        self.persist_dir = persist_dir
        self.file_path = os.path.join(persist_dir, "bm25_index.pkl")
        self.bm25: BM25Okapi = None
        self.chunks: list[dict] = []
        self._load()

    def _tokenize(self, text: str) -> list[str]:
        # Unicode-aware word tokenization: handles punctuation like "SOW?"
        raw_tokens = re.findall(r"\w+", (text or "").lower())
        stopwords = {
            "co",
            "to",
            "jest",
            "i",
            "oraz",
            "a",
            "w",
            "we",
            "na",
            "z",
            "za",
            "do",
            "o",
            "od",
            "po",
            "dla",
            "czy",
            "jak",
            "jaki",
            "jakie",
            "ktory",
            "która",
            "the",
            "is",
            "are",
            "what",
            "of",
            "in",
            "for",
            "and",
        }
        return [t for t in raw_tokens if len(t) > 1 and t not in stopwords]

    def _chunk_key(self, metadata: dict, content: str) -> str:
        file_path = metadata.get("file_path", "")
        chunk_index = metadata.get("chunk_index", "")
        return f"{file_path}::{chunk_index}::{hash(content)}"

    def _rebuild_index(self):
        if not self.chunks:
            self.bm25 = None
            return
        corpus = [self._tokenize(c["content"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "rb") as f:
                    data = pickle.load(f)
                    # Always rebuild to keep tokenization behavior current.
                    self.chunks = data["chunks"]
                    self._rebuild_index()
                logger.info("Loaded BM25 index.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")

    def save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            with open(self.file_path, "wb") as f:
                pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)
            logger.info("Saved BM25 index.")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        existing_keys = {self._chunk_key(c.get("metadata", {}), c.get("content", "")) for c in self.chunks}

        for chunk in chunks:
            record = {"id": chunk.id, "content": chunk.content, "metadata": chunk.metadata.model_dump()}
            key = self._chunk_key(record["metadata"], record["content"])
            if key in existing_keys:
                continue
            self.chunks.append(record)
            existing_keys.add(key)

        self._rebuild_index()
        self.save()

    def query(self, query_text: str, n_results: int = 5) -> list[tuple[str, dict, float]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query_text)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)

        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]

        results = []
        for idx in top_n:
            if scores[idx] > 0:
                chunk = self.chunks[idx]
                results.append((chunk["content"], chunk["metadata"], float(scores[idx])))

        return results
