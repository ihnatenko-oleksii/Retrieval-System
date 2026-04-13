import ollama
import logging
from typing import List, Tuple, Optional, Dict
from app.core.config import settings

logger = logging.getLogger(__name__)

class Generator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.llm_model

    def _build_prompt(self, query: str, context: str) -> str:
        prompt = f"""
You are a knowledgeable and precise assistant. Your task is to answer the user's question based strictly on the provided context.

Context information is below, separated by source markers:
---------------------
{context}
---------------------

Instructions:
1. Answer the question using ONLY the provided context.
2. If the answer is not contained in the context, say "I don't know based on the provided context." Do not guess or make up information.
3. When providing facts, include inline citations referencing the source number, like [1], [2], etc.

Question:
{query}

Answer:
"""
        return prompt.strip()

    def _safe_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if "text" in value and isinstance(value["text"], str):
                return value["text"]
            return str(value)
        if isinstance(value, list):
            parts = []
            for v in value:
                text = self._safe_text(v)
                if text:
                    parts.append(text)
            return " ".join(parts).strip()
        return str(value)

    def generate_answer(self, query: str, retrieved_chunks: List[Tuple[str, dict, float]], chat_history: Optional[List[Dict[str, str]]] = None) -> dict:
        """
        Generates an answer from the retrieved chunks, taking optional chat_history into account.
        Returns a dictionary with the final answer and metadata.
        """
        if not retrieved_chunks:
            return {
                "final_answer": "No relevant context found to answer the question.",
                "retrieved_sources": [],
                "confidence_notes": "Low confidence due to missing context."
            }

        # Format context
        context_parts = []
        sources = []
        for i, (content, meta, dist) in enumerate(retrieved_chunks, start=1):
            source_name = meta.get("file_name", "Unknown Source")
            context_parts.append(f"--- Source [{i}]: {source_name} ---\n{content}\n")
            sources.append({
                "source_id": i,
                "file_name": source_name,
                "distance": dist,
                "chunk_index": meta.get("chunk_index")
            })

        context_text = "\n".join(context_parts)
        prompt = self._build_prompt(query, context_text)

        messages = []
        if chat_history:
            # We don't want the LLM to get confused by past contexts if they were included,
            # so we just pass the history as standard user/assistant messages.
            # To keep it lightweight, we limit to the last 4 messages.
            for msg in chat_history[-4:]:
                role = msg.get("role")
                content = self._safe_text(msg.get("content"))
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
                
        messages.append({"role": "user", "content": prompt})

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            answer = response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            answer = f"Error generating answer: {str(e)}"

        return {
            "final_answer": answer,
            "retrieved_sources": sources,
            "confidence_notes": "Generated using local LLM with provided context."
        }
