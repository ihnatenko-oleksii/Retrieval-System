import ollama
import logging
from typing import List, Optional, Dict
from app.core.config import settings

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.llm_model

    def rewrite_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        enabled: Optional[bool] = None,
    ) -> str:
        """
        Rewrites a query to make it clearer and more specific for retrieval.
        If chat_history is provided, rewrites follow-up questions into standalone queries.

        `enabled` overrides `settings.query_rewriting_on` when provided.
        """
        rewriting_enabled = settings.query_rewriting_on if enabled is None else bool(enabled)
        if not rewriting_enabled and not chat_history:
            return query

        if chat_history:
            history_text = ""
            for msg in chat_history[-4:]:
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content")
                if content is None:
                    continue
                history_text += f"{role}: {str(content)}\n"

            prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all relevant context from the history.
Return ONLY the standalone query text, without any explanations or formatting.

Conversation History:
{history_text}

Follow-up question: {query}
Standalone query:"""
        else:
            prompt = f"Rewrite the following search query to be clear, highly specific, and optimized for semantic search retrieval. Return ONLY the rewritten query text, without any explanations or formatting.\n\nOriginal query: {query}\n\nRewritten query:"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            rewritten = response.get("message", {}).get("content", "").strip()
            rewritten = rewritten.strip('"\'')
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            return rewritten if rewritten else query
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}")
            return query

    def expand_query(
        self,
        query: str,
        num_expansions: int = 2,
        enabled: Optional[bool] = None,
    ) -> List[str]:
        """
        Generates alternate versions of a query to increase recall.

        `enabled` overrides `settings.query_expansion_on` when provided.
        """
        expansion_enabled = settings.query_expansion_on if enabled is None else bool(enabled)
        if not expansion_enabled:
            return [query]

        prompt = f"Generate {num_expansions} different versions of the following search query to help retrieve more relevant documents. Focus on different keywords, synonyms, or angles of the same intent. Return each query on a new line. Do not include numbering, bullets, or explanations.\n\nOriginal query: {query}\n\nAlternate queries:"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.get("message", {}).get("content", "").strip()
            expansions = [q.strip("- *\"'") for q in content.split("\n") if q.strip()]
            queries = [query] + expansions[:num_expansions]
            logger.info(f"Expanded query into: {queries}")
            return queries
        except Exception as e:
            logger.error(f"Failed to expand query: {e}")
            return [query]
