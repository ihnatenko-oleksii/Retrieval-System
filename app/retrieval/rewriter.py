import logging

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

DETERMINISTIC_LLM_OPTIONS = {"temperature": 0, "seed": 1729, "num_predict": 128}


class QueryRewriter:
    def __init__(
        self,
        model_name: str = None,
        timeout_seconds: float | None = None,
        strict: bool = False,
    ):
        self.model_name = model_name or settings.llm_model
        self.timeout_seconds = timeout_seconds
        self.strict = strict
        self._client = ollama.Client(timeout=timeout_seconds) if timeout_seconds is not None else None
        self._rewrite_cache: dict[tuple[str, str, bool], str] = {}
        self._expansion_cache: dict[tuple[str, int], list[str]] = {}

    def _chat(self, messages: list[dict[str, str]]):
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "options": DETERMINISTIC_LLM_OPTIONS,
        }
        if self._client is not None:
            return self._client.chat(**kwargs)
        return ollama.chat(**kwargs)

    @staticmethod
    def _response_content(response) -> str:
        if isinstance(response, dict):
            return str(response.get("message", {}).get("content", ""))
        message = getattr(response, "message", None)
        return str(getattr(message, "content", "") or "")

    def rewrite_query(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        enabled: bool | None = None,
    ) -> str:
        """
        Rewrites a query to make it clearer and more specific for retrieval.
        If chat_history is provided, rewrites follow-up questions into standalone queries.

        `enabled` overrides `settings.query_rewriting_on` when provided.
        """
        rewriting_enabled = settings.query_rewriting_on if enabled is None else bool(enabled)
        if not rewriting_enabled and not chat_history:
            return query
        cache_key = (query, repr(chat_history), rewriting_enabled)
        if cache_key in self._rewrite_cache:
            return self._rewrite_cache[cache_key]

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
            response = self._chat([{"role": "user", "content": prompt}])
            rewritten = self._response_content(response).strip()
            rewritten = rewritten.strip("\"'")
            logger.info(f"Rewrote query: '{query}' -> '{rewritten}'")
            result = rewritten if rewritten else query
            self._rewrite_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Failed to rewrite query: {e}")
            if self.strict:
                raise
            return query

    def expand_query(
        self,
        query: str,
        num_expansions: int = 2,
        enabled: bool | None = None,
    ) -> list[str]:
        """
        Generates alternate versions of a query to increase recall.

        `enabled` overrides `settings.query_expansion_on` when provided.
        """
        expansion_enabled = settings.query_expansion_on if enabled is None else bool(enabled)
        if not expansion_enabled:
            return [query]
        cache_key = (query, num_expansions)
        if cache_key in self._expansion_cache:
            return self._expansion_cache[cache_key]

        prompt = f"Generate {num_expansions} different versions of the following search query to help retrieve more relevant documents. Focus on different keywords, synonyms, or angles of the same intent. Return each query on a new line. Do not include numbering, bullets, or explanations.\n\nOriginal query: {query}\n\nAlternate queries:"

        try:
            response = self._chat([{"role": "user", "content": prompt}])
            content = self._response_content(response).strip()
            expansions = [q.strip("- *\"'") for q in content.split("\n") if q.strip()]
            queries = [query] + expansions[:num_expansions]
            logger.info(f"Expanded query into: {queries}")
            self._expansion_cache[cache_key] = queries
            return queries
        except Exception as e:
            logger.error(f"Failed to expand query: {e}")
            if self.strict:
                raise
            return [query]
