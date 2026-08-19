# Query Rewriting and Expansion

Before retrieval runs, the query can optionally be preprocessed by the LLM
in two independent ways.

## Query rewriting

Query rewriting turns a short or ambiguous query into a clearer, more
specific search query. It is also used, regardless of the rewriting toggle,
whenever chat history is present: a follow-up question like "what about the
sparse side?" is rewritten into a standalone question that captures the
context from earlier turns, so retrieval does not depend on conversational
context it cannot see.

## Query expansion

Query expansion asks the LLM to generate a small number of alternate
phrasings of the same query, covering different keywords, synonyms, or
angles on the same intent. Retrieval then runs once per variant (including
the original query) and all candidate results are merged before fusion.
This increases recall on queries where the corpus uses different wording
than the user, at the cost of extra retrieval calls per question.

## Failure behavior

Both rewriting and expansion call the local LLM through Ollama. If that
call fails for any reason (model not pulled, Ollama not running, timeout),
the system logs the error and falls back to using the original query
unmodified, so a preprocessing failure never blocks retrieval entirely.
