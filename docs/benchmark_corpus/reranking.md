# Reranking

Reranking is an optional second-stage step that reorders the hybrid
candidate list using a cross-encoder model.

## Why rerank

Hybrid fusion combines two independent, weakly-calibrated signals (dense
similarity and BM25 relevance), so its top results are a reasonable but
imperfect approximation of true relevance. A cross-encoder reads the query
and each candidate chunk together, rather than comparing separately encoded
vectors, and produces a much sharper relevance score at the cost of extra
compute per query.

## Model and behavior

The default reranker model is `cross-encoder/ms-marco-MiniLM-L-6-v2`. When
reranking is enabled, the retriever fetches a wider candidate pool than the
final `top_k`, scores each candidate against the query with the
cross-encoder, sorts by that score, and truncates to `rerank_top_n`. If the
model fails to load or prediction raises an exception, reranking falls back
silently to the original hybrid ordering so a broken model never causes a
hard failure.

## When to enable it

Reranking is off by default because it adds latency and requires
downloading an additional model. It is most valuable when the hybrid
fusion step returns many plausible-looking chunks that need a final,
finer-grained relevance judgment before being shown to the LLM.
