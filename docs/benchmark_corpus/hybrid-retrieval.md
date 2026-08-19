# Hybrid Dense + BM25 Retrieval

This system retrieves candidate chunks with two independent methods and
fuses their scores into a single ranking.

## Dense retrieval

Dense retrieval embeds the query and every chunk with a sentence-transformer
model (the default is `intfloat/multilingual-e5-base`) and stores vectors in
ChromaDB. Similarity is measured with cosine distance, where a lower
distance means a closer match.

## Sparse retrieval

Sparse retrieval uses the BM25 algorithm (via `rank_bm25`) over a tokenized,
stopword-filtered corpus. BM25 scores are unbounded and higher is better,
which is the opposite convention from cosine distance.

## Score normalization and fusion

Because dense distances and BM25 scores live on different scales, each
candidate list is independently min-max normalized into the 0 to 1 range
before fusion. Dense scores are inverted after normalization so that a
smaller distance becomes a larger (better) score. The final hybrid score is
a weighted sum: `dense_weight * normalized_dense + sparse_weight *
normalized_sparse`. The default weights are 0.7 for dense and 0.3 for
sparse, and both are configurable at runtime.

## Deduplication

When the same chunk is found by both dense and sparse retrieval, its scores
are merged rather than counted twice. Chunks are deduplicated using a
stable identity built from their source file path and chunk index, not
their raw text content, because two different chunks can coincidentally
contain identical or near-identical text.
