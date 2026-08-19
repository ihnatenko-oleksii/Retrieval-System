# Evaluation Metrics

The built-in evaluator measures retrieval and answer quality against a
JSONL file of labeled test cases, where each case has a question, an
expected source document, and expected answer keywords.

## Retrieval metrics

Recall@K is 1 if the expected source appears anywhere in the top K
retrieved chunks, and 0 otherwise. Precision@K is the fraction of the top K
retrieved chunks that come from the expected source. MRR (Mean Reciprocal
Rank) is the reciprocal of the rank of the first chunk from the expected
source, averaged across all cases. nDCG (normalized Discounted Cumulative
Gain) rewards relevant chunks appearing near the top of the ranking more
than relevant chunks appearing near the bottom, normalized against the best
possible ordering of the relevant chunks that were actually found.

## Answer metric

Keyword hit rate measures whether the generated answer text contains each
of the case's expected keywords, as a fraction of keywords found. This
requires the LLM to actually generate an answer, unlike the retrieval
metrics, which only need the retriever to run.

## Tuning workflow

The tuning dashboard sweeps combinations of `top_k`, hybrid weights,
reranking, and query preprocessing switches across an evaluation file, and
ranks configurations by a composite score that weights keyword hit rate,
MRR, and recall@K.
