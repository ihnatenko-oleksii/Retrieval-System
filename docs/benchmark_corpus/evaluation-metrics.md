# Evaluation Metrics

The built-in evaluator measures retrieval and answer quality against a JSONL
file of labeled test cases. New cases identify one or more stable chunk IDs
with graded relevance gains; the legacy expected source document format is
still accepted for existing evaluation files.

## Retrieval metrics

For chunk-labeled cases, Recall@K is the fraction of all positively labeled
chunks retrieved in the top K. Precision@K is the fraction of the top K
results that are labeled relevant. MRR (Mean Reciprocal Rank) is the
reciprocal of the rank of the first relevant chunk. nDCG (normalized
Discounted Cumulative Gain) uses the case's graded gains and normalizes
against the ideal ordering of all labeled relevant chunks, including chunks
that were missed. This makes multiple-relevant cases and fine-grained labels
meaningful. Legacy expected_source cases retain binary source-level recall
and cannot estimate the number of relevant chunks outside the result list.

Stable benchmark chunk IDs use the corpus-relative filename and chunk index,
for example atlas/api-retries.md::1. The JSONL relevance object maps those IDs
to gains such as 3 for the primary answer and 1 or 2 for related evidence.

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
