# Atlas Search: Ranking and Reranking
## Hybrid score fusion
The default hybrid ranker min-max normalizes the dense similarity signal and the BM25 signal independently, then computes 0.7 * dense_score + 0.3 * lexical_score. Dense distance is inverted after normalization because lower distance is better; BM25 already uses higher-is-better scores. A chunk present in only one candidate list still receives that side's weighted score. The fusion weights are runtime settings, not properties of the embedding model.

## Candidate pool and cross-encoder
When reranking is enabled, Atlas first retrieves up to 50 hybrid candidates, then sends each query-candidate pair to the cross-encoder. The cross-encoder reads both strings together and predicts a relevance score that can distinguish a specific exception from a broad topical match. Only after reranking does Atlas return the requested top 10 results. Fetching fewer than 50 candidates can hide the best answer before the cross-encoder sees it.

## Ranking features
The production ranker may add freshness and source-authority features after text retrieval. Freshness is capped so a new but low-quality note cannot overwhelm a highly authoritative runbook. Authority is derived from document ownership and review status, not from the number of words in a passage. Benchmark comparisons that are meant to isolate retrieval should keep these non-text features disabled and compare the same candidate corpus.

## Deterministic tie-breaking
Equal final scores are ordered by document review timestamp descending and then by stable chunk ID ascending. The tie-break is applied after fusion and again after reranking if the cross-encoder scores are equal within a small tolerance. Stable tie-breaking makes repeated benchmark runs comparable and prevents an unrelated insertion from changing every result with a tied score. It is not a relevance boost and should not be interpreted as evidence that a newer document is correct.

## Explainability
The search debug endpoint can return dense distance, normalized dense contribution, BM25 score, normalized lexical contribution, reranker score, and the final rank for each candidate. These values explain a ranking decision but are not calibrated probabilities. A high BM25 contribution means lexical overlap, while a high reranker score means the cross-encoder judged the pair relevant. Comparing explanations across different embedding models is useful only after rebuilding the index.
