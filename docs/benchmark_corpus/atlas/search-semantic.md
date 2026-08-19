# Atlas Search: Semantic Retrieval
## Meaning beyond exact terms
Semantic retrieval compares dense embeddings of the query and passages, so it can connect “how long can a login stay open?” with a chunk that says “idle browser session expires after 60 minutes.” It is useful when the question paraphrases the corpus or uses a related expression rather than an indexed word. Dense similarity can also blur nearby concepts, so an embedding match for “renewal” may retrieve both refresh-token rotation and service-account credential rotation.

## Passage representation
The Atlas embedding model is atlas-embed-v2, trained for multilingual support and short technical passages. Indexing targets about 400 tokens per passage with an overlap of 50 tokens when a section crosses a boundary. The query encoder and passage encoder use their respective prefixes; sending a passage prefix on a query reduces retrieval quality. Re-embedding is required after changing the model because vector distances from different spaces are not comparable.

## Similarity and candidate recall
The vector index uses cosine distance, where a smaller distance is a closer semantic match. Dense retrieval returns a candidate pool before the final top-k cut so a later fusion stage can recover a chunk that was not rank one. A high similarity score is evidence of topical closeness, not a truth judgment. Operators should inspect the candidate text and use lexical or reranking signals when two procedures share most of their vocabulary.

## Multilingual and paraphrase behavior
The model handles English, Italian, and Polish paraphrases reasonably well, but rare product codes and exact error strings remain lexical concerns. Translating an acronym into a natural-language phrase can improve semantic recall while losing the literal identifier needed for diagnosis. Evaluation should include both paraphrased questions and exact terminology rather than treating one retrieval signal as universally superior. Language detection is advisory and does not select a separate index.

## Dense failure modes
Dense retrieval may over-rank a broad overview above a narrow exception when both discuss the same operation. It can also treat negation and numeric thresholds as weak distinctions: “does not retry 409” and “retries 409” may share a high score. A result should therefore be judged against the labeled chunk and its exact claim, especially for policy exceptions, limits, and status-code behavior. Reranking can help only after the correct candidate enters the pool.
