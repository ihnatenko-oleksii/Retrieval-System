# Atlas Search: Lexical Retrieval
## When lexical search is the right tool
Lexical retrieval uses BM25 over analyzed terms and is strongest when a user knows an exact product code, error string, legal phrase, or field name. It rewards terms that are rare across the index and repeated in a candidate, so a query containing idempotency_key_reused should prefer a chunk containing that literal error code. Lexical search does not infer that credential rollover means service-account rotation unless the synonym is indexed or the query is expanded.

## Field boosts
The lexical ranker searches title, tags, and body fields with boosts of 3.0, 2.0, and 1.0 respectively. A rare term in a title can therefore outrank a longer body match, but boosts do not make an absent term appear. The explanation endpoint reports field term matches and the contribution of each boost. When a document has no title or tags, its body still participates at the base weight rather than being discarded.

## Phrase and punctuation handling
Text inside quotation marks is treated as a phrase and rewards adjacent terms in order. The analyzer lowercases ordinary words and normalizes a hyphen to a separator for natural-language terms, but it preserves a code-like token such as SKU-42 as an exact alias as well. A query for rate limit can match rate-limit, while a query for "rate limit" requires the two terms to be adjacent after normalization.

## Tokenization and stopwords
Common English stopwords are removed from the BM25 term stream, while domain identifiers, numbers, and error codes are retained. Stemming is enabled for English prose but not for fields marked keyword. The keyword analyzer is used for organization IDs, external IDs, and status codes because changing their spelling changes their identity. A tokenization preview is available in the search debug response when an operator needs to explain a miss.

## Fuzzy matching
Fuzzy matching is disabled for the default lexical query because edit-distance expansion can turn a precise error code into many noisy candidates. A caller can opt in with fuzzy=true and a maximum edit distance of one for user-facing typo tolerance. Fuzzy mode is not applied to keyword fields, quoted phrases, or identifiers beginning with atk_. For exact incident codes, correcting the user's spelling before searching is safer than broadening every term.
