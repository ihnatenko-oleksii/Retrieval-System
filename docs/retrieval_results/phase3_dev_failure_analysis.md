# Phase 3 DEV Failure Analysis

Configuration: `qwen3-hybrid-generic-instruction`; cases: 40; misses or non-rank-1 cases: 3.

Ranks are recorded without using relevance labels as runtime features. Learned-sparse, ColBERT, and reranker ranks are explicitly null because they are not part of the frozen configuration.

| Case | Category | First relevant rank | nDCG | Relevant chunk stream ranks |
|---|---|---:|---:|---|
| exact-12 | exact_terminology | 3 | 0.5 | atlas/search-lexical.md::1 dense=3 bm25=8 final=3 |
| ambiguous-06 | ambiguous | 2 | 0.1749 | atlas/api-errors.md::1 dense=18 bm25=None final=27; atlas/api-retries.md::1 dense=None bm25=4 final=14; atlas/ingestion-webhooks.md::2 dense=None bm25=8 final=17; atlas/workflows-retries.md::1 dense=1 bm25=None final=2 |
| ambiguous-10 | ambiguous | 3 | 0.1743 | atlas/access-service-accounts.md::2 dense=4 bm25=None final=5; atlas/billing-usage.md::0 dense=10 bm25=18 final=11; atlas/ingestion-dedup.md::0 dense=16 bm25=9 final=14; atlas/ingestion-webhooks.md::1 dense=3 bm25=6 final=3 |
