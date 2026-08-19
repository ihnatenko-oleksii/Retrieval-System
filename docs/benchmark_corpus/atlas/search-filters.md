# Atlas Search: Filters and Facets
## Structured filters
Structured filters apply exact constraints such as project, status, owner, date range, and schema version. They are evaluated against typed fields and do not use fuzzy or semantic similarity. A filter on status=active excludes an archived record even when its text is a strong semantic match. The search request reports invalid field names and type mismatches before ranking, so a zero-result query is not automatically evidence that the corpus lacks a document.

## Facets and counts
Facets group candidate records by a configured field and return counts for the selected result set. Facet counts are calculated after structured filters but before the final page cut, so counts can exceed the number of displayed results. High-cardinality identifiers are not enabled as facets by default. A facet is descriptive and does not reorder hits unless a caller explicitly selects a sort such as count descending.

## Date boundaries
Date filters use an inclusive start and exclusive end boundary. A request for 2026-01-01 through 2026-02-01 therefore includes January 31 but excludes records exactly at February 1 in the chosen timezone. Timestamp fields are converted to UTC after the boundary timezone is applied. Mixing a local date with a UTC literal can shift records around midnight, so the response echoes the normalized interval.

## Filter and semantic ordering
Structured filters are applied before vector candidate selection when the index supports the field; otherwise they are applied to the candidate pool and the response reports an approximate-filter flag. A semantic score cannot bring a record back after a filter excludes it. For high-selectivity filters, a lexical or exact lookup may have better recall than a small dense candidate pool. Benchmark cases should distinguish filter correctness from text-ranking quality.

## Filter explanations
The debug response explains each filter as matched, not matched, or unknown, including the field type and normalized value. Unknown is possible when an older record predates a field introduction and should not be conflated with false. The explanation also shows whether the filter ran pre-index or post-candidate. Operators can use this information to diagnose an unexpectedly empty result without changing ranking weights.
