# Atlas API: Pagination and Filtering
## Cursor pagination
List endpoints use opaque cursor pagination. The response contains items and `next_cursor`; the client sends that value as `cursor` to request the next page. A cursor encodes a position and a snapshot boundary, so clients must not parse, sort, or manufacture one. Cursors expire after 15 minutes and can become invalid after a retention boundary. If the response has no `next_cursor`, the current page is the end of the snapshot.

## Page-size defaults and limits
The default page size is 25 and the maximum is 100. Asking for `page_size=0` or a value above 100 returns `invalid_page_size`; the server does not silently clamp it. A client can use a smaller page when records are large or latency matters. Page size changes are allowed between cursor requests, but changing the sort order or filter while continuing an old cursor is invalid because the position belongs to the original query.

## Filters and sort order
Multiple filters are combined with AND. For example, `status=active` and `project_id=p7` narrows results to active records in project p7. A filter value can contain a comma-separated OR list only when the field documentation explicitly marks it as multi-valued. The default sort is ascending `created_at` with the record ID as a tie-breaker. Full-text search is a separate endpoint and is not enabled by adding a `q` parameter to a list request.

## Snapshot consistency
The first page establishes a read snapshot. Records created after that boundary do not appear in later pages, while a record deleted during traversal may disappear. This prevents an insertion from shifting every subsequent cursor. If a caller needs a current view rather than a consistent export, it should start a new request after reaching the end. The snapshot boundary is not a wall-clock filter that can be copied into a different query.

## Invalid cursors
An expired, malformed, or query-mismatched cursor returns 400 with error code `invalid_cursor` and a hint to restart pagination. The client should discard the cursor and restart from the first page using the original filters and sort. Replaying the same invalid cursor will not repair it. If a job needs a durable traversal over many hours, use an export job, whose snapshot and continuation tokens have a longer retention period.
