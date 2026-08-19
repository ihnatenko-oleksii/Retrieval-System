# Atlas API: Rate Limits and Quotas
## Limit dimensions
Atlas applies request-rate limits independently at the organization, project, and source-IP dimensions. A project can therefore receive a 429 even while its organization has unused capacity, and two projects in one organization do not borrow each other's per-project burst allowance. Limits are also grouped by endpoint family: reads, writes, searches, and exports have separate buckets. The most restrictive bucket that applies to a request determines whether the request is admitted.

## Response headers and 429 handling
Successful responses expose `RateLimit-Limit`, `RateLimit-Remaining`, and `RateLimit-Reset` in seconds until the current window ends. A throttled response has status 429 and includes `Retry-After`; that value is the minimum delay before the client should try again. Clients should honor the longer of `Retry-After` and their own backoff calculation. A 429 is a rate signal, not evidence that the request body was invalid, so changing payload syntax will not resolve it.

## Default request rates
The default project limit is 600 read requests per minute and 120 write requests per minute. A short burst of up to twice the per-minute rate is allowed for ten seconds, but sustained traffic is still measured against the one-minute budget. These are defaults rather than contractual capacity: an enterprise plan can have a different limit, and the effective value is returned in `RateLimit-Limit`. Raising a quota does not remove endpoint-specific concurrency limits.

## Weighted request cost
Some operations consume more than one request unit. A normal search costs one unit, a bulk write costs ten, and a full export costs twenty. A client that sends 120 bulk writes can exhaust a 120-unit write budget even though it sent fewer than 120 HTTP requests. The response headers report remaining units, not remaining calls. SDKs should expose the cost in retry diagnostics so operators do not misread a fast drop in `Remaining`.

## Rate limit versus quota
A rate limit controls how quickly requests may arrive; a quota controls how many records or units an organization may consume over a billing period. Reaching a monthly quota returns `quota_exceeded` and will not be fixed by waiting for `RateLimit-Reset`. Reaching a rate limit returns 429 and normally clears after the window or after a plan change. Dashboards show both values because increasing the monthly quota does not increase per-minute throughput.
