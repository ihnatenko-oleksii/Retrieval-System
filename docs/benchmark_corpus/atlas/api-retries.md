# Atlas API: Retries and Idempotency
## Idempotency keys
Clients should send an `Idempotency-Key` for POST operations that create a charge, start an import, or launch an export. Atlas retains the key and the normalized request fingerprint for 24 hours. Repeating the same key and body returns the original result without repeating the side effect; repeating it with a different body returns `idempotency_key_reused`. The key is scoped to the organization and endpoint, so a key used for an import cannot safely be reused for a billing request.

## Retryable responses
The SDK may retry 408, 429, 500, 502, 503, and 504 responses when the operation is safe to repeat. It must not blindly retry 400, 401, 403, 404, or 409 responses because these usually indicate a malformed request, missing authorization, a missing resource, or a state conflict. A network disconnect after the server accepted a POST is unknown outcome; the client should recover with the same idempotency key rather than create a new key.

## Backoff and attempt limit
The recommended delay is `min(30 seconds, 0.5 * 2^n) + jitter`, where `n` is the zero-based retry number. The default SDK makes at most six retries and caps an individual delay at 30 seconds. Jitter is required so many workers do not reconnect on the same boundary. A caller may lower the attempt limit for an interactive request, but it should not remove the delay or treat a 429's `Retry-After` as optional.

## Connect, read, and total timeouts
Atlas distinguishes connect timeout, read timeout, and total operation timeout. A connect timeout means no socket was established; a read timeout means the server may still be processing the request. The SDK defaults to three seconds for connect, 30 seconds for a read, and two minutes total for a normal API call. Long-running imports and exports return a job before the total timeout, so clients should poll the job rather than hold one HTTP request open.

## Duplicate outcome recovery
When a request times out after transmission, the client cannot infer whether the server committed it. It should query the idempotency status endpoint or repeat the request with the original key. A response marked `replayed: true` confirms that Atlas returned a stored result. Creating a fresh key can produce a duplicate charge or import. For endpoints without idempotency support, the safest recovery is a read-based reconciliation before any second write.
