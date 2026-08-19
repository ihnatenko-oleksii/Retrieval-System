# Atlas API: Errors and Status Codes
## Error envelope
An error response contains a stable code, a human-readable message, a request ID, and optional field details. Clients should branch on the code rather than parse the message, because wording can change while the code remains compatible. The request ID is safe to show to support and is the primary key for server-side logs. A successful HTTP status with an application warning is not converted into an error envelope.

## Client and authorization errors
400 indicates a malformed request or invalid field value, while 401 means the access credential is missing, expired, or invalid. 403 means the credential is valid but lacks permission, and 404 means the caller cannot access the named resource or the resource does not exist. A client must not turn 401 into a password retry loop or turn 403 into a request for a different token without checking scopes. The response code and error code together identify the remediation.

## Conflict and precondition errors
409 indicates a state conflict such as an already existing external ID or an invalid workflow transition. 412 indicates that an If-Match or version precondition failed. These responses are not transient network errors; retrying the same body without reading current state will usually repeat the conflict. A caller should fetch the resource, reconcile the version or identity, and then submit an intentional update.

## Rate and server errors
429 means a rate bucket is exhausted and includes Retry-After. 500 is an unexpected application error, while 502 and 503 generally indicate an upstream or temporary service problem. 504 means the gateway did not receive a timely upstream response; the server may still have completed a write. Retry policy depends on idempotency support and should use backoff. Error severity in a log should not be inferred from HTTP status alone.

## Request tracing
Every request receives a request ID, and clients can supply a valid correlation ID that Atlas echoes into trace metadata. A retry receives a new request ID even when an idempotency key returns the same stored result. Support investigations should collect the organization, endpoint, timestamp, request ID, idempotency key when relevant, and response code. Do not include access tokens or full customer payloads in a support ticket merely to improve traceability.
