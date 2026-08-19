# Atlas Workflows: Task Retries
## Attempts and backoff
Each task policy defines a maximum attempt count, initial delay, and backoff multiplier. A transient task can be retried with exponential delay and jitter; the default policy allows five total attempts, including the first execution. The run history records each attempt separately. A retry is a new task attempt within the same workflow run, not a new workflow run, so downstream deduplication should use the run ID and task ID.

## Concurrency and queues
A workflow has a maximum in-flight task count and each task type can have its own queue concurrency. Reaching the limit leaves a ready task queued; it is not a task failure and should not consume an attempt. Queue lag and running counts are exposed separately. Increasing concurrency can improve throughput but may violate a downstream API's rate limit, so the workflow limit and the integration's request limit should be tuned together.

## Failure classification
Transient failures include network timeouts, 429 responses, and temporary dependency unavailability. Permanent failures include invalid input, a missing required permission, and a schema violation that will not change without a new payload. A task may return a typed failure classification; otherwise Atlas applies conservative status-code rules. Permanent failures skip automatic retry and route to the configured failure branch, while transient failures remain eligible until the attempt limit is reached.

## Compensation versus retry
Retry repeats the failed task because its side effect is expected to be safe or idempotent. Compensation runs a separate undo or correction action after a later task makes the original plan impossible. A payment capture that times out should be reconciled with its idempotency key before retrying, while a successfully captured payment followed by an inventory failure may need a refund compensation. These are different controls and should not be represented by one generic retry checkbox.

## Manual replay
Operators can replay a failed task attempt or resume the workflow from a failure branch. A task replay keeps the original run ID but increments the manual-attempt counter and requires a reason. Replaying a permanent validation failure without changing its input will produce the same result. A full workflow replay creates a new run and can repeat earlier side effects, so it requires a new idempotency context and explicit confirmation.
