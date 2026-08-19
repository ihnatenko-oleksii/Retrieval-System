# Atlas Operations: Queue Reliability
## Queue lag and consumer health
Queue lag is the age of the oldest ready message, while consumer lag is the number of messages not yet acknowledged by active workers. A queue can have low message count but high age when one old message is stuck. Dashboards show ready, in-flight, acknowledged, and dead-letter counts separately. A high producer rate with healthy consumer age is not an incident by itself; alert thresholds should combine age with throughput and error rate.

## Poison messages
A poison message repeatedly fails because its payload or dependency cannot be processed. After the configured delivery limit, Atlas moves it to a quarantine queue and records the last error, attempt count, and consumer version. Quarantine is different from a transient retry: it stops automatic redelivery to protect the rest of the queue. Operators should inspect and repair the message or deploy a consumer fix before replaying it.

## Visibility timeout
A worker lease has a 60-second visibility timeout by default. The worker must acknowledge before the timeout or extend the lease while processing a long task. If the lease expires, another consumer may receive the same message, so handlers must be idempotent. A visibility timeout is not a retry delay and does not mean the original worker has stopped; the original process may still finish after a duplicate delivery starts.

## Priority queues
Critical, normal, and bulk queues have separate consumer pools. Critical work can bypass normal and bulk backlog, but it cannot exceed the downstream service's rate limit. Moving a message from bulk to critical changes scheduling priority and is audited with an operator reason. Priority does not change message order within one queue; FIFO behavior is preserved only among messages that share the same priority and partition.

## Draining and shutdown
Draining stops new message claims, allows in-flight leases to finish, and reports a deadline. A worker that reaches the deadline should release or let leases expire and exit so another worker can continue. Force shutdown abandons in-flight work and can produce duplicate delivery. During deployment, operators should drain consumers before removing capacity and monitor both ready age and in-flight count until the queue is stable.
