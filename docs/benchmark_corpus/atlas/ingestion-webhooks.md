# Atlas Ingestion: Webhooks
## Delivery schedule and acknowledgement
Atlas delivers an event immediately, then retries after approximately 1 minute, 5 minutes, 30 minutes, and 6 hours when the endpoint does not acknowledge it. Any 2xx response acknowledges the event; a 3xx redirect, 4xx validation response, timeout, or 5xx response is a failed delivery. The receiver should persist the event before returning 2xx and do expensive processing asynchronously so the acknowledgement arrives within ten seconds.

## Signature verification
Each webhook includes X-Atlas-Signature and X-Atlas-Timestamp headers. The signature is an HMAC-SHA256 over the timestamp, a period, and the exact raw request body using the endpoint secret. The verifier must compare signatures in constant time and reject timestamps more than five minutes from its current clock. Parsing and reserializing JSON before verification changes the signed bytes and will make an otherwise valid event fail.

## Event IDs and ordering
X-Atlas-Event-Id is globally unique for one event and remains the same across every retry. Receivers should store it with a processed status and treat a duplicate delivery as an acknowledgement, not as a second business event. Events from one topic are usually ordered but not guaranteed to arrive in order across workers or after a retry. Consumers that require ordering should compare the event sequence and pause until missing predecessors arrive.

## Replay window
The webhook console can replay an event from the previous seven days. A replay receives a new delivery attempt but retains the original event ID and event timestamp, so signature verification still uses the replayed body and its current delivery headers. Replaying does not bypass endpoint authorization or change the topic. Use replay for a recovered receiver or a controlled backfill, not as a substitute for a durable event archive.

## Dead-letter behavior
After ten unsuccessful deliveries, an event moves to the endpoint's dead-letter queue and automatic delivery stops. The queue retains the body, headers, failure summaries, and attempt timestamps for 14 days. An operator can retry one event or a bounded time range after fixing the receiver. A successful manual retry removes that event from the dead-letter view but does not erase the original failure history.
