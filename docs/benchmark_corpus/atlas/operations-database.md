# Atlas Operations: Database Reliability
## Replica lag
The read replica alert threshold is 30 seconds for customer-facing queries and 120 seconds for analytical replicas. Lag is measured from the primary commit timestamp to the replica replay timestamp, not from the time a monitoring sample was collected. During lag, reads that require read-after-write consistency should use the primary or a session pinned to the primary. A healthy connection pool does not prove that replicas are current.

## Failover objectives
The database service targets a five-minute recovery time objective and a one-minute recovery point objective for the primary region. A failover promotes the most recent healthy replica, so writes acknowledged after the last durable replication point may need reconciliation. Read endpoints can return a temporary primary_transition error during promotion. Clients should retry idempotent reads with backoff and should not blindly replay an unknown-outcome write.

## Connection pools
Each application process has a bounded connection pool with a maximum of 40 connections and a five-second acquisition timeout. Pool exhaustion returns a distinct db_pool_timeout metric; it is not the same as a database connection refusal. Long transactions hold connections and can starve short reads, so operators should inspect transaction duration and pool wait time together. Raising the pool limit without checking database capacity can increase failure during a traffic spike.

## Migration locks
Schema migrations acquire an advisory lock for the target database and fail fast after a two-minute wait rather than blocking application traffic indefinitely. Online migrations should add nullable structures first, backfill in batches, and remove old fields only after readers no longer depend on them. A migration lock timeout means another migration is active or stuck; restarting application workers does not release a lock owned by the database session.

## Backups and restore tests
Full backups run daily and incremental WAL archives run continuously. A backup is considered usable only after a restore test verifies schema, row counts, and application-level checksums in an isolated environment. The restore runbook records the backup timestamp, recovery point, elapsed time, and missing-object count. A successful backup job without a recent restore test is evidence of storage, not evidence that the service can meet its recovery objective.
