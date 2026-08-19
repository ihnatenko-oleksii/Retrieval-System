# Atlas Ingestion: Deduplication and Corrections
## Record identity
The primary ingestion identity is source ID plus external ID. Atlas does not deduplicate solely on name, email, timestamp, or matching content because those fields can legitimately repeat. If a producer cannot provide a stable external ID, it must use a pre-ingestion identity service and persist the assigned ID. Identity decisions are made before schema validation so a malformed update can still be reported against the correct existing record.

## Upsert and create-only modes
Upsert mode accepts a new record or creates a new revision of the record with the same identity. Create-only mode accepts only unseen identities and reports duplicate_external_id for a repeat. Neither mode silently merges two different external IDs that happen to have equal content. The import job summary separates created, updated, duplicate, and rejected counts so a caller can reconcile its source ledger.

## Conflicting retries
The same source ID, external ID, idempotency key, and body can be safely retried after a network failure. Reusing an idempotency key with a different body returns idempotency_key_reused before a second write occurs. A new idempotency key with the same identity is allowed to create an update according to the selected mode. Producers should persist both keys and identities because they answer different questions about delivery versus record state.

## Corrections
A correction changes the record through a new revision and records the reason, actor, and source event. It does not edit the original raw payload. A correction event may use the same external ID with a later producer event ID, and an audit reader can follow the revision chain. Deleting and recreating a record with a new external ID is not a correction because it breaks references and creates a new identity.

## Reconciliation
Daily reconciliation compares source counts by identity and revision state with Atlas accepted, updated, rejected, and quarantined counts. Mismatches are classified as delivery gap, duplicate, validation failure, or source-side cancellation. Reconciliation should query the import job and event logs before resubmitting data. A full resend can increase duplicates and usage without repairing a missing source event, while a targeted correction preserves the audit trail.
