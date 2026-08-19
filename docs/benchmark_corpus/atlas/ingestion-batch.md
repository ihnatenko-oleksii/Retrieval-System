# Atlas Ingestion: Batch Imports
## Batch submission and limits
The batch endpoint accepts up to 10,000 records or 25 MB per request, whichever limit is reached first. A successful submission returns 202 and a job ID; it does not mean every record has been validated or written. The caller should persist the job ID and poll the import-status endpoint. Sending a second request while the first job is still running can duplicate records unless the same idempotency key and source identity are used.

## Per-record validation
Validation is reported per record rather than as one all-or-nothing batch failure. The job summary contains accepted, rejected, and skipped counts, while the error stream identifies the record number, field path, error code, and human-readable message. A record with an unknown required field is rejected, but valid records in the same batch continue. Clients should repair rejected records and submit a small correction batch instead of resubmitting every accepted record.

## External-ID deduplication
Every source must provide an external_id; Atlas uses the pair (source_id, external_id) as the deduplication identity. A repeated pair is an update candidate, not a new record. In upsert mode the latest valid record replaces the previous version; in create_only mode it is reported as duplicate_external_id. Changing only a display name does not create a new identity, while changing the source ID intentionally creates a separate record.

## Date and timezone mapping
Date-only fields are stored as calendar dates and are not shifted by timezone. Timestamps without an offset are rejected in strict mode and interpreted in the source's declared timezone in warning mode. Atlas stores accepted instants in UTC while retaining the original offset in ingestion diagnostics. A source should declare an IANA timezone such as Europe/Rome, not a fixed +01:00, when its local clock observes daylight-saving changes.

## Safe batch retries
The batch request's idempotency key is retained for 24 hours, and a retry with the same body returns the original job ID. If the client loses the response, it should first query that key before submitting a new batch. Polling a job until completed or completed_with_errors is safe and does not consume write rate units. A failed job can be corrected and resubmitted with a new key because accepted records are already protected by external-ID deduplication.
