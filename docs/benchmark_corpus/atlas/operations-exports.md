# Atlas Operations: Export Jobs
## Job creation and formats
An export request returns 202 with a job ID and does not stream a large result through the request connection. The caller selects CSV or Parquet, fields, filters, and a consistency mode. CSV is convenient for spreadsheets, while Parquet preserves typed columns and is preferred for analytical pipelines. The job status moves from queued to running to completed, completed_with_errors, or failed; a 202 alone is not evidence that a file exists.

## Export permissions
Export authorization is checked when the job is created and again when a download URL is issued. A user who loses project access before completion cannot download the result. Organization-wide exports require the export scope and, for regulated fields, a recent step-up authentication. The job metadata retains the creator and authorization decision so an administrator can explain why a file was or was not released.

## File retention and URLs
Completed export files are retained for seven days. Download URLs are signed, scoped to one job, and expire after 15 minutes; requesting a fresh URL during the retention window does not recompute the export. After seven days the file is deleted, but the job metadata and row counts remain. A long-running consumer should copy the file into its own controlled storage rather than polling for a permanent Atlas URL.

## Consistency modes
Snapshot mode reads from a consistent boundary established when the job starts, so records created later do not appear. Current mode reads pages as workers process them and may include changes made during the run. Snapshot mode is the default for billing and compliance exports; current mode is suitable for operational lists where freshness matters more than a single boundary. The selected mode is stored in job metadata and the output manifest.

## Large export parts
Large exports are divided into numbered parts with a manifest containing row counts, checksums, schema version, and the price or aggregation version where applicable. Parts may finish out of order, but the manifest is published only after all required parts pass checksum validation. A consumer should download the manifest first, verify every part, and concatenate in part-number order. Retrying one failed part does not restart the completed parts or change the job ID.
