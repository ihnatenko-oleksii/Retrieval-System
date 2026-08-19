# Atlas Ingestion: Schema Evolution
## Registry versions
Schemas are registered with a name and a monotonically increasing version. A new version starts in draft, becomes active after compatibility checks, and may later be deprecated; only one version is active for a source at a time. Existing records retain the schema version that accepted them. Publishing a schema does not rewrite old records, so a migration job is required when historical data must follow the new shape.

## Compatible and breaking changes
Adding an optional field with a default is backward compatible. Adding a required field, removing a field, changing a field's primitive type, or changing an enum value is breaking for existing producers or consumers. A rename is treated as remove-plus-add unless an explicit alias is declared. Compatibility is checked against the last active version, not against every draft that was abandoned.

## Field aliases and mapping
A field alias maps a legacy input name to one canonical schema field during ingestion. For example, acct_no may alias account_number; both names cannot be present with different values in one record. Aliases are scoped to a schema version and are included in validation diagnostics. A mapping rule changes the shape or value, while an alias changes only the accepted input name; confusing those operations can hide an unintended data transformation.

## Validation modes
Strict mode rejects the record on the first schema violation and returns a field path plus code. Warning mode accepts the record when the violation is recoverable and writes a diagnostic event. Quarantine mode stores the raw record outside the searchable dataset for later repair and does not count it as accepted usage. The mode is selected per import job, so changing a source default does not silently alter an already running job.

## Migration jobs
A migration job reads records in bounded pages, validates them against the target active version, and writes a new revision while retaining the original revision for audit. It reports checkpoint, converted, failed, and quarantined counts. A failed page can be retried from its checkpoint, but a target-version write must be idempotent. The job does not change the registry's active version; publish the schema separately after migration evidence is reviewed.
