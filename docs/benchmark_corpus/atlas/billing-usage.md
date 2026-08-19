# Atlas Billing: Usage Metering
## Meter event acceptance
The usage meter accepts events with an account, meter name, quantity, occurred-at timestamp, and producer event ID. The producer event ID is the deduplication key, so resending an event after a timeout is safe when the ID and quantity are unchanged. An event with a different quantity but an already seen ID is rejected as a conflicting duplicate. Accepted events are immutable; a correction is represented by a compensating event.

## Aggregation windows and late events
Usage is aggregated into hourly windows for operational dashboards and daily windows for invoice calculation. Events can arrive up to 72 hours late and are assigned by occurred-at time, not arrival time. A late event may reopen a provisional daily total and will appear in the next invoice preview refresh. Events older than the late-arrival window require a billing adjustment because finalized historical invoices are not silently rewritten.

## Billable units
Each meter defines how raw quantities become billable units. A search request costs one unit, a bulk write costs ten, and a full export costs twenty; these costs are billing weights, not HTTP rate-limit counters even when the numbers happen to match. A meter can round fractional quantities according to its contract. The usage API returns both raw quantity and billable units so a customer can reproduce the invoice calculation.

## Estimates and finalized totals
The usage dashboard shows an estimate based on currently accepted events and the account's active price schedule. An invoice preview adds pending late-event risk and tax assumptions, while a finalized invoice uses the locked schedule and closes its billing window. Estimates can move in either direction before finalization. A customer should not treat a temporary dashboard total as an invoice commitment or use it to reconcile a paid invoice.

## Usage exports
Usage exports are asynchronous jobs that produce CSV or Parquet files by account, meter, day, and producer event ID. The export records the aggregation version and price schedule reference used for billable-unit calculations. Download URLs expire after seven days, and requesting a new URL does not rerun the job. A reconciliation process should keep the export's job ID and compare its sum with the invoice line-item quantity.
