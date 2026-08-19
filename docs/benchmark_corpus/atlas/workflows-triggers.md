# Atlas Workflows: Triggers
## Trigger types
A workflow can start from an event, a schedule, or a manual request. Event triggers subscribe to a named topic, schedule triggers use a cron expression, and manual triggers require an authenticated user with run permission. These trigger types share the same workflow definition but produce different trigger metadata. A manual run does not prove that the event subscription or schedule is configured correctly.

## Event filters
Event-trigger filters support AND and OR groups over payload paths, event type, and project. A missing payload path evaluates as false rather than as an empty string. Filters are evaluated before a workflow run is created, so a nonmatching event does not consume task concurrency. The debug view shows the normalized filter and the first predicate that failed; changing a filter affects future events and does not retroactively reprocess old deliveries.

## Debounce and cooldown
Debounce collects matching events for five minutes and starts one run with the latest payload after the window closes. Cooldown is different: it suppresses new runs for a configured period after a run starts. A workflow can use both, but the debounce window is measured before run creation while cooldown is measured from the created run. Setting cooldown to zero does not disable debounce, and setting debounce to zero does not serialize concurrent events.

## Schedule timezones and DST
A schedule stores an IANA timezone with its cron expression. At a daylight-saving spring transition, a nonexistent local time is skipped; at an autumn transition, an ambiguous local time runs once using the first occurrence. The scheduler records the resolved UTC instant in the run metadata. Using a fixed offset instead of Europe/Rome or another IANA name prevents Atlas from applying the correct seasonal change.

## Trigger deduplication
Event deliveries carry an event ID and workflow triggers keep a seven-day deduplication record. Receiving the same event again does not create a second run, even if the first run failed; an operator can replay the existing run explicitly. Manual requests use a caller-supplied idempotency key for 24 hours. Schedule occurrences use their resolved UTC instant and workflow ID as the identity, so a scheduler retry does not double-start a run.
