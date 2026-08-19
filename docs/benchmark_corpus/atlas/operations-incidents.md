# Atlas Operations: Incidents
## Severity and ownership
An incident is assigned a severity, service, commander, communications owner, and current hypothesis. Severity reflects customer impact and urgency, not the number of alerts. The incident commander can change severity as evidence improves, while the original declaration remains in the timeline. Ownership is explicit so a handoff does not depend on which team happens to be watching a dashboard.

## Timeline events
The incident timeline records declaration, acknowledgements, hypotheses, mitigations, deployments, and recovery checks with an actor and timestamp. Monitoring alerts are linked events, not substitutes for a narrative update. Editing a description adds a correction event rather than changing the original text. A post-incident review uses the immutable timeline to separate what was known at the time from what became clear later.

## Mitigation versus resolution
Mitigation reduces customer impact without proving the underlying fault is fixed, such as disabling a feature or routing reads to a replica. Resolution requires a recovery check and an owner who confirms the service objective is restored. Closing an incident after a mitigation hides recurrence risk. The status API therefore distinguishes mitigated, monitoring, resolved, and closed states.

## Communication cadence
For a high-severity incident, the communications owner posts an internal update every 30 minutes and a customer update at the cadence in the status policy. A quiet period is still an update when the hypothesis has not changed. Status messages should state impact, mitigation, next check, and uncertainty without exposing private customer data. A resolved post is separate from the first message so readers can see the incident duration.

## Post-incident review
The review includes impact window, detection, timeline, contributing factors, successful and failed mitigations, and follow-up owners with due dates. It should distinguish a trigger from a root cause and should not blame an individual for following the approved runbook. Follow-up tasks are tracked separately from incident closure. A review can be amended when new evidence appears, but the amendment preserves the original conclusion and date.
