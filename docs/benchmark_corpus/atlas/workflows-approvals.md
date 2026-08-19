# Atlas Workflows: Approvals
## Approval states
An approval step is pending until the required decision count is reached, then becomes approved or rejected. It can become expired when its deadline passes without a decision, and a canceled workflow leaves the step canceled rather than rejected. Downstream tasks start only after approved. The API exposes the decision records separately from the step state so an audit can show who voted, when, and from which session.

## Quorum rules
A step may require one approval, unanimous approval, or a quorum such as two of three eligible approvers. A rejection ends the step immediately unless the workflow explicitly enables a reconsideration branch. Extra approvals after quorum are recorded but do not change the approved state. The quorum is evaluated against distinct approver identities, so two browser sessions belonging to one user count as one decision.

## Approver eligibility
Eligibility can require organization role, project membership, a group, or separation from the requester. The requester is excluded when separation-of-duties is enabled, even if they also hold an administrator role. Eligibility is checked when the decision is submitted, not only when the step is created. Removing a user from a group therefore prevents a later approval but does not erase a decision that was already validly recorded.

## Timeouts and reassignment
The default approval timeout is 48 hours. An owner can reassign a pending step to another eligible approver; reassignment does not reset the original deadline unless the workflow policy explicitly grants an extension. Escalation can notify a manager at 24 hours, but notification is not approval. An expired step can route to a fallback branch or stop the workflow; it cannot be approved retroactively through the normal decision endpoint.

## Approval audit
Approval audit events include workflow run ID, step ID, decision, approver, eligibility snapshot, timestamp, request ID, and optional comment. Comments are immutable after submission. A policy change later does not rewrite the eligibility snapshot. Compliance exports should use the audit stream rather than the current user directory because a deleted user or changed group membership must remain identifiable at decision time.
