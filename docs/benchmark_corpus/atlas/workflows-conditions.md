# Atlas Workflows: Conditions and Branches
## Condition evaluation
A condition step evaluates an expression against the run input, outputs from prior tasks, and selected environment facts. It produces true, false, or unknown when a referenced value is absent. Unknown follows the policy's explicit branch; it is not automatically treated as false for security-sensitive workflows. The condition result and expression version are stored in the run history so a later definition change does not alter the old explanation.

## Branch joins
A workflow can branch into parallel paths and join when all required paths complete, when any path succeeds, or when a quorum is met. A skipped branch counts as complete only when its skip rule says so. A failed branch does not become successful merely because another branch completed. Join behavior should be selected separately from task retry behavior because a retry changes an attempt, while a join changes the run's dependency state.

## Input and output references
Expressions use typed paths such as input.customer.region or steps.lookup.output.plan. A missing path is reported with its path and step ID. String coercion is disabled for numeric comparisons, so the value 10 is not equal to the string "10" unless the expression explicitly converts it. This avoids a condition accidentally taking the paid branch for a value received from an untyped webhook.

## Versioned definitions
A running workflow uses the definition version captured when its run was created. Publishing a new condition or branch changes future runs only. An operator can choose a controlled migration for a suspended run, but the migration records old and new definition versions and re-evaluates from a declared checkpoint. Editing a draft workflow does not mutate any active or historical run.

## Branch diagnostics
The run detail page shows the normalized expression, input paths used, values after redaction, result, selected branch, and policy version. Sensitive fields are displayed as present or absent rather than their content. A diagnostic marked unknown should lead the operator to inspect upstream output and schema mapping, not to raise a retry count. Replaying the same run without changing input will repeat the same branch result.
