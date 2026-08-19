# Atlas Billing: Credits
## Credit ledger and expiry
Credits are ledger entries with a grant amount, remaining amount, source, effective date, and optional expiry date. Consumption creates separate debit entries rather than mutating the original grant, so the ledger can explain every balance. Expired units become unavailable at midnight in the billing account timezone. A credit balance is not cash and cannot be refunded unless the grant terms explicitly allow a refund.

## Promotional and prepaid credits
Promotional credits and prepaid credits are both usable units but have different precedence and refund rules. Atlas consumes promotional credits that expire sooner before prepaid credits; within one class it consumes the oldest effective grant first. Promotional grants are not included in a cash refund, while unused prepaid credits may be refundable under the contract. The balance API exposes the grant source so a billing UI can explain why a total changed.

## Negative balances and overage
If billable usage arrives after all credits are consumed, the account can enter a negative credit balance. A negative balance is an overage amount owed on the next invoice, not a temporary rate-limit condition. New credits first offset the negative amount before becoming available units. Suspending an account stops new usage but does not erase an existing negative ledger balance or convert it into a credit.

## Adjustments and approvals
A manual credit adjustment requires a reason, source ticket, amount, and approver when the amount exceeds 500 units. The creator cannot approve their own adjustment. Approved adjustments create a new ledger grant or debit and never rewrite historical entries. A rejected request remains visible with its reason. Billing administrators should use an adjustment rather than editing an invoice when the correction concerns account credit consumption.

## Credit audit events
Credit audit events identify the account, grant or debit ID, source, actor, request ID, and resulting balance. A usage debit includes the meter event that caused it; a promotional grant includes the campaign or contract reference. Deleting a project does not delete its credit events because grants can be organization-scoped. Reconciliation should compare the ledger sum to the reported balance and investigate any drift before issuing a refund.
