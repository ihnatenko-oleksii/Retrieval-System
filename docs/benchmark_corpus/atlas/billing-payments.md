# Atlas Billing: Payments and Collection
## Payment attempts
An invoice payment creates an attempt with a provider, amount, currency, and idempotency key. An attempt can be pending, succeeded, failed, or canceled; a pending attempt is not a paid invoice. The provider response may arrive after the request times out, so the collection worker reconciles by provider reference before creating a second attempt. Payment state and invoice state are related but are not the same field.

## Collection retries
Automatic collection retries use a schedule of one hour, one day, and three days after a retryable provider failure. A hard decline, invalid payment method, or customer-requested stop is not retried automatically. The collection policy counts attempts per invoice and payment method, while the API rate limit counts requests. Waiting for the collection schedule is therefore different from waiting for a 429 reset.

## Refunds and reversals
A refund is requested against a succeeded payment and may be full or partial. A provider reversal can move a payment from succeeded to reversed when the provider reports a post-settlement failure; Atlas creates a receivable adjustment rather than silently marking the invoice paid. Refunds and credit notes are both customer-visible reductions but affect payment rail and invoice ledger differently. Reconciliation stores provider references for both operations.

## Currency and rounding
Payment amounts are integer minor units, such as cents, and the currency is explicit on every attempt. Atlas does not convert a payment by using the customer's current currency setting after an invoice is issued. Tax and discounts are rounded at the line and invoice levels according to the billing currency rules. A provider amount mismatch is a reconciliation error, not a reason to retry the payment with a different rounded value.

## Payment audit
Payment audit events record invoice ID, attempt ID, provider reference, actor or collection worker, idempotency key, amount, currency, and transition reason. Sensitive payment credentials are tokenized by the provider and are not present in Atlas event bodies. Investigators should use the attempt and provider reference together because one invoice can have several failed attempts before one succeeds. Deleting a payment method does not erase historical payment events.
