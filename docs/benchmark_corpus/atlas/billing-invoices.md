# Atlas Billing: Invoices
## Invoice lifecycle
An invoice moves through draft, open, paid, void, and uncollectible states. Draft invoices can be edited; issuing one changes it to open and freezes its line items except through a credit note. A successful payment changes open to paid. Void means the invoice was canceled before collection, while uncollectible records a failed collection decision without pretending the balance was paid. State transitions are audited and cannot be reversed by editing the display status.

## Issue dates and due dates
The issue date is when Atlas finalizes an invoice; the due date is when payment is expected. Both are stored as calendar dates in the billing account's configured timezone, not as UTC instants. A timezone change affects future invoice generation and does not rewrite existing dates. A due date can be extended by an approved adjustment, but changing a customer's profile timezone is not an extension and should not be used to hide an overdue balance.

## Credit notes and refunds
A credit note reduces an invoice balance and remains linked to the original invoice. A refund sends money back through the payment rail after a payment has been collected; it does not change an invoice's historical line items. A credit note can be applied to a future open invoice, while a refund requires a paid transaction and may be limited by the provider. Reporting should distinguish credited amount, refunded amount, and remaining receivable.

## Line items and tax
Each line item records description, quantity, unit price, discount, tax code, and the amount calculated in the invoice currency. Tax is computed from the customer's billing address and the tax code at issuance time. Updating an address later does not recalculate a finalized invoice. A correction requires a credit note and a replacement invoice so the original tax decision remains auditable.

## PDF and delivery
The invoice PDF is generated asynchronously after issuance and can be downloaded from the invoice detail page or delivered by email. A temporary download URL expires after 15 minutes and is scoped to the requesting user's invoice permission. The PDF is a presentation artifact, not the source of truth for amounts; integrations should use the invoice API fields. Regenerating a PDF after a template update does not alter the invoice state or totals.
