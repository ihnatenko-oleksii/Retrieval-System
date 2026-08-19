"""Create and validate the sealed Phase 4 benchmark-v3 artifact.

The cases are intentionally authored against source sections rather than
ingestion-time chunk IDs. The generated labels contain document-relative
character intervals, so changing chunk size, overlap, or representation does
not require changing the benchmark labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CORPUS_ROOT = REPO_ROOT / "docs" / "benchmark_corpus"
OUTPUT_ROOT = REPO_ROOT / "docs" / "benchmark_v3"


def target(document: str, section_slug: str, gain: int = 3) -> tuple[str, int]:
    return f"{document}#{section_slug}", gain


CASES: list[tuple[str, str, str, list[tuple[str, int]]]] = [
    # Lexical: exact identifiers, codes, numbers, headers, and field names.
    (
        "v3-lexical-01",
        "lexical",
        "Which literal error should invalidate a cursor before another page request is attempted?",
        [target("atlas/api-pagination.md", "invalid-cursors")],
    ),
    (
        "v3-lexical-02",
        "lexical",
        "Name the three response headers that expose the current request allowance.",
        [target("atlas/api-rate-limits.md", "response-headers-and-429-handling")],
    ),
    (
        "v3-lexical-03",
        "lexical",
        "What exact delay formula does Atlas recommend before a retry, and what cap appears in it?",
        [target("atlas/api-retries.md", "backoff-and-attempt-limit")],
    ),
    (
        "v3-lexical-04",
        "lexical",
        "Which header carries the HMAC value that a webhook receiver must verify?",
        [target("atlas/ingestion-webhooks.md", "signature-verification")],
    ),
    (
        "v3-lexical-05",
        "lexical",
        "What prefix distinguishes a newly issued Atlas API key from ordinary text?",
        [target("atlas/access-service-accounts.md", "api-key-storage-and-prefixes")],
    ),
    (
        "v3-lexical-06",
        "lexical",
        "How many seconds is the worker's default visibility lease?",
        [target("atlas/operations-queue.md", "visibility-timeout")],
    ),
    (
        "v3-lexical-07",
        "lexical",
        "Which registry state indicates that producers may use a schema version?",
        [target("atlas/ingestion-schema.md", "registry-versions")],
    ),
    (
        "v3-lexical-08",
        "lexical",
        "What boost is assigned to the title field in the documented lexical ranker?",
        [target("atlas/search-lexical.md", "field-boosts")],
    ),
    (
        "v3-lexical-09",
        "lexical",
        "Which stable key is used as the final tie-break after review timestamps?",
        [target("atlas/search-ranking.md", "deterministic-tie-breaking")],
    ),
    (
        "v3-lexical-10",
        "lexical",
        "What paired values identify the duplicate-detection key for a meter event?",
        [target("atlas/billing-usage.md", "meter-event-acceptance")],
    ),
    (
        "v3-lexical-11",
        "lexical",
        "Which identity pair makes an ingested record distinct from another record?",
        [target("atlas/ingestion-dedup.md", "record-identity")],
    ),
    (
        "v3-lexical-12",
        "lexical",
        "What status code represents a conflict caused by an already existing state?",
        [target("atlas/api-errors.md", "conflict-and-precondition-errors")],
    ),
    (
        "v3-lexical-13",
        "lexical",
        "For how many days are completed export files retained?",
        [target("atlas/operations-exports.md", "file-retention-and-urls")],
    ),
    (
        "v3-lexical-14",
        "lexical",
        "Which timezone representation must a scheduled workflow store?",
        [target("atlas/workflows-triggers.md", "schedule-timezones-and-dst")],
    ),
    (
        "v3-lexical-15",
        "lexical",
        "Which telemetry label identifies a database pool acquisition wait?",
        [target("atlas/operations-database.md", "connection-pools")],
    ),
    (
        "v3-lexical-16",
        "lexical",
        "Are date-filter boundaries inclusive at the beginning and exclusive at the end?",
        [target("atlas/search-filters.md", "date-boundaries")],
    ),
    (
        "v3-lexical-17",
        "lexical",
        "How many failed MFA attempts are allowed in the rolling lockout window?",
        [target("atlas/access-mfa.md", "failed-attempts-and-lockout")],
    ),
    (
        "v3-lexical-18",
        "lexical",
        "Which unit is used when Atlas represents a monetary amount for storage?",
        [target("atlas/billing-payments.md", "currency-and-rounding")],
    ),
    (
        "v3-lexical-19",
        "lexical",
        "What typed path syntax is used to reference workflow inputs and outputs?",
        [target("atlas/workflows-conditions.md", "input-and-output-references")],
    ),
    (
        "v3-lexical-20",
        "lexical",
        "What does a credit note do to an invoice balance?",
        [target("atlas/billing-invoices.md", "credit-notes-and-refunds")],
    ),
    # Semantic: paraphrased information needs meaning beyond exact overlap.
    (
        "v3-semantic-01",
        "semantic",
        "If a write request may have reached the server before a timeout, how can the caller safely discover its outcome?",
        [target("atlas/api-retries.md", "duplicate-outcome-recovery")],
    ),
    (
        "v3-semantic-02",
        "semantic",
        "Where are browser credentials kept so ordinary page scripts cannot read them?",
        [target("atlas/access-sessions.md", "cookie-and-bearer-transport")],
    ),
    (
        "v3-semantic-03",
        "semantic",
        "Why can a multilingual search request still find a passage written with different wording?",
        [target("atlas/search-semantic.md", "multilingual-and-paraphrase-behavior")],
    ),
    (
        "v3-semantic-04",
        "semantic",
        "When a bulk import contains bad rows, what happens to the rows that passed validation?",
        [target("atlas/ingestion-batch.md", "per-record-validation")],
    ),
    (
        "v3-semantic-05",
        "semantic",
        "What evidence should an operator use to decide whether a lagging replica needs attention?",
        [target("atlas/operations-database.md", "replica-lag")],
    ),
    (
        "v3-semantic-06",
        "semantic",
        "How is an approval request represented while it is still waiting for a decision?",
        [target("atlas/workflows-approvals.md", "approval-states")],
    ),
    (
        "v3-semantic-07",
        "semantic",
        "Why should a responder not call an incident resolved merely because customer impact has decreased?",
        [target("atlas/operations-incidents.md", "mitigation-versus-resolution")],
    ),
    (
        "v3-semantic-08",
        "semantic",
        "How does the credit system preserve the history of grants and consumption instead of overwriting a balance?",
        [target("atlas/billing-credits.md", "credit-ledger-and-expiry")],
    ),
    (
        "v3-semantic-09",
        "semantic",
        "How can a caller keep seeing one consistent collection while records are being added?",
        [target("atlas/api-pagination.md", "snapshot-consistency")],
    ),
    (
        "v3-semantic-10",
        "semantic",
        "Why is a typo-tolerant search mode kept separate from the normal precise identifier search?",
        [target("atlas/search-lexical.md", "fuzzy-matching")],
    ),
    (
        "v3-semantic-11",
        "semantic",
        "How can one expensive operation consume more allowance than one network request?",
        [target("atlas/api-rate-limits.md", "weighted-request-cost")],
    ),
    (
        "v3-semantic-12",
        "semantic",
        "Which kind of schema edit can be deployed without forcing old producers to change immediately?",
        [target("atlas/ingestion-schema.md", "compatible-and-breaking-changes")],
    ),
    (
        "v3-semantic-13",
        "semantic",
        "When should an export consumer prefer a consistent boundary over the newest individual rows?",
        [target("atlas/operations-exports.md", "consistency-modes")],
    ),
    (
        "v3-semantic-14",
        "semantic",
        "How does the workflow engine distinguish a temporary infrastructure problem from a permanent task defect?",
        [target("atlas/workflows-retries.md", "failure-classification")],
    ),
    (
        "v3-semantic-15",
        "semantic",
        "How does account recovery differ when the user forgot a secret from when the account is merely locked?",
        [target("atlas/access-passwords.md", "recovery-versus-unlock")],
    ),
    (
        "v3-semantic-16",
        "semantic",
        "What stages does an invoice pass through before it represents an amount that is due or settled?",
        [target("atlas/billing-invoices.md", "invoice-lifecycle")],
    ),
    (
        "v3-semantic-17",
        "semantic",
        "What sequence does automatic collection follow after an invoice payment attempt does not succeed?",
        [target("atlas/billing-payments.md", "collection-retries")],
    ),
    (
        "v3-semantic-18",
        "semantic",
        "How far back can an operator ask the webhook console to resend a delivery?",
        [target("atlas/ingestion-webhooks.md", "replay-window")],
    ),
    (
        "v3-semantic-19",
        "semantic",
        "Why might a ranking service deliberately retrieve more passages before applying a pairwise judge?",
        [target("atlas/search-ranking.md", "candidate-pool-and-cross-encoder")],
    ),
    (
        "v3-semantic-20",
        "semantic",
        "What makes a repeatedly failing queue message different from a normal transient failure?",
        [target("atlas/operations-queue.md", "poison-messages")],
    ),
    # Ambiguous: broad wording with several plausible, separately relevant sources.
    (
        "v3-ambiguous-01",
        "ambiguous",
        "When should an Atlas operation be attempted again, and when should it be left alone?",
        [
            target("atlas/api-retries.md", "retryable-responses"),
            target("atlas/workflows-retries.md", "failure-classification", 2),
            target("atlas/billing-payments.md", "collection-retries", 2),
        ],
    ),
    (
        "v3-ambiguous-02",
        "ambiguous",
        "What does it mean for an Atlas thing to expire?",
        [
            target("atlas/access-sessions.md", "session-lifetime-and-token-claims"),
            target("atlas/billing-credits.md", "credit-ledger-and-expiry", 2),
            target("atlas/operations-exports.md", "file-retention-and-urls", 2),
        ],
    ),
    (
        "v3-ambiguous-03",
        "ambiguous",
        "How does Atlas decide whether two activities are the same?",
        [
            target("atlas/ingestion-dedup.md", "record-identity"),
            target("atlas/ingestion-webhooks.md", "event-ids-and-ordering", 2),
            target("atlas/api-retries.md", "idempotency-keys", 2),
        ],
    ),
    (
        "v3-ambiguous-04",
        "ambiguous",
        "What should a client do when the platform says it cannot accept more right now?",
        [
            target("atlas/api-rate-limits.md", "response-headers-and-429-handling"),
            target("atlas/api-errors.md", "rate-and-server-errors", 2),
            target("atlas/operations-queue.md", "visibility-timeout", 2),
        ],
    ),
    (
        "v3-ambiguous-05",
        "ambiguous",
        "How are time boundaries interpreted across Atlas features?",
        [
            target("atlas/search-filters.md", "date-boundaries"),
            target("atlas/billing-usage.md", "aggregation-windows-and-late-events", 2),
            target("atlas/workflows-triggers.md", "schedule-timezones-and-dst", 2),
        ],
    ),
    (
        "v3-ambiguous-06",
        "ambiguous",
        "What does a successful submission actually prove?",
        [
            target("atlas/ingestion-batch.md", "batch-submission-and-limits"),
            target("atlas/operations-exports.md", "job-creation-and-formats", 2),
            target("atlas/billing-payments.md", "payment-attempts", 2),
        ],
    ),
    (
        "v3-ambiguous-07",
        "ambiguous",
        "Who is allowed to take the next action?",
        [
            target("atlas/access-service-accounts.md", "scope-boundaries"),
            target("atlas/operations-exports.md", "export-permissions", 2),
            target("atlas/workflows-approvals.md", "approver-eligibility", 2),
        ],
    ),
    (
        "v3-ambiguous-08",
        "ambiguous",
        "Which failures are client mistakes and which can recover on their own?",
        [
            target("atlas/api-errors.md", "client-and-authorization-errors"),
            target("atlas/api-retries.md", "retryable-responses", 2),
            target("atlas/workflows-retries.md", "failure-classification", 2),
        ],
    ),
    (
        "v3-ambiguous-09",
        "ambiguous",
        "How does Atlas make repeating an event safe?",
        [
            target("atlas/api-retries.md", "idempotency-keys"),
            target("atlas/ingestion-webhooks.md", "event-ids-and-ordering", 2),
            target("atlas/workflows-triggers.md", "trigger-deduplication", 2),
        ],
    ),
    (
        "v3-ambiguous-10",
        "ambiguous",
        "What is a limit in Atlas, and what exactly is being limited?",
        [
            target("atlas/api-rate-limits.md", "limit-dimensions"),
            target("atlas/api-pagination.md", "page-size-defaults-and-limits", 2),
            target("atlas/workflows-retries.md", "concurrency-and-queues", 2),
        ],
    ),
    (
        "v3-ambiguous-11",
        "ambiguous",
        "How can an operator recover something after a process appears to have failed?",
        [
            target("atlas/api-retries.md", "duplicate-outcome-recovery"),
            target("atlas/operations-exports.md", "file-retention-and-urls", 2),
            target("atlas/operations-database.md", "backups-and-restore-tests", 2),
        ],
    ),
    (
        "v3-ambiguous-12",
        "ambiguous",
        "What kind of evidence makes a search result trustworthy?",
        [
            target("atlas/search-lexical.md", "when-lexical-search-is-the-right-tool"),
            target("atlas/search-semantic.md", "meaning-beyond-exact-terms", 2),
            target("atlas/search-ranking.md", "explainability", 2),
        ],
    ),
    (
        "v3-ambiguous-13",
        "ambiguous",
        "How does Atlas keep a financial state understandable after changes?",
        [
            target("atlas/billing-invoices.md", "invoice-lifecycle"),
            target("atlas/billing-payments.md", "refunds-and-reversals", 2),
            target("atlas/billing-credits.md", "credit-audit-events", 2),
        ],
    ),
    (
        "v3-ambiguous-14",
        "ambiguous",
        "What can a process be waiting for?",
        [
            target("atlas/workflows-approvals.md", "timeouts-and-reassignment"),
            target("atlas/operations-queue.md", "visibility-timeout", 2),
            target("atlas/operations-database.md", "migration-locks", 2),
        ],
    ),
    (
        "v3-ambiguous-15",
        "ambiguous",
        "What changes when an older definition or record is no longer current?",
        [
            target("atlas/ingestion-schema.md", "registry-versions"),
            target("atlas/ingestion-dedup.md", "corrections", 2),
            target("atlas/workflows-conditions.md", "versioned-definitions", 2),
        ],
    ),
    (
        "v3-ambiguous-16",
        "ambiguous",
        "How are long-running jobs made manageable?",
        [
            target("atlas/operations-exports.md", "job-creation-and-formats"),
            target("atlas/ingestion-batch.md", "batch-submission-and-limits", 2),
            target("atlas/api-retries.md", "connect-read-and-total-timeouts", 2),
        ],
    ),
    (
        "v3-ambiguous-17",
        "ambiguous",
        "How does Atlas prevent an incoming event from causing uncontrolled work?",
        [
            target("atlas/ingestion-webhooks.md", "delivery-schedule-and-acknowledgement"),
            target("atlas/workflows-triggers.md", "event-filters", 2),
            target("atlas/operations-queue.md", "poison-messages", 2),
        ],
    ),
    (
        "v3-ambiguous-18",
        "ambiguous",
        "What makes a second attempt safe rather than dangerous?",
        [
            target("atlas/api-retries.md", "idempotency-keys"),
            target("atlas/workflows-retries.md", "compensation-versus-retry", 2),
            target("atlas/billing-payments.md", "collection-retries", 2),
        ],
    ),
    (
        "v3-ambiguous-19",
        "ambiguous",
        "Which Atlas actions can be replayed, and what protects them from duplication?",
        [
            target("atlas/ingestion-webhooks.md", "replay-window"),
            target("atlas/workflows-retries.md", "manual-replay", 2),
            target("atlas/ingestion-dedup.md", "reconciliation", 2),
        ],
    ),
    (
        "v3-ambiguous-20",
        "ambiguous",
        "How does Atlas keep a read aligned with the state it is supposed to represent?",
        [
            target("atlas/api-pagination.md", "snapshot-consistency"),
            target("atlas/operations-database.md", "replica-lag", 2),
            target("atlas/operations-exports.md", "consistency-modes", 2),
        ],
    ),
    # Fine-grained: near-neighbor distinctions and boundary questions.
    (
        "v3-fine-01",
        "fine_grained",
        "How does an invalid pagination cursor differ from an expired export download URL?",
        [
            target("atlas/api-pagination.md", "invalid-cursors"),
            target("atlas/operations-exports.md", "file-retention-and-urls"),
        ],
    ),
    (
        "v3-fine-02",
        "fine_grained",
        "What rotates on refresh, and what instead governs how long the browser session may remain alive?",
        [
            target("atlas/access-sessions.md", "refresh-token-rotation"),
            target("atlas/access-sessions.md", "session-lifetime-and-token-claims"),
        ],
    ),
    (
        "v3-fine-03",
        "fine_grained",
        "How should a client distinguish an API read timeout from a queue worker visibility deadline?",
        [
            target("atlas/api-retries.md", "connect-read-and-total-timeouts"),
            target("atlas/operations-queue.md", "visibility-timeout"),
        ],
    ),
    (
        "v3-fine-04",
        "fine_grained",
        "Why is a 429 retry delay different from the period before a monthly quota renews?",
        [
            target("atlas/api-rate-limits.md", "response-headers-and-429-handling"),
            target("atlas/api-rate-limits.md", "rate-limit-versus-quota"),
        ],
    ),
    (
        "v3-fine-05",
        "fine_grained",
        "Does an invoice credit note reverse a payment in the same way as a refund?",
        [
            target("atlas/billing-invoices.md", "credit-notes-and-refunds"),
            target("atlas/billing-payments.md", "refunds-and-reversals"),
        ],
    ),
    (
        "v3-fine-06",
        "fine_grained",
        "What does an accepted batch submission establish compared with later per-record validation?",
        [
            target("atlas/ingestion-batch.md", "batch-submission-and-limits"),
            target("atlas/ingestion-batch.md", "per-record-validation"),
        ],
    ),
    (
        "v3-fine-07",
        "fine_grained",
        "When should a receiver use an event ID for deduplication instead of an idempotency key?",
        [
            target("atlas/ingestion-webhooks.md", "event-ids-and-ordering"),
            target("atlas/api-retries.md", "idempotency-keys"),
        ],
    ),
    (
        "v3-fine-08",
        "fine_grained",
        "How is a field alias different from a schema change that breaks an existing producer?",
        [
            target("atlas/ingestion-schema.md", "field-aliases-and-mapping"),
            target("atlas/ingestion-schema.md", "compatible-and-breaking-changes"),
        ],
    ),
    (
        "v3-fine-09",
        "fine_grained",
        "Why are replica lag and export snapshot consistency separate concerns?",
        [
            target("atlas/operations-database.md", "replica-lag"),
            target("atlas/operations-exports.md", "consistency-modes"),
        ],
    ),
    (
        "v3-fine-10",
        "fine_grained",
        "What is collected during debounce, and how is that different from cooldown after a trigger?",
        [
            target("atlas/workflows-triggers.md", "debounce-and-cooldown"),
            target("atlas/workflows-triggers.md", "trigger-types"),
        ],
    ),
    (
        "v3-fine-11",
        "fine_grained",
        "Can approval reassignment change the original timeout in the way token expiry changes access?",
        [
            target("atlas/workflows-approvals.md", "timeouts-and-reassignment"),
            target("atlas/access-sessions.md", "clock-skew-and-expiry-errors"),
        ],
    ),
    (
        "v3-fine-12",
        "fine_grained",
        "When would an export use a snapshot boundary rather than merely read the latest available rows?",
        [
            target("atlas/operations-exports.md", "consistency-modes"),
            target("atlas/operations-exports.md", "large-export-parts"),
        ],
    ),
    (
        "v3-fine-13",
        "fine_grained",
        "How is a negative billing balance different from a promotional credit that has not expired?",
        [
            target("atlas/billing-credits.md", "negative-balances-and-overage"),
            target("atlas/billing-credits.md", "promotional-and-prepaid-credits"),
        ],
    ),
    (
        "v3-fine-14",
        "fine_grained",
        "When should a workflow compensate instead of retrying the task that failed?",
        [
            target("atlas/workflows-retries.md", "compensation-versus-retry"),
            target("atlas/workflows-retries.md", "failure-classification"),
        ],
    ),
    (
        "v3-fine-15",
        "fine_grained",
        "What does quoted phrase matching preserve that fuzzy identifier matching intentionally does not?",
        [
            target("atlas/search-lexical.md", "phrase-and-punctuation-handling"),
            target("atlas/search-lexical.md", "fuzzy-matching"),
        ],
    ),
    (
        "v3-fine-16",
        "fine_grained",
        "Why can a candidate-pool size change matter to a reranker even when the embedding similarity is unchanged?",
        [
            target("atlas/search-ranking.md", "candidate-pool-and-cross-encoder"),
            target("atlas/search-semantic.md", "similarity-and-candidate-recall"),
        ],
    ),
    (
        "v3-fine-17",
        "fine_grained",
        "How does a migration lock timeout differ from the recovery target for a database failover?",
        [
            target("atlas/operations-database.md", "migration-locks"),
            target("atlas/operations-database.md", "failover-objectives"),
        ],
    ),
    (
        "v3-fine-18",
        "fine_grained",
        "Why is a 409 conflict not handled like a retryable 503 response?",
        [
            target("atlas/api-errors.md", "conflict-and-precondition-errors"),
            target("atlas/api-errors.md", "rate-and-server-errors"),
        ],
    ),
    (
        "v3-fine-19",
        "fine_grained",
        "Does the exclusive end of a date filter have the same meaning as the end of a usage aggregation window?",
        [
            target("atlas/search-filters.md", "date-boundaries"),
            target("atlas/billing-usage.md", "aggregation-windows-and-late-events"),
        ],
    ),
    (
        "v3-fine-20",
        "fine_grained",
        "How does administrator revocation differ from rotating a refresh token?",
        [
            target("atlas/access-sessions.md", "logout-and-administrator-revocation"),
            target("atlas/access-sessions.md", "refresh-token-rotation"),
        ],
    ),
    # Multiple relevant: the answer requires several independent passages.
    (
        "v3-multi-01",
        "multiple_relevant",
        "How should a client submit, retry, and reconcile a bulk import when the first response disappears?",
        [
            target("atlas/ingestion-batch.md", "batch-submission-and-limits"),
            target("atlas/ingestion-batch.md", "safe-batch-retries"),
            target("atlas/api-retries.md", "duplicate-outcome-recovery"),
        ],
    ),
    (
        "v3-multi-02",
        "multiple_relevant",
        "How do metered usage, invoice state, and collection behavior combine into a billing outcome?",
        [
            target("atlas/billing-usage.md", "aggregation-windows-and-late-events"),
            target("atlas/billing-invoices.md", "invoice-lifecycle"),
            target("atlas/billing-payments.md", "collection-retries"),
        ],
    ),
    (
        "v3-multi-03",
        "multiple_relevant",
        "How should search balance literal codes, meaning-level similarity, and an explainable hybrid score?",
        [
            target("atlas/search-lexical.md", "when-lexical-search-is-the-right-tool"),
            target("atlas/search-semantic.md", "meaning-beyond-exact-terms"),
            target("atlas/search-ranking.md", "hybrid-score-fusion"),
        ],
    ),
    (
        "v3-multi-04",
        "multiple_relevant",
        "What must a webhook receiver verify before acknowledging, deduplicating, and replaying an event?",
        [
            target("atlas/ingestion-webhooks.md", "signature-verification"),
            target("atlas/ingestion-webhooks.md", "delivery-schedule-and-acknowledgement"),
            target("atlas/ingestion-webhooks.md", "event-ids-and-ordering"),
            target("atlas/ingestion-webhooks.md", "replay-window", 2),
        ],
    ),
    (
        "v3-multi-05",
        "multiple_relevant",
        "How do trigger filtering, condition evaluation, task retries, and approvals shape one workflow run?",
        [
            target("atlas/workflows-triggers.md", "event-filters"),
            target("atlas/workflows-conditions.md", "condition-evaluation"),
            target("atlas/workflows-retries.md", "attempts-and-backoff"),
            target("atlas/workflows-approvals.md", "approval-states", 2),
        ],
    ),
    (
        "v3-multi-06",
        "multiple_relevant",
        "What should operators examine when database failover and queued work occur during a reliability incident?",
        [
            target("atlas/operations-database.md", "replica-lag"),
            target("atlas/operations-database.md", "failover-objectives"),
            target("atlas/operations-queue.md", "queue-lag-and-consumer-health"),
            target("atlas/operations-incidents.md", "timeline-events", 2),
        ],
    ),
    (
        "v3-multi-07",
        "multiple_relevant",
        "How do export creation, authorization, retention, and multipart delivery fit together?",
        [
            target("atlas/operations-exports.md", "job-creation-and-formats"),
            target("atlas/operations-exports.md", "export-permissions"),
            target("atlas/operations-exports.md", "file-retention-and-urls"),
            target("atlas/operations-exports.md", "large-export-parts", 2),
        ],
    ),
    (
        "v3-multi-08",
        "multiple_relevant",
        "How can a schema rollout preserve compatibility while mapping aliases and handling invalid records?",
        [
            target("atlas/ingestion-schema.md", "registry-versions"),
            target("atlas/ingestion-schema.md", "compatible-and-breaking-changes"),
            target("atlas/ingestion-schema.md", "field-aliases-and-mapping"),
            target("atlas/ingestion-schema.md", "validation-modes", 2),
        ],
    ),
    (
        "v3-multi-09",
        "multiple_relevant",
        "What controls keep a service account's secret, scopes, rotation window, and audit trail aligned?",
        [
            target("atlas/access-service-accounts.md", "client-credentials-tokens"),
            target("atlas/access-service-accounts.md", "rotation-overlap"),
            target("atlas/access-service-accounts.md", "scope-boundaries"),
            target("atlas/access-service-accounts.md", "audit-identity", 2),
        ],
    ),
    (
        "v3-multi-10",
        "multiple_relevant",
        "How should access recovery combine password reset, MFA recovery codes, and revocation of old sessions?",
        [
            target("atlas/access-passwords.md", "reset-links"),
            target("atlas/access-mfa.md", "recovery-codes"),
            target("atlas/access-sessions.md", "logout-and-administrator-revocation"),
        ],
    ),
    (
        "v3-multi-11",
        "multiple_relevant",
        "How should an API client classify errors, obey rate responses, and preserve a trace for support?",
        [
            target("atlas/api-errors.md", "error-envelope"),
            target("atlas/api-errors.md", "client-and-authorization-errors"),
            target("atlas/api-rate-limits.md", "response-headers-and-429-handling"),
            target("atlas/api-errors.md", "request-tracing", 2),
        ],
    ),
    (
        "v3-multi-12",
        "multiple_relevant",
        "How do record identity, upsert policy, corrections, and reconciliation protect an ingestion pipeline?",
        [
            target("atlas/ingestion-dedup.md", "record-identity"),
            target("atlas/ingestion-dedup.md", "upsert-and-create-only-modes"),
            target("atlas/ingestion-dedup.md", "corrections"),
            target("atlas/ingestion-dedup.md", "reconciliation", 2),
        ],
    ),
    (
        "v3-multi-13",
        "multiple_relevant",
        "How are request dimensions, response headers, weighted operations, and quota periods related?",
        [
            target("atlas/api-rate-limits.md", "limit-dimensions"),
            target("atlas/api-rate-limits.md", "response-headers-and-429-handling"),
            target("atlas/api-rate-limits.md", "weighted-request-cost"),
            target("atlas/api-rate-limits.md", "rate-limit-versus-quota", 2),
        ],
    ),
    (
        "v3-multi-14",
        "multiple_relevant",
        "How should a queue operator handle lag, poison messages, lease expiry, and orderly draining?",
        [
            target("atlas/operations-queue.md", "queue-lag-and-consumer-health"),
            target("atlas/operations-queue.md", "poison-messages"),
            target("atlas/operations-queue.md", "visibility-timeout"),
            target("atlas/operations-queue.md", "draining-and-shutdown", 2),
        ],
    ),
    (
        "v3-multi-15",
        "multiple_relevant",
        "What information should an incident record capture from declaration through review and follow-up?",
        [
            target("atlas/operations-incidents.md", "severity-and-ownership"),
            target("atlas/operations-incidents.md", "timeline-events"),
            target("atlas/operations-incidents.md", "communication-cadence"),
            target("atlas/operations-incidents.md", "post-incident-review", 2),
        ],
    ),
    (
        "v3-multi-16",
        "multiple_relevant",
        "How should a filtered search explain which records qualified before semantic ordering and final ranking?",
        [
            target("atlas/search-filters.md", "structured-filters"),
            target("atlas/search-filters.md", "date-boundaries"),
            target("atlas/search-filters.md", "filter-and-semantic-ordering"),
            target("atlas/search-filters.md", "filter-explanations", 2),
        ],
    ),
    (
        "v3-multi-17",
        "multiple_relevant",
        "How do approval state, quorum, eligibility, and reassignment determine whether a workflow can proceed?",
        [
            target("atlas/workflows-approvals.md", "approval-states"),
            target("atlas/workflows-approvals.md", "quorum-rules"),
            target("atlas/workflows-approvals.md", "approver-eligibility"),
            target("atlas/workflows-approvals.md", "timeouts-and-reassignment", 2),
        ],
    ),
    (
        "v3-multi-18",
        "multiple_relevant",
        "How should a durable asynchronous operation connect submission, status, and delivery?",
        [
            target("atlas/operations-exports.md", "job-creation-and-formats"),
            target("atlas/ingestion-batch.md", "batch-submission-and-limits"),
            target("atlas/ingestion-webhooks.md", "delivery-schedule-and-acknowledgement"),
        ],
    ),
    (
        "v3-multi-19",
        "multiple_relevant",
        "How do payment attempts, collection retries, reversals, and audit records explain a customer's final charge state?",
        [
            target("atlas/billing-payments.md", "payment-attempts"),
            target("atlas/billing-payments.md", "collection-retries"),
            target("atlas/billing-payments.md", "refunds-and-reversals"),
            target("atlas/billing-payments.md", "payment-audit", 2),
        ],
    ),
    (
        "v3-multi-20",
        "multiple_relevant",
        "How can versioned workflow definitions and schema migration jobs change future execution without rewriting history?",
        [
            target("atlas/workflows-conditions.md", "versioned-definitions"),
            target("atlas/ingestion-schema.md", "migration-jobs"),
            target("atlas/ingestion-dedup.md", "corrections", 2),
        ],
    ),
]


def _corpus_sha256(corpus_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in corpus_root.rglob("*") if path.is_file() and not path.name.startswith(".")):
        digest.update(path.relative_to(corpus_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _question_sha256(cases: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{case['id']}\t{case['question']}" for case in cases)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def _load_questions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        _normalise_question(str(json.loads(line).get("question", "")))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_cases(spans: list[Any]) -> list[dict[str, Any]]:
    by_id = {span.span_id: span for span in spans}
    if len(by_id) != len(spans):
        raise ValueError("Source span IDs are not unique")
    cases: list[dict[str, Any]] = []
    for case_id, category, question, raw_targets in CASES:
        relevance_spans: list[dict[str, Any]] = []
        for span_id, gain in raw_targets:
            span = by_id.get(span_id)
            if span is None:
                raise ValueError(f"Unknown source span target: {span_id}")
            relevance_spans.append({**span.as_dict(), "gain": gain})
        cases.append(
            {
                "id": case_id,
                "category": category,
                "question": question,
                "relevance_spans": relevance_spans,
            }
        )
    return cases


def validate_cases(cases: list[dict[str, Any]], spans: list[Any]) -> None:
    if len(cases) < 100:
        raise ValueError(f"benchmark-v3 requires at least 100 cases, got {len(cases)}")
    ids = [str(case.get("id", "")) for case in cases]
    if len(ids) != len(set(ids)) or any(not case_id.startswith("v3-") for case_id in ids):
        raise ValueError("benchmark-v3 case IDs must be unique and start with v3-")
    categories = Counter(str(case.get("category", "")) for case in cases)
    required = {"lexical", "semantic", "ambiguous", "fine_grained", "multiple_relevant"}
    if set(categories) != required or min(categories.values()) < 20:
        raise ValueError(f"benchmark-v3 category balance is invalid: {dict(categories)}")

    known_spans = {span.span_id: span for span in spans}
    seen_questions: set[str] = set()
    legacy_questions = _load_questions(REPO_ROOT / "docs" / "benchmark_eval.jsonl")
    legacy_questions |= _load_questions(REPO_ROOT / "docs" / "benchmark_v2" / "eval.jsonl")
    for case in cases:
        question = _normalise_question(str(case.get("question", "")))
        if not question or question in seen_questions:
            raise ValueError(f"benchmark-v3 question is empty or duplicated: {case.get('id')}")
        if question in legacy_questions:
            raise ValueError(f"benchmark-v3 question duplicates an observed benchmark: {case.get('id')}")
        seen_questions.add(question)

        labels = case.get("relevance_spans")
        if not isinstance(labels, list) or len(labels) == 0:
            raise ValueError(f"case has no source-span labels: {case.get('id')}")
        for label in labels:
            span_id = str(label.get("span_id", ""))
            span = known_spans.get(span_id)
            if span is None:
                raise ValueError(f"case points to an unknown span: {case.get('id')} -> {span_id}")
            if (
                label.get("document_id") != span.document_id
                or label.get("start") != span.start
                or label.get("end") != span.end
            ):
                raise ValueError(f"case label does not match the sealed span catalog: {case.get('id')} -> {span_id}")


def _write_artifact(output_root: Path, cases: list[dict[str, Any]], spans: list[Any]) -> None:
    from app.evals.span_relevance import span_catalog_sha256

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "eval.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n" for case in cases),
        encoding="utf-8",
    )
    (output_root / "spans.json").write_text(
        json.dumps([span.as_dict() for span in spans], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    categories = Counter(str(case["category"]) for case in cases)
    manifest = {
        "protocol_version": 1,
        "benchmark": "benchmark-v3",
        "selection_eligible": False,
        "sealed_before_phase4_selection": True,
        "source_corpus": "docs/benchmark_corpus",
        "corpus_sha256": _corpus_sha256(CORPUS_ROOT),
        "span_catalog_sha256": span_catalog_sha256(spans),
        "question_sha256": _question_sha256(cases),
        "case_count": len(cases),
        "category_counts": dict(sorted(categories.items())),
        "case_ids": [str(case["id"]) for case in cases],
        "label_protocol": "graded source-section intervals; a retrieved chunk is relevant when its source interval overlaps a labeled interval",
        "selection_data": "not used for Phase 4 architecture or parameter selection",
        "final_scoring": "run exactly once for the original E5 dense baseline, frozen Phase 3 final, and frozen Phase 4 final",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from app.evals.span_relevance import load_markdown_spans, span_catalog_sha256

    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    spans = load_markdown_spans(CORPUS_ROOT)
    cases = build_cases(spans)
    validate_cases(cases, spans)
    if not args.validate_only:
        _write_artifact(OUTPUT_ROOT, cases, spans)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "categories": dict(sorted(Counter(case["category"] for case in cases).items())),
                "spans": len(spans),
                "span_catalog_sha256": span_catalog_sha256(spans),
                "mode": "validate" if args.validate_only else "write",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
