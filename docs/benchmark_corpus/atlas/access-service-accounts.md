# Atlas Access: Service Accounts
## Client-credentials tokens
Service accounts authenticate with the OAuth client-credentials grant rather than a human password. A token request names the service account and requested scopes; Atlas issues an access token valid for one hour and does not issue a refresh token. The service must request a new token before expiry and cache it until then. Service-account tokens carry an actor type of `service_account`, so downstream audit records do not mistake an automated job for the employee who created the credential.

## API key storage and prefixes
Legacy integrations may use API keys, but new integrations should use client credentials. An API key is shown exactly once at creation, stored as a salted hash, and identified in the UI by its label and last four characters. Atlas keys begin with the `atk_` prefix, which helps secret scanners recognize them without exposing the secret. The prefix is not an authentication factor by itself; copying a visible prefix into a configuration file does not make a request valid.

## Rotation overlap
Credential rotation supports at most two active secrets for one service account. After a new secret is issued, the old secret remains valid for a 24-hour overlap window so a deployment can roll through its instances. Revoking the old secret immediately ends the overlap. A rotation job should create the new credential, deploy it, verify a harmless authenticated request, and then revoke the old credential instead of deleting both values before rollout completes.

## Scope boundaries
Scopes are granted at organization or project level and are evaluated on every request. `records:read` does not imply `records:write`, and a project-scoped token cannot access a sibling project even when the service account belongs to the same organization. Wildcard scopes are disabled for production accounts. Atlas recommends one service account per workload so that a leaked export credential can be revoked without interrupting ingestion or billing jobs.

## Audit identity
Audit events for service accounts include the service account ID, credential ID, token actor type, project, source IP, and request ID. The human owner appears as a separate `created_by` or `approved_by` field and is not substituted for the runtime actor. Deleting a service account preserves its audit events with a tombstone identity. Investigators should filter on the credential ID when separating two rotated secrets used by the same automation.
