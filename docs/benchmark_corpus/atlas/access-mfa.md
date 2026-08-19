# Atlas Access: Multi-Factor Authentication
## Enrollment methods and assurance levels
An organization can require TOTP, WebAuthn, or SMS as a second factor, although WebAuthn is preferred for phishing resistance. Enrollment begins with a password-authenticated session and ends only after the user verifies one generated TOTP code or one WebAuthn ceremony. SMS enrollment records the verified phone number but is not considered phishing-resistant. The resulting assurance level is recorded on the session so policy checks can distinguish a password-only session from an `aal2` session.

## Step-up authentication
Atlas requests step-up authentication for changing payout details, creating an organization owner, exporting a full customer dataset, or rotating a high-privilege service credential. A successful second factor raises the current session to `aal2` for ten minutes; it does not permanently change the user's baseline login assurance. An API call that needs step-up receives `mfa_required` with the required assurance level. The client should complete the challenge and repeat the original operation with the same idempotency key.

## Recovery codes
MFA enrollment creates ten one-use recovery codes. The codes are displayed once, stored by Atlas as hashes, and cannot be retrieved later from the account page. Using a recovery code consumes it immediately, including when the subsequent password-reset flow fails. A user can generate a new set after authenticating with an existing factor; generation invalidates every unused code from the previous set. Support agents cannot read or manually reveal a recovery code.

## Failed attempts and lockout
The MFA verifier permits five failed attempts in a rolling 15-minute window for one account and factor. A sixth failure returns `mfa_locked` and adds a progressively longer delay before another challenge can start. The lockout is separate from the password lockout and does not reset merely because the user signs out. Security support can clear the factor lock after identity verification, but the action is audited with the agent identity, reason, and ticket reference.

## Remembered devices
“Remember this device” issues a device-bound, HttpOnly cookie that suppresses routine MFA prompts for 30 days. It does not raise the session assurance for sensitive operations, and a remembered device is ignored after a password reset, an all-session revocation, or a risk engine decision. Users can inspect and revoke remembered devices independently from active sessions. Browser privacy modes may discard the cookie, so clients should not treat its absence as an account error.
