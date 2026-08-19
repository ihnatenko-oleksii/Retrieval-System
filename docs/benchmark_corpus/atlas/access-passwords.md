# Atlas Access: Passwords and Recovery
## Password policy
Atlas passwords require at least 12 characters and reject the 20,000 most common passwords. A password may contain spaces, but leading and trailing whitespace is removed before policy validation. The server stores a memory-hard password hash and never returns the password or its hash through an API. Changing a password invalidates refresh tokens for the current user but leaves the user's audit history intact.

## Reset links
A password-reset request returns the same generic response whether or not an account exists. If an account is eligible, Atlas sends a single-use reset link that expires after 30 minutes. Opening the link does not change the password; the new password is committed only after policy validation and confirmation. Requesting another link invalidates earlier links, so a user should use the newest message rather than retrying an old URL.

## Recovery versus unlock
Password recovery proves control of a verified email or an approved recovery factor and sets a new password. Account unlock clears a temporary password lock without changing the password and requires stronger support verification. A user who sees password_locked should not use a reset link merely to bypass the lock policy. Support actions are separately audited because an unlock and a recovery have different security consequences.

## Breached-password checks
New passwords are checked against a breach corpus using a privacy-preserving prefix lookup; the full candidate is not sent to the breach service. A match returns password_compromised and the password is rejected even when it meets length requirements. The check runs on password creation, password reset, and password change. Existing stored passwords are not checked on every login, but a successful login may trigger a forced reset if a later breach signal matches the account.

## Session consequences
After a password change or recovery, Atlas revokes all refresh-token families and remembered-device cookies for that user. Already issued access tokens remain usable until their short expiry unless the security service has placed an emergency revocation marker. API clients should refresh normally, handle a revoked refresh token by reauthenticating, and avoid sending a password in a retry payload. The security page records the source and time of the change.
