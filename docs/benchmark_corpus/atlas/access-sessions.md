# Atlas Access: Sessions and Tokens
## Session lifetime and token claims
An Atlas browser session uses a short-lived access token and a longer-lived refresh token. Access tokens are valid for 15 minutes, while a refresh token can renew a session for up to 30 days unless it is revoked earlier. An idle browser session expires after 60 minutes without activity. Every token carries a subject, organization, session identifier, issued-at time, and expiry time. The session identifier is the useful audit key when an administrator needs to distinguish two sessions belonging to the same user.

## Refresh-token rotation
Every successful refresh request rotates the refresh token. The previous token is retained for a ten-second grace window to tolerate a duplicated mobile request, then it is marked spent. Reuse of an older token outside that window is treated as token theft: Atlas revokes the entire refresh-token family and requires the user to sign in again. A client must replace its stored refresh token atomically after a successful response; retrying with the old value is not a safe refresh strategy.

## Cookie and bearer transport
The web console stores its session in an HttpOnly, Secure cookie with SameSite=Lax. JavaScript cannot read the cookie, and cross-site form submissions do not automatically send it. API clients use an `Authorization: Bearer` header instead and should never place an access token in a URL query parameter. A reverse proxy may terminate TLS, but the application still expects the external request to have been HTTPS so that secure-cookie and redirect decisions remain correct.

## Logout and administrator revocation
Normal logout revokes the current refresh token and deletes the browser cookie; it does not invalidate already issued access tokens before their short expiry. The security page has separate actions for revoking one session and revoking all sessions for a user. “Sign out everywhere” creates a revocation timestamp checked during refresh. Incident responders should use that action after a suspected credential leak rather than merely asking the user to close a browser tab.

## Clock skew and expiry errors
Atlas allows 90 seconds of clock skew when validating `iat`, `nbf`, and `exp` claims. A machine whose clock is more than 90 seconds behind may receive `token_not_yet_valid`; a clock ahead can produce `token_expired` earlier than expected. Clients should synchronize with a trusted time source and treat either error as a signal to refresh or reauthenticate, not as a reason to extend token lifetime. The server's UTC timestamp is authoritative in audit records.
