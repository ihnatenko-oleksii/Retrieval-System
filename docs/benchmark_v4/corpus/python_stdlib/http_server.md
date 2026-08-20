# Python standard library: `http.server`

Official source: https://docs.python.org/3/library/http.server.html

### Command-line interface

http.server can also be invoked directly using the -m switch of the interpreter. The following example illustrates how to serve files relative to the current directory:

python -m http.server [OPTIONS] [port]

The following options are accepted:

port

The server listens to port 8000 by default. The default can be overridden by passing the desired port number as an argument:

python -m http.server 9000

-b, --bind <address>

Specifies a specific address to which it should bind. Both IPv4 and IPv6 addresses are supported. By default, the server binds itself to all interfaces. For example, the following command causes the server to bind to localhost only:

python -m http.server --bind 127.0.0.1

Added in version 3.4.

Changed in version 3.8: Support IPv6 in the --bind option.

-d, --directory <dir>

Specifies a directory to which it should serve the files. By default, the server uses the current directory. For example, the following command uses a specific directory:

python -m http.server --directory /tmp/

Added in version 3.7.

-p, --protocol <version>

Specifies the HTTP version to which the server is conformant. By default, the server is conformant to HTTP/1.0. For example, the following command runs an HTTP/1.1 conformant server:

python -m http.server --protocol HTTP/1.1

Added in version 3.11.

--cgi

CGIHTTPRequestHandler can be enabled in the command line by passing the --cgi option:

python -m http.server --cgi

Deprecated since version 3.13, will be removed in version 3.15: http.server command line --cgi support is being removed because CGIHTTPRequestHandler is being removed.

Warning

CGIHTTPRequestHandler and the --cgi command-line option are not intended for use by untrusted clients and may be vulnerable to exploitation. Always use within a secure environment.

--tls-cert

Specifies a TLS certificate chain for HTTPS connections:

python -m http.server --tls-cert fullchain.pem

Added in version 3.14.

--tls-key

Specifies a private key file for HTTPS connections.

This option requires --tls-cert to be specified.

Added in version 3.14.

--tls-password-file

Specifies the password file for password-protected private keys:

python -m http.server \ --tls-cert cert.pem \ --tls-key key.pem \ --tls-password-file password.txt

This option requires --tls-cert to be specified.

Added in version 3.14.

### Security considerations

SimpleHTTPRequestHandler will follow symbolic links when handling requests which makes it possible for files outside of the specified directory to be served.

Methods BaseHTTPRequestHandler.send_header() and BaseHTTPRequestHandler.send_response_only() assume sanitized input and do not perform input validation such as checking for the presence of CRLF sequences. Untrusted input may result in HTTP header injection attacks.

Earlier versions of Python did not scrub control characters from the log messages emitted to stderr from python -m http.server or the default BaseHTTPRequestHandler .log_message implementation. This could allow remote clients connecting to your server to send nefarious control codes to your terminal.

Changed in version 3.12: Control characters are scrubbed in stderr logs.
