# Python standard library: `http.client`

Official source: https://docs.python.org/3/library/http.client.html

### HTTPConnection Objects

HTTPConnection instances have the following methods:

HTTPConnection.request(method, url, body=None, headers={}, *, encode_chunked=False)

This will send a request to the server using the HTTP request method method and the request URI url. The provided url must be an absolute path to conform with RFC 2616 §5.1.2 (unless connecting to an HTTP proxy server or using the OPTIONS or CONNECT methods).

If body is specified, the specified data is sent after the headers are finished. It may be a str, a bytes-like object, an open file object, or an iterable of bytes. If body is a string, it is encoded as ISO-8859-1, the default for HTTP. If it is a bytes-like object, the bytes are sent as is. If it is a file object, the contents of the file is sent; this file object should support at least the read() method. If the file object is an instance of io.TextIOBase, the data returned by the read() method will be encoded as ISO-8859-1, otherwise the data returned by read() is sent as is. If body is an iterable, the elements of the iterable are sent as is until the iterable is exhausted.

The headers argument should be a mapping of extra HTTP headers to send with the request. A Host header must be provided to conform with RFC 2616 §5.1.2 (unless connecting to an HTTP proxy server or using the OPTIONS or CONNECT methods).

If headers contains neither Content-Length nor Transfer-Encoding, but there is a request body, one of those header fields will be added automatically. If body is None, the Content-Length header is set to 0 for methods that expect a body (PUT, POST, and PATCH). If body is a string or a bytes-like object that is not also a file, the Content-Length header is set to its length. Any other type of body (files and iterables in general) will be chunk-encoded, and the Transfer-Encoding header will automatically be set instead of Content-Length.

The encode_chunked argument is only relevant if Transfer-Encoding is specified in headers. If encode_chunked is False, the HTTPConnection object assumes that all encoding is handled by the calling code. If it is True, the body will be chunk-encoded.

For example, to perform a GET request to https://docs.python.org/3/:

>>> import http.client >>> host = "docs.python.org" >>> conn = http.client.HTTPSConnection(host) >>> conn.request("GET", "/3/", headers={"Host": host}) >>> response = conn.getresponse() >>> print(response.status, response.reason) 200 OK

Note

Chunked transfer encoding has been added to the HTTP protocol version 1.1. Unless the HTTP server is known to handle HTTP 1.1, the caller must either specify the Content-Length, or must pass a str or bytes-like object that is not also a file as the body representation.

Note

Note that you must have read the whole response or call close() if getresponse() raised an non-ConnectionError exception before you can send a new request to the server.

Changed in version 3.2: body can now be an iterable.

Changed in version 3.6: If neither Content-Length nor Transfer-Encoding are set in headers, file and iterable body objects are now chunk-encoded. The encode_chunked argument was added. No attempt is made to determine the Content-Length for file objects.

HTTPConnection.getresponse()

Should be called after a request is sent to get the response from the server. Returns an HTTPResponse instance.

Changed in version 3.5: If a ConnectionError or subclass is raised, the HTTPConnection object will be ready to reconnect when a new request is sent.

Note that this does not apply to OSErrors raised by the underlying socket. Instead the caller is responsible to call close() on the existing connection.

HTTPConnection.set_debuglevel(level)

Set the debugging level. The default debug level is 0, meaning no debugging output is printed. Any value greater than 0 will cause all currently defined debug output to be printed to stdout. The debuglevel is passed to any new HTTPResponse objects that are created.

Added in version 3.1.

HTTPConnection.set_tunnel(host, port=None, headers=None)

Set the host and the port for HTTP Connect Tunnelling. This allows running the connection through a proxy server.

The host and port arguments specify the endpoint of the tunneled connection (i.e. the address included in the CONNECT request, not the address of the proxy server).

The headers argument should be a mapping of extra HTTP headers to send with the CONNECT request.

As HTTP/1.1 is used for HTTP CONNECT tunnelling request, as per the RFC, a HTTP Host: header must be provided, matching the authority-form of the request target provided as the destination for the CONNECT request. If a HTTP Host: header is not provided via the headers argument, one is generated and transmitted automatically.

For example, to tunnel through a HTTPS proxy server running locally on port 8080, we would pass the address of the proxy to the HTTPSConnection constructor, and the address of the host that we eventually want to reach to the set_tunnel() method:

>>> import http.client >>> conn = http.client.HTTPSConnection("localhost", 8080) >>> conn.set_tunnel("www.python.org") >>> conn.request("HEAD","/index.html")

Added in version 3.2.

Changed in version 3.12: HTTP CONNECT tunnelling requests use protocol HTTP/1.1, upgraded from protocol HTTP/1.0. Host: HTTP headers are mandatory for HTTP/1.1, so one will be automatically generated and transmitted if not provided in the headers argument.

HTTPConnection.get_proxy_response_headers()

Returns a dictionary with the headers of the response received from the proxy server to the CONNECT request.

If the CONNECT request was not sent, the method returns None.

Added in version 3.12.

HTTPConnection.connect()

Connect to the server specified when the object was created. By default, this is called automatically when making a request if the client does not already have a connection.

Raises an auditing event http.client.connect with arguments self, host, port.

HTTPConnection.close()

Close the connection to the server.

HTTPConnection.blocksize

Buffer size in bytes for sending a file-like message body.

Added in version 3.7.

As an alternative to using the request() method described above, you can also send your request step by step, by using the four functions below.

HTTPConnection.putrequest(method, url, skip_host=False, skip_accept_encoding=False)

This should be the first call after the connection to the server has been made. It sends a line to the server consisting of the method string, the url string, and the HTTP version (HTTP/1.1). To disable automatic sending of Host: or Accept-Encoding: headers (for example to accept additional content encodings), specify skip_host or skip_accept_encoding with non-False values.

HTTPConnection.putheader(header, argument[, ...])

Send an RFC 822-style header to the server. It sends a line to the server consisting of the header, a colon and a space, and the first argument. If more arguments are given, continuation lines are sent, each consisting of a tab and an argument.

HTTPConnection.endheaders(message_body=None, *, encode_chunked=False)

Send a blank line to the server, signalling the end of the headers. The optional message_body argument can be used to pass a message body associated with the request.

If encode_chunked is True, the result of each iteration of message_body will be chunk-encoded as specified in RFC 7230, Section 3.3.1. How the data is encoded is dependent on the type of message_body. If message_body implements the buffer interface the encoding will result in a single chunk. If message_body is a collections.abc.Iterable, each iteration of message_body will result in a chunk. If message_body is a file object, each call to .read() will result in a chunk. The method automatically signals the end of the chunk-encoded data immediately after message_body.

Note

Due to the chunked encoding specification, empty chunks yielded by an iterator body will be ignored by the chunk-encoder. This is to avoid premature termination of the read of the request by the target server due to malformed encoding.

Changed in version 3.6: Added chunked encoding support and the encode_chunked parameter.

HTTPConnection.send(data)

Send data to the server. This should be used directly only after the endheaders() method has been called and before getresponse() is called.

Raises an auditing event http.client.send with arguments self, data.

### HTTPResponse Objects

An HTTPResponse instance wraps the HTTP response from the server. It provides access to the request headers and the entity body. The response is an iterable object and can be used in a with statement.

Changed in version 3.5: The io.BufferedIOBase interface is now implemented and all of its reader operations are supported.

HTTPResponse.read([amt])

Reads and returns the response body, or up to the next amt bytes.

HTTPResponse.readinto(b)

Reads up to the next len(b) bytes of the response body into the buffer b. Returns the number of bytes read.

Added in version 3.3.

HTTPResponse.getheader(name, default=None)

Return the value of the header name, or default if there is no header matching name. If there is more than one header with the name name, return all of the values joined by ‘, ‘. If default is any iterable other than a single string, its elements are similarly returned joined by commas.

HTTPResponse.getheaders()

Return a list of (header, value) tuples.

HTTPResponse.fileno()

Return the fileno of the underlying socket.

HTTPResponse.msg

A http.client.HTTPMessage instance containing the response headers. http.client.HTTPMessage is a subclass of email.message.Message.

HTTPResponse.version

HTTP protocol version used by server. 10 for HTTP/1.0, 11 for HTTP/1.1.

HTTPResponse.url

URL of the resource retrieved, commonly used to determine if a redirect was followed.

HTTPResponse.headers

Headers of the response in the form of an email.message.EmailMessage instance.

HTTPResponse.status

Status code returned by server.

HTTPResponse.reason

Reason phrase returned by server.

HTTPResponse.debuglevel

A debugging hook. If debuglevel is greater than zero, messages will be printed to stdout as the response is read and parsed.

HTTPResponse.closed

Is True if the stream is closed.

HTTPResponse.geturl()

Deprecated since version 3.9: Deprecated in favor of url.

HTTPResponse.info()

Deprecated since version 3.9: Deprecated in favor of headers.

HTTPResponse.getcode()

Deprecated since version 3.9: Deprecated in favor of status.

### Examples

Here is an example session that uses the GET method:

>>> import http.client >>> conn = http.client.HTTPSConnection("www.python.org") >>> conn.request("GET", "/") >>> r1 = conn.getresponse() >>> print(r1.status, r1.reason) 200 OK >>> data1 = r1.read() # This will return entire content. >>> # The following example demonstrates reading data in chunks. >>> conn.request("GET", "/") >>> r1 = conn.getresponse() >>> while chunk := r1.read(200): ... print(repr(chunk)) b'<!doctype html>\n<!--[if"... ... >>> # Example of an invalid request >>> conn = http.client.HTTPSConnection("docs.python.org") >>> conn.request("GET", "/parrot.spam") >>> r2 = conn.getresponse() >>> print(r2.status, r2.reason) 404 Not Found >>> data2 = r2.read() >>> conn.close()

Here is an example session that uses the HEAD method. Note that the HEAD method never returns any data.

>>> import http.client >>> conn = http.client.HTTPSConnection("www.python.org") >>> conn.request("HEAD", "/") >>> res = conn.getresponse() >>> print(res.status, res.reason) 200 OK >>> data = res.read() >>> print(len(data)) 0 >>> data == b'' True

Here is an example session that uses the POST method:

>>> import http.client, urllib.parse >>> params = urllib.parse.urlencode({'@number': 12524, '@type': 'issue', '@action': 'show'}) >>> headers = {"Content-type": "application/x-www-form-urlencoded", ... "Accept": "text/plain"} >>> conn = http.client.HTTPConnection("bugs.python.org") >>> conn.request("POST", "", params, headers) >>> response = conn.getresponse() >>> print(response.status, response.reason) 302 Found >>> data = response.read() >>> data b'Redirecting to <a href="https://bugs.python.org/issue12524">https://bugs.python.org/issue12524</a>' >>> conn.close()

Client side HTTP PUT requests are very similar to POST requests. The difference lies only on the server side where HTTP servers will allow resources to be created via PUT requests. It should be noted that custom HTTP methods are also handled in urllib.request.Request by setting the appropriate method attribute. Here is an example session that uses the PUT method:

>>> # This creates an HTTP request >>> # with the content of BODY as the enclosed representation >>> # for the resource http://localhost:8080/file ... >>> import http.client >>> BODY = "***filecontents***" >>> conn = http.client.HTTPConnection("localhost", 8080) >>> conn.request("PUT", "/file", BODY) >>> response = conn.getresponse() >>> print(response.status, response.reason) 200, OK

### HTTPMessage Objects

class http.client.HTTPMessage(email.message.Message)

An http.client.HTTPMessage instance holds the headers from an HTTP response. It is implemented using the email.message.Message class.
