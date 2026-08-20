# Python standard library: `tempfile`

Official source: https://docs.python.org/3/library/tempfile.html

### Examples

Here are some examples of typical usage of the tempfile module:

>>> import tempfile # create a temporary file and write some data to it >>> fp = tempfile.TemporaryFile() >>> fp.write(b'Hello world!') # read data from file >>> fp.seek(0) >>> fp.read() b'Hello world!' # close the file, it will be removed >>> fp.close() # create a temporary file using a context manager >>> with tempfile.TemporaryFile() as fp: ... fp.write(b'Hello world!') ... fp.seek(0) ... fp.read() b'Hello world!' >>> # file is now closed and removed # create a temporary file using a context manager # close the file, use the name to open the file again >>> with tempfile.NamedTemporaryFile(delete_on_close=False) as fp: ... fp.write(b'Hello world!') ... fp.close() ... # the file is closed, but not removed ... # open the file again by using its name ... with open(fp.name, mode='rb') as f: ... f.read() b'Hello world!' >>> # file is now removed # create a temporary directory using the context manager >>> with tempfile.TemporaryDirectory() as tmpdirname: ... print('created temporary directory', tmpdirname) >>> # directory and contents have been removed

### Deprecated functions and variables

A historical way to create temporary files was to first generate a file name with the mktemp() function and then create a file using this name. Unfortunately this is not secure, because a different process may create a file with this name in the time between the call to mktemp() and the subsequent attempt to create the file by the first process. The solution is to combine the two steps and create the file immediately. This approach is used by mkstemp() and the other functions described above.

tempfile.mktemp(suffix='', prefix='tmp', dir=None)

Deprecated since version 2.3: Use mkstemp() instead.

Return an absolute pathname of a file that did not exist at the time the call is made. The prefix, suffix, and dir arguments are similar to those of mkstemp(), except that bytes file names, suffix=None and prefix=None are not supported.

Warning

Use of this function may introduce a security hole in your program. By the time you get around to doing anything with the file name it returns, someone else may have beaten you to the punch. mktemp() usage can be replaced easily with NamedTemporaryFile(), passing it the delete=False parameter:

>>> f = NamedTemporaryFile(delete=False) >>> f.name '/tmp/tmptjujjt' >>> f.write(b"Hello World!\n") 13 >>> f.close() >>> os.unlink(f.name) >>> os.path.exists(f.name) False
