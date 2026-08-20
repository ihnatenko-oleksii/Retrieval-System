# Python standard library: `asyncio`

Official source: https://docs.python.org/3/library/asyncio.html

### asyncio REPL

You can experiment with an asyncio concurrent context in the REPL:

$ python -m asyncio asyncio REPL ... Use "await" directly instead of "asyncio.run()". Type "help", "copyright", "credits" or "license" for more information. >>> import asyncio >>> await asyncio.sleep(10, result='hello') 'hello'

This REPL provides limited compatibility with PYTHON_BASIC_REPL. It is recommended that the default REPL is used for full functionality and the latest features.

Raises an auditing event cpython.run_stdin with no arguments.

Changed in version 3.12.5: (also 3.11.10, 3.10.15, 3.9.20, and 3.8.20) Emits audit events.

Changed in version 3.13: Uses PyREPL if possible, in which case PYTHONSTARTUP is also executed. Emits audit events.

### Reference

High-level APIs

Runners

Coroutines and tasks

Streams

Synchronization Primitives

Subprocesses

Queues

Exceptions

Introspection APIs

Call graph introspection

Command-line introspection tools

Low-level APIs

Event loop

Futures

Transports and Protocols

Policies

Platform Support

Extending

Guides and Tutorials

High-level API Index

Low-level API Index

Developing with asyncio

asyncio and free-threaded Python

Note

The source code for asyncio can be found in Lib/asyncio/.
