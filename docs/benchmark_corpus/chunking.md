# Chunking

Documents are split into smaller chunks before they are embedded and
indexed, since retrieval quality depends on chunks being small enough to be
topically focused but large enough to preserve context.

## Recursive character splitting

The default splitter tries a sequence of separators, from largest to
smallest: double newline, single newline, space, and finally individual
characters. It picks the first separator actually present in the text,
splits on it, and greedily packs pieces into chunks up to `chunk_size`
characters. If a resulting chunk is still too large, the same process runs
again recursively with the next, finer-grained separator.

## Overlap

Consecutive chunks share a configurable `chunk_overlap` of trailing content
from the previous chunk, carried into the next one. This keeps sentences or
ideas that straddle a chunk boundary from losing context in either chunk.

## Defaults

The default configuration is a `chunk_size` of 1000 characters and a
`chunk_overlap` of 200 characters, both configurable through environment
variables. Every chunk keeps metadata back to its source document,
including the file path, file name, extension, loader type, and a
sequential chunk index within that document.
