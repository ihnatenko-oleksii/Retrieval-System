# Troubleshooting

## "No sentence-transformers model found..." / HF proxy errors

If your environment blocks model downloads:
- ensure internet/proxy access to Hugging Face, or
- pre-download/cache the embedding model, then rerun.

## Embedding model or chunking changed

Chroma and BM25 indexes are built for the embedding model and chunk boundaries
used during ingestion. After changing `EMBEDDING_MODEL`, `CHUNK_SIZE`, or
`CHUNK_OVERLAP`, move only the production indexes aside and ingest again:

```bash
mv ./storage/chroma ./storage/chroma.previous
mv ./storage/bm25_index.pkl ./storage/bm25_index.pkl.previous
uv run main.py ingest "./data"
```

Keep the `.previous` paths until the new index has been checked; they are
recoverable local backups. Other benchmark indexes under `storage/` are left
untouched.

When a process is configured for Qwen3 but finds an E5 (or other incompatible)
index, it raises an error beginning with `Existing index was created with a
different embedding model`. This check happens before retrieval and does not
reset, delete, or convert the persisted vectors. Set `EMBEDDING_MODEL` to the
original model if you need to keep using that index.

## UI chat errors after multiple messages

This project normalizes chat history content before sending it to the
model. If you still hit a UI issue: stop the UI, rerun `uv run main.py ui`,
and share the latest traceback from the terminal.

## Acronym definitions not found

- Re-ingest after data/code changes: `uv run main.py ingest "./data"`
- Increase retrieval depth: `uv run main.py ask "What does CI stand for?" --top-k 8`
- Keep glossary files in supported text formats (`.md`, `.txt`) with clean
  `ACRONYM - definition` lines.

## Duplicate or stale retrieval results

Indexes are persisted in `./storage`. If needed, move the current indexes
aside and rebuild:

```bash
mv ./storage/chroma ./storage/chroma.previous
mv ./storage/bm25_index.pkl ./storage/bm25_index.pkl.previous
uv run main.py ingest "./data"
```
