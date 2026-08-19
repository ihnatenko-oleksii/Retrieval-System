# Troubleshooting

## "No sentence-transformers model found..." / HF proxy errors

If your environment blocks model downloads:
- ensure internet/proxy access to Hugging Face, or
- pre-download/cache the embedding model, then rerun.

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

Indexes are persisted in `./storage`. If needed, clear and rebuild:

```bash
rm -rf ./storage/chroma ./storage/bm25_index.pkl
uv run main.py ingest "./data"
```
