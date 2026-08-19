# Recommended Workflow (Best Quality)

1) **Preprocess your source data for RAG.** Retrieval quality depends
   heavily on source quality — usually a bigger lever than any retrieval
   parameter. Aim for: one topic per chunk-sized section, no duplicate or
   stale content, acronyms expanded on first use, clear stable headings, and
   important facts kept close to the heading that names them. Prefer a
   local/private LLM for this step if your documents are sensitive.

   Suggested prompt for an LLM data-prep pass:

   ```text
   Rewrite this document to be RAG-ready in Markdown.
   Rules:
   - Keep all factual information (do not invent or remove facts).
   - Use clear section headings and short, self-contained paragraphs.
   - Expand acronyms on first use: "ACRONYM (full name)".
   - Remove boilerplate, duplicated lines, and irrelevant fluff.
   - Keep critical entities exact: IDs, URLs, versions, statuses, enum values.
   - If a section mixes topics, split it into separate sections.
   Return only the improved Markdown.
   ```

2) **Ingest your prepared data.**

   ```bash
   uv run main.py ingest "./data"
   ```

3) **Create an evaluation set** (`evals-json.jsonl`). Use an LLM to draft
   cases, then review them by hand — don't trust generated evals blindly.
   Include edge cases and negative cases, not just easy glossary lookups.

   ```text
   You are creating an evaluation dataset for a RAG system.
   Given the documents, generate JSONL lines with schema:
   {"question":"...","expected_keywords":["..."],"expected_source":"..."}

   Requirements:
   - Create diverse questions: definitions, process steps, URLs, statuses, technical details.
   - Include both easy and hard questions.
   - expected_source must be the exact relative source path/file used in the corpus.
   - expected_keywords should contain key terms that must appear in a correct answer.
   - Keep questions unambiguous and answerable from one or few sources.
   - Output only valid JSONL (one JSON object per line, no markdown).
   Produce at least 30-100 cases depending on corpus size.
   ```

4) **Run hyperparameter tuning** to find the best `top_k`, hybrid
   dense/sparse weight, reranker, query rewriting, and query expansion
   settings for your corpus:

   ```bash
   uv run main.py tuning-ui
   ```

5) **Apply the winning config** in the production chat UI — set the same
   values in the sidebar and use them for real Q&A:

   ```bash
   uv run main.py ui
   ```

See [docs/benchmark.md](benchmark.md) for a fully worked, reproducible
example of steps 2–4 on a small sample corpus.

## Setting up an AI coding agent to run this project end-to-end

If you're using a coding agent (Claude Code, Cursor, etc.) to stand this
project up in a new environment, this prompt works well once your source
files are in `./data`:

```text
Set up and run this Retrieval System project end-to-end.

Follow these steps exactly:
1) Install dependencies:
   - Run: uv sync
2) Ensure Ollama model is available:
   - Run: ollama pull llama3.2
3) Ingest documents from ./data:
   - Run: uv run main.py ingest "./data"
4) Run a quick sanity question:
   - Run: uv run main.py ask "What is OOP?"
5) Launch the main UI:
   - Run: uv run main.py ui

After each step, report success/failure and the key output.
If a step fails, diagnose and fix it before continuing.
```
