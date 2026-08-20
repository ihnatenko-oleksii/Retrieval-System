"""Create the sealed, corpus-separated Phase 5 benchmark-v4.

The corpus is a deterministic Markdown extraction of 50 Python 3 standard
library reference pages. Questions are generated from section headings and
documented terms with fixed templates; relevance remains a source-span label.
This script must run before final scoring and its output is never selection
eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.evals.span_relevance import (  # noqa: E402
    SourceSpan,
    load_markdown_spans,
    span_catalog_sha256,
)
from app.retrieval.phase5 import Phase5Config, build_records, corpus_sha256  # noqa: E402

OUTPUT_ROOT = REPO_ROOT / "docs" / "benchmark_v4"
CORPUS_ROOT = OUTPUT_ROOT / "corpus" / "python_stdlib"
PYTHON_DOCS_ROOT = "https://docs.python.org/3/library"
LICENSE_URL = "https://docs.python.org/3/license.html"

MODULES = (
    "asyncio",
    "argparse",
    "array",
    "contextlib",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "hashlib",
    "heapq",
    "http",
    "http.client",
    "http.server",
    "importlib",
    "inspect",
    "io",
    "itertools",
    "json",
    "logging",
    "multiprocessing",
    "pathlib",
    "pickle",
    "platform",
    "queue",
    "re",
    "secrets",
    "selectors",
    "shutil",
    "signal",
    "socket",
    "sqlite3",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "traceback",
    "types",
    "typing",
    "unicodedata",
    "urllib.parse",
    "urllib.request",
    "uuid",
    "venv",
    "warnings",
    "weakref",
    "zipfile",
)

STOPWORDS = {
    "about",
    "allows",
    "class",
    "from",
    "function",
    "module",
    "object",
    "python",
    "return",
    "returns",
    "section",
    "the",
    "this",
    "used",
    "using",
    "when",
}


class PythonDocsParser(HTMLParser):
    """Extract the main reference content without depending on an HTML library."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.main_depth = 0
        self.div_depth = 0
        self.main_role_div_depth: int | None = None
        self.skip_depth = 0
        self.heading: tuple[str, int, list[str]] | None = None
        self.block: tuple[str, list[str]] | None = None
        self.events: list[tuple[str, int | str, str]] = []

    def _finish_block(self) -> None:
        if self.block is not None:
            kind, values = self.block
            text = _clean_text("".join(values))
            if text:
                self.events.append((kind, 0, text))
            self.block = None

    def _finish_heading(self) -> None:
        if self.heading is not None:
            tag, level, values = self.heading
            text = _clean_text("".join(values))
            if text:
                self.events.append(("heading", level, text))
            self.heading = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "div":
            self.div_depth += 1
        if tag == "main":
            self.main_depth += 1
            return
        if tag == "div" and attributes.get("role") == "main":
            self.main_depth += 1
            self.main_role_div_depth = self.div_depth
            return
        if self.main_depth == 0:
            return
        if tag in {"script", "style", "nav", "aside", "footer"}:
            self._finish_heading()
            self._finish_block()
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4"}:
            self._finish_heading()
            self._finish_block()
            self.heading = (tag, int(tag[1]), [])
        elif tag == "p" and "rubric" in str(attributes.get("class", "")):
            self._finish_heading()
            self._finish_block()
            self.heading = ("rubric", 2, [])
        elif tag in {"p", "pre", "li", "dt", "dd"}:
            self._finish_heading()
            self._finish_block()
            self.block = ("block", [])
        elif tag == "br" and self.block is not None:
            self.block[1].append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self._finish_heading()
            self._finish_block()
            self.main_depth = max(0, self.main_depth - 1)
            return
        if tag == "div":
            if self.main_role_div_depth == self.div_depth:
                self._finish_heading()
                self._finish_block()
                self.main_depth = max(0, self.main_depth - 1)
                self.main_role_div_depth = None
            self.div_depth = max(0, self.div_depth - 1)
            return
        if self.skip_depth:
            if tag in {"script", "style", "nav", "aside", "footer"}:
                self.skip_depth -= 1
            return
        if self.heading is not None and (tag == self.heading[0] or self.heading[0] == "rubric" and tag == "p"):
            self._finish_heading()
        elif self.block is not None and tag in {"p", "pre", "li", "dt", "dd"}:
            self._finish_block()

    def handle_data(self, data: str) -> None:
        if self.main_depth == 0 or self.skip_depth:
            return
        if self.heading is not None:
            self.heading[2].append(data)
        elif self.block is not None:
            self.block[1].append(data)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("¶", "")).strip()


def _fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": "retrieval-system-benchmark-v4/1.0"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _markdown_for_module(module: str, html: str, url: str) -> str:
    parser = PythonDocsParser()
    parser.feed(html)
    headings = [(int(level), text) for kind, level, text in parser.events if kind == "heading"]
    if not headings:
        raise ValueError(f"no headings extracted from {url}")
    title = headings[0][1]
    lines = [f"# Python standard library: `{module}`", "", f"Official source: {url}", ""]
    current_heading: tuple[int, str] | None = None
    body: list[str] = []
    section_count = 0

    def flush() -> None:
        nonlocal body, section_count
        if current_heading is None:
            body = []
            return
        text = "\n\n".join(body).strip()
        if len(text) < 40:
            body = []
            return
        level, heading = current_heading
        lines.extend([f"{'#' * min(6, max(2, level + 1))} {heading}", "", text, ""])
        section_count += 1
        body = []

    for kind, level, text in parser.events:
        if kind == "heading":
            if text == title and current_heading is None:
                continue
            flush()
            current_heading = (int(level), text)
        elif current_heading is not None:
            body.append(text)
    flush()
    if section_count < 2:
        blocks = [text for kind, _, text in parser.events if kind == "block" and len(text) >= 20]
        if len(blocks) < 2:
            raise ValueError(f"too few substantive sections extracted from {url}: {section_count}")
        lines = [f"# Python standard library: `{module}`", "", f"Official source: {url}", ""]
        group_size = max(2, min(8, len(blocks) // 3 or 2))
        for group_index in range(0, len(blocks), group_size):
            group = blocks[group_index : group_index + group_size]
            lines.extend([f"## Reference overview {group_index // group_size + 1}", "", "\n\n".join(group), ""])
    return "\n".join(lines).rstrip() + "\n"


def _write_corpus(*, refresh: bool) -> dict[str, str]:
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)
    source_urls: dict[str, str] = {}
    for module in MODULES:
        slug = module.replace(".", "_")
        document_id = f"python_stdlib/{slug}.md"
        path = CORPUS_ROOT.parent / document_id
        url = f"{PYTHON_DOCS_ROOT}/{module}.html"
        source_urls[document_id] = url
        if path.exists() and not refresh:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_markdown_for_module(module, _fetch(url), url), encoding="utf-8")
    return source_urls


def _document_id(span: SourceSpan) -> str:
    return span.document_id


def _topic(section_text: str, heading: str) -> str:
    code_terms = re.findall(r"`([^`]+)`", section_text)
    if code_terms:
        return code_terms[0]
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9_.-]{3,}", heading + " " + section_text)
        if word.casefold() not in STOPWORDS
    ]
    return " ".join(words[:4]) or heading


def _label(span: SourceSpan, gain: int) -> dict[str, object]:
    return {**span.as_dict(), "gain": gain}


def _sections() -> list[tuple[SourceSpan, str, str]]:
    spans = load_markdown_spans(OUTPUT_ROOT / "corpus")
    by_document: dict[str, list[SourceSpan]] = {}
    for span in spans:
        by_document.setdefault(span.document_id, []).append(span)
    sections: list[tuple[SourceSpan, str, str]] = []
    for document_id, document_spans in sorted(by_document.items()):
        text = (OUTPUT_ROOT / "corpus" / document_id).read_text(encoding="utf-8")
        for span in document_spans:
            section_text = text[span.start : span.end]
            sections.append((span, document_id, section_text))
    return sections


def _pick(sections: list[tuple[SourceSpan, str, str]], count: int, offset: int) -> list[tuple[SourceSpan, str, str]]:
    if len(sections) < count:
        raise ValueError(f"benchmark-v4 needs {count} sections but found {len(sections)}")
    return [sections[(offset + index * 7) % len(sections)] for index in range(count)]


def _build_cases() -> list[dict[str, object]]:
    sections = _sections()
    cases: list[dict[str, object]] = []
    templates = {
        "lexical": lambda module, heading, topic: (
            f"Where does the Python {module} reference describe {heading}, including the documented term {topic}?"
        ),
        "semantic": lambda module, heading, topic: (
            f"How does the Python {module} reference explain the behavior associated with {topic}?"
        ),
        "ambiguous": lambda module, heading, topic: (
            f"For Python {module}, what surrounding context should a developer check when dealing with {topic}?"
        ),
        "fine_grained": lambda module, heading, topic: (
            f"What specific constraint or implementation detail is stated under {heading} in Python {module}?"
        ),
    }
    for category_index, category in enumerate(("lexical", "semantic", "ambiguous", "fine_grained")):
        for question_index, (span, document_id, section_text) in enumerate(
            _pick(sections, 25, category_index * 3), start=1
        ):
            module = Path(document_id).stem.replace("_", ".")
            topic = _topic(section_text, span.heading)
            cases.append(
                {
                    "id": f"v4-{category}-{question_index:03d}",
                    "category": category,
                    "question": templates[category](module, span.heading, topic),
                    "relevance_spans": [_label(span, 3)],
                    "source_document": document_id,
                }
            )

    by_document: dict[str, list[tuple[SourceSpan, str, str]]] = {}
    for section in sections:
        by_document.setdefault(section[1], []).append(section)
    pairs: list[tuple[tuple[SourceSpan, str, str], tuple[SourceSpan, str, str]]] = []
    for document_id in sorted(by_document):
        document_sections = by_document[document_id]
        pairs.extend(zip(document_sections[::2], document_sections[1::2], strict=False))
    if len(pairs) < 25:
        raise ValueError(f"benchmark-v4 needs 25 multi-relevant section pairs but found {len(pairs)}")
    for question_index, (left, right) in enumerate(pairs[:25], start=1):
        left_span, document_id, left_text = left
        right_span, _, right_text = right
        module = Path(document_id).stem.replace("_", ".")
        cases.append(
            {
                "id": f"v4-multiple_relevant-{question_index:03d}",
                "category": "multiple_relevant",
                "question": (
                    f"How do the {left_span.heading} and {right_span.heading} parts of the Python {module} reference "
                    "fit together?"
                ),
                "relevance_spans": [_label(left_span, 3), _label(right_span, 2)],
                "source_document": document_id,
                "source_topics": [_topic(left_text, left_span.heading), _topic(right_text, right_span.heading)],
            }
        )
    cases.sort(key=lambda case: str(case["id"]))
    return cases


def _question_sha256(cases: list[dict[str, object]]) -> str:
    payload = "\n".join(
        json.dumps(
            {key: case[key] for key in ("id", "category", "question", "relevance_spans")},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for case in cases
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate(cases: list[dict[str, object]]) -> None:
    if len(cases) < 100:
        raise ValueError(f"benchmark-v4 must contain at least 100 cases, found {len(cases)}")
    if len({str(case["id"]) for case in cases}) != len(cases):
        raise ValueError("benchmark-v4 case IDs are not unique")
    categories = {category: sum(case["category"] == category for case in cases) for category in sorted({case["category"] for case in cases})}
    if categories != {"ambiguous": 25, "fine_grained": 25, "lexical": 25, "multiple_relevant": 25, "semantic": 25}:
        raise ValueError(f"benchmark-v4 category balance changed: {categories}")
    spans = load_markdown_spans(OUTPUT_ROOT / "corpus")
    span_ids = {span.span_id for span in spans}
    for case in cases:
        labels = case.get("relevance_spans", [])
        if not labels:
            raise ValueError(f"case has no span labels: {case['id']}")
        for label in labels:
            if label["span_id"] not in span_ids:
                raise ValueError(f"unknown span label {label['span_id']} in {case['id']}")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Redownload and rebuild corpus documents.")
    args = parser.parse_args()

    source_urls = _write_corpus(refresh=args.refresh)
    cases = _build_cases()
    _validate(cases)
    spans = load_markdown_spans(OUTPUT_ROOT / "corpus")
    canonical_chunks = build_records(OUTPUT_ROOT / "corpus", Phase5Config(name="v4-canonical"))
    corpus_hash = corpus_sha256(OUTPUT_ROOT / "corpus")
    atlas_hash = corpus_sha256(REPO_ROOT / "docs" / "benchmark_corpus")
    if corpus_hash == atlas_hash:
        raise ValueError("benchmark-v4 corpus fingerprint matches Atlas; corpus separation failed")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_jsonl(OUTPUT_ROOT / "eval.jsonl", cases)
    (OUTPUT_ROOT / "spans.json").write_text(
        json.dumps([span.as_dict() for span in spans], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    manifest = {
        "artifact": "benchmark-v4",
        "selection_eligible": False,
        "sealed_before_final_evaluation": True,
        "tuning_allowed": False,
        "question_count": len(cases),
        "category_counts": {category: sum(case["category"] == category for case in cases) for category in sorted({case["category"] for case in cases})},
        "document_count": len(source_urls),
        "chunk_count_at_1000_200": len(canonical_chunks),
        "span_count": len(spans),
        "corpus_sha256": corpus_hash,
        "atlas_corpus_sha256": atlas_hash,
        "span_catalog_sha256": span_catalog_sha256(spans),
        "question_sha256": _question_sha256(cases),
        "source": {
            "collection": "Python 3 standard library reference documentation",
            "base_url": PYTHON_DOCS_ROOT,
            "license": "Python Software Foundation License",
            "license_url": LICENSE_URL,
            "documents": source_urls,
        },
        "question_generation": {
            "method": "deterministic templates over section headings and first documented code/technical terms",
            "categories": ["lexical", "semantic", "ambiguous", "fine_grained", "multiple_relevant"],
            "relevance": "heading-delimited source intervals from the generated Markdown, with gains 3 and 2 for paired evidence",
            "answer_generation_used": False,
            "benchmark_tuning_used": False,
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
