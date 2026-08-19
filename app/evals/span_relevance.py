"""Source-span utilities for labels that survive chunking changes."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")


@dataclass(frozen=True)
class SourceSpan:
    """A stable source section identified by document path and byte offsets."""

    document_id: str
    span_id: str
    heading: str
    level: int
    start: int
    end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "span_id": self.span_id,
            "heading": self.heading,
            "level": self.level,
            "start": self.start,
            "end": self.end,
        }


def slugify_heading(heading: str) -> str:
    """Create a readable, deterministic ID component from a Markdown heading."""
    value = re.sub(r"[`*_]", "", heading.casefold())
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "section"


def extract_markdown_spans(document_id: str, text: str) -> list[SourceSpan]:
    """Return heading-delimited source spans, including their heading text."""
    matches = list(_HEADING_RE.finditer(text))
    spans: list[SourceSpan] = []
    seen_slugs: dict[str, int] = {}
    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        base_slug = slugify_heading(heading)
        seen_slugs[base_slug] = seen_slugs.get(base_slug, 0) + 1
        suffix = f"-{seen_slugs[base_slug]}" if seen_slugs[base_slug] > 1 else ""
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        spans.append(
            SourceSpan(
                document_id=document_id,
                span_id=f"{document_id}#{base_slug}{suffix}",
                heading=heading,
                level=len(match.group(1)),
                start=match.start(),
                end=end,
            )
        )
    return spans


def load_markdown_spans(corpus_root: Path) -> list[SourceSpan]:
    """Build a deterministic source-span catalog for a Markdown corpus."""
    corpus_root = corpus_root.resolve()
    spans: list[SourceSpan] = []
    for path in sorted(corpus_root.rglob("*.md")):
        document_id = path.relative_to(corpus_root).as_posix()
        spans.extend(extract_markdown_spans(document_id, path.read_text(encoding="utf-8")))
    return spans


def span_catalog_sha256(spans: list[SourceSpan]) -> str:
    """Hash the canonical span catalog for benchmark integrity checks."""
    payload = "\n".join(
        f"{span.document_id}\t{span.span_id}\t{span.start}\t{span.end}\t{span.heading}"
        for span in spans
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalise_source(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().casefold()


def span_matches_metadata(label: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Return whether a chunk's source interval overlaps one labeled span."""
    document_id = _normalise_source(label.get("document_id"))
    file_name = _normalise_source(metadata.get("file_name"))
    file_path = _normalise_source(metadata.get("file_path"))
    source_matches = any(
        candidate == document_id or candidate.endswith(f"/{document_id}")
        for candidate in (file_name, file_path)
        if candidate and document_id
    )
    if not source_matches:
        return False

    try:
        label_start = int(label["start"])
        label_end = int(label["end"])
        chunk_start = int(metadata["source_char_start"])
        chunk_end = int(metadata["source_char_end"])
    except (KeyError, TypeError, ValueError):
        return False
    return max(label_start, chunk_start) < min(label_end, chunk_end)
