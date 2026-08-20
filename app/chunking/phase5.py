"""Chunking strategies used by the Phase 5 retrieval experiments.

The strategies return source intervals as well as text.  Keeping the interval
explicit is important: Phase 5 labels are source-span labels and must not be
silently rewritten when a chunking strategy changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.chunking.splitter import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class SourceChunk:
    text: str
    start: int
    end: int
    heading_path: str = ""


@dataclass(frozen=True)
class Heading:
    start: int
    end: int
    level: int
    title: str


def markdown_headings(text: str) -> tuple[Heading, ...]:
    return tuple(
        Heading(
            start=match.start(),
            end=match.end(),
            level=len(match.group(1)),
            title=match.group(2).strip(),
        )
        for match in re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$", text)
    )


def heading_path_at(text: str, position: int) -> str:
    stack: list[tuple[int, str]] = []
    for heading in markdown_headings(text):
        if heading.start > position:
            break
        stack = [item for item in stack if item[0] < heading.level]
        stack.append((heading.level, heading.title))
    return " > ".join(title for _, title in stack)


def _locate(text: str, pieces: list[str]) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    cursor = 0
    for piece in pieces:
        start = text.find(piece, cursor)
        if start < 0:
            start = text.find(piece)
        if start < 0:
            # The legacy recursive splitter can join separator fragments that
            # are not byte-contiguous in the source. Keep the candidate and
            # anchor it to its first recoverable line instead of silently
            # changing the chunk count or stable chunk indexes.
            first_line = next((line.strip() for line in piece.splitlines() if line.strip()), "")
            start = text.find(first_line, cursor) if first_line else -1
            if start < 0 and first_line:
                start = text.find(first_line)
            if start < 0:
                start = min(cursor, max(0, len(text) - 1))
        end = min(len(text), start + max(1, len(piece)))
        chunks.append(SourceChunk(piece, start, end, heading_path_at(text, start)))
        cursor = max(start + 1, end - 1)
    return chunks


def character_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[SourceChunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return _locate(text, splitter.split_text(text))


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"(?s)\S.*?(?=\n\s*\n|\Z)", text)]


def _pack_ranges(text: str, ranges: list[tuple[int, int]], target: int, overlap_items: int) -> list[SourceChunk]:
    if not ranges:
        return []
    chunks: list[SourceChunk] = []
    index = 0
    while index < len(ranges):
        start_index = index
        end_index = index
        length = 0
        while end_index < len(ranges):
            start, end = ranges[end_index]
            proposed = end - ranges[start_index][0]
            if end_index > start_index and proposed > target:
                break
            length = proposed
            end_index += 1
            if length >= target:
                break
        start = ranges[start_index][0]
        end = ranges[end_index - 1][1]
        chunks.append(SourceChunk(text[start:end], start, end, heading_path_at(text, start)))
        next_index = end_index - overlap_items
        index = max(index + 1, next_index)
    return chunks


def paragraph_chunks(text: str, *, target: int = 800, overlap_paragraphs: int = 1) -> list[SourceChunk]:
    return _pack_ranges(text, _paragraph_ranges(text), target, overlap_paragraphs)


def _section_ranges(text: str, max_level: int) -> list[tuple[int, int]]:
    headings = markdown_headings(text)
    boundaries = [heading.start for heading in headings if heading.level <= max_level]
    if not boundaries:
        return [(0, len(text))] if text.strip() else []
    ranges: list[tuple[int, int]] = []
    if boundaries[0] > 0 and text[: boundaries[0]].strip():
        ranges.append((0, boundaries[0]))
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else len(text)
        if text[start:end].strip():
            ranges.append((start, end))
    return ranges


def _split_large_sections(text: str, ranges: list[tuple[int, int]], target: int) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    for start, end in ranges:
        section = text[start:end]
        if len(section) <= target:
            chunks.append(SourceChunk(section, start, end, heading_path_at(text, start)))
            continue
        pieces = character_chunks(section, target, min(120, max(0, target // 5)))
        chunks.extend(
            SourceChunk(piece.text, start + piece.start, start + piece.end, heading_path_at(text, start + piece.start))
            for piece in pieces
        )
    return chunks


def heading_chunks(text: str, *, target: int = 800) -> list[SourceChunk]:
    """Keep every Markdown heading section separate, splitting only oversized sections."""
    return _split_large_sections(text, _section_ranges(text, max_level=6), target)


def section_chunks(text: str, *, target: int = 1000) -> list[SourceChunk]:
    """Keep top-level/major sections together, including their nested headings."""
    return _split_large_sections(text, _section_ranges(text, max_level=2), target)


def heading_context_chunks(text: str, *, target: int = 800, overlap_paragraphs: int = 1) -> list[SourceChunk]:
    """Add neighboring paragraph context while retaining the base source interval."""
    base = paragraph_chunks(text, target=target, overlap_paragraphs=overlap_paragraphs)
    ranges = _paragraph_ranges(text)
    enriched: list[SourceChunk] = []
    for chunk in base:
        paragraph_index = next(
            (index for index, (start, end) in enumerate(ranges) if start <= chunk.start < end),
            None,
        )
        neighbors: list[str] = []
        if paragraph_index is not None:
            for index in (paragraph_index - 1, paragraph_index + 1):
                if 0 <= index < len(ranges):
                    start, end = ranges[index]
                    neighbors.append(text[start:end])
        heading = heading_path_at(text, chunk.start)
        prefix = f"{heading}\n" if heading else ""
        context = "\n\n".join(neighbors)
        enriched.append(SourceChunk(f"{prefix}{context}\n\n{chunk.text}".strip(), chunk.start, chunk.end, heading))
    return enriched


def build_phase5_chunks(
    text: str,
    strategy: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[SourceChunk]:
    if strategy == "character":
        return character_chunks(text, chunk_size, chunk_overlap)
    if strategy == "paragraph":
        return paragraph_chunks(text, target=chunk_size, overlap_paragraphs=1)
    if strategy == "heading":
        return heading_chunks(text, target=chunk_size)
    if strategy == "section":
        return section_chunks(text, target=chunk_size)
    if strategy == "heading_context":
        return heading_context_chunks(text, target=chunk_size, overlap_paragraphs=1)
    raise ValueError(f"unsupported Phase 5 chunking strategy: {strategy}")
