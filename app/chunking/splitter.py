import re
import uuid

from app.core.config import settings
from app.core.models import Chunk, ChunkMetadata, Document


class TextSplitter:
    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.chunk_size = settings.chunk_size if chunk_size is None else chunk_size
        self.chunk_overlap = settings.chunk_overlap if chunk_overlap is None else chunk_overlap

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError

    def split_document(self, document: Document) -> list[Chunk]:
        texts = self.split_text(document.content)
        chunks = []
        source_cursor = 0
        headings = list(re.finditer(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$", document.content))
        for i, text in enumerate(texts):
            stable_source = str(document.metadata.file_name or document.metadata.file_path).replace("\\", "/").strip()
            source_start = document.content.find(text, source_cursor)
            if source_start < 0:
                source_start = document.content.find(text)
            source_end = source_start + len(text) if source_start >= 0 else None
            if source_start >= 0:
                source_cursor = source_start + 1

            heading_stack: list[tuple[int, str]] = []
            if source_start is not None and source_start >= 0:
                for heading in headings:
                    if heading.start() > source_start:
                        break
                    level = len(heading.group(1))
                    heading_stack = [item for item in heading_stack if item[0] < level]
                    heading_stack.append((level, heading.group(2).strip()))
            chunk_metadata = ChunkMetadata(
                document_id=document.id,
                chunk_id=f"{stable_source}::{i}",
                file_path=document.metadata.file_path,
                file_name=document.metadata.file_name,
                extension=document.metadata.extension,
                loader_type=document.metadata.loader_type,
                chunk_index=i,
                source_char_start=source_start if source_start >= 0 else None,
                source_char_end=source_end,
                heading_path=" > ".join(title for _, title in heading_stack),
                **document.metadata.model_dump(exclude={"file_path", "file_name", "extension", "loader_type"}),
            )
            chunks.append(Chunk(id=str(uuid.uuid4()), document_id=document.id, content=text, metadata=chunk_metadata))
        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if re.search(re.escape(_s), text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits = []
        for s in splits:
            if s:
                good_splits.append(s)

        current_doc = []
        length = 0
        for d in good_splits:
            d_len = len(d)
            if length + d_len > self.chunk_size and length > 0:
                chunk = separator.join(current_doc)
                final_chunks.append(chunk)

                # Handling overlap
                while current_doc:
                    if len(separator.join(current_doc)) > self.chunk_overlap:
                        current_doc.pop(0)
                    else:
                        break
                if not current_doc:
                    length = 0
                else:
                    length = len(separator.join(current_doc)) + (len(separator) if current_doc else 0)

            current_doc.append(d)
            length += d_len + (len(separator) if current_doc and length > 0 else 0)

        if current_doc:
            chunk = separator.join(current_doc)
            final_chunks.append(chunk)

        # Process recursively if needed
        recursive_chunks = []
        for chunk in final_chunks:
            if len(chunk) > self.chunk_size and new_separators:
                recursive_chunks.extend(self._split_text(chunk, new_separators))
            else:
                recursive_chunks.append(chunk)

        return recursive_chunks

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self.separators)


def get_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None) -> TextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
