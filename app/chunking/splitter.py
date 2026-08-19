import re
import uuid

from app.core.config import settings
from app.core.models import Chunk, ChunkMetadata, Document


class TextSplitter:
    def __init__(self, chunk_size: int = settings.chunk_size, chunk_overlap: int = settings.chunk_overlap):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError

    def split_document(self, document: Document) -> list[Chunk]:
        texts = self.split_text(document.content)
        chunks = []
        for i, text in enumerate(texts):
            stable_source = str(document.metadata.file_name or document.metadata.file_path).replace("\\", "/").strip()
            chunk_metadata = ChunkMetadata(
                document_id=document.id,
                chunk_id=f"{stable_source}::{i}",
                file_path=document.metadata.file_path,
                file_name=document.metadata.file_name,
                extension=document.metadata.extension,
                loader_type=document.metadata.loader_type,
                chunk_index=i,
                **document.metadata.model_dump(exclude={"file_path", "file_name", "extension", "loader_type"}),
            )
            chunks.append(Chunk(id=str(uuid.uuid4()), document_id=document.id, content=text, metadata=chunk_metadata))
        return chunks


class RecursiveCharacterTextSplitter(TextSplitter):
    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        separators: list[str] = None,
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


def get_splitter() -> TextSplitter:
    return RecursiveCharacterTextSplitter()
