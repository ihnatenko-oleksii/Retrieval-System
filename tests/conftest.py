from app.core.models import Chunk, ChunkMetadata


def make_chunk(
    content: str,
    file_path: str,
    chunk_index: int,
    file_name: str | None = None,
    document_id: str = "doc-1",
    extension: str = ".md",
    loader_type: str = "TextLoader",
) -> Chunk:
    """Build a real Chunk model for store/ingestion-level tests."""
    return Chunk(
        document_id=document_id,
        content=content,
        metadata=ChunkMetadata(
            document_id=document_id,
            file_path=file_path,
            file_name=file_name or file_path,
            extension=extension,
            loader_type=loader_type,
            chunk_index=chunk_index,
        ),
    )


def make_meta(file_path: str, chunk_index: int, file_name: str | None = None, **extra) -> dict:
    """Build a plain metadata dict as returned by the vector/BM25 stores."""
    meta = {
        "document_id": "doc-1",
        "file_path": file_path,
        "file_name": file_name or file_path,
        "extension": ".md",
        "loader_type": "TextLoader",
        "chunk_index": chunk_index,
    }
    meta.update(extra)
    return meta
