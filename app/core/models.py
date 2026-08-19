import uuid

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    file_path: str
    file_name: str
    extension: str
    loader_type: str
    section: str | None = None
    page_number: int | None = None

    model_config = {"extra": "allow"}


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: DocumentMetadata


class ChunkMetadata(BaseModel):
    document_id: str
    # Stable, human-readable identity used by evaluation labels. `Chunk.id`
    # remains an ingestion-time identifier, while this value survives a fresh
    # benchmark index rebuild.
    chunk_id: str | None = None
    file_path: str
    file_name: str
    extension: str
    loader_type: str
    chunk_index: int

    model_config = {"extra": "allow"}


class Chunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str
    content: str
    metadata: ChunkMetadata
