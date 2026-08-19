import os
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.generation.generator import Generator
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.vector_store.bm25_store import BM25Store
from app.vector_store.chroma_store import VectorStore

app = FastAPI(title="Production RAG System API")


# Lazily-constructed, process-wide singletons. Building a VectorStore loads the
# embedding model, so this must not happen at import time (it would make the
# module impossible to import in tests or CLI tooling without a model already
# downloaded, and would slow down app startup unnecessarily). FastAPI's
# dependency overrides also let tests swap these out entirely.
@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


@lru_cache
def get_bm25_store() -> BM25Store:
    return BM25Store()


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_vector_store(), get_bm25_store())


@lru_cache
def get_generator() -> Generator:
    return Generator()


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=settings.retrieval_top_k, ge=1)
    model: str | None = settings.llm_model

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class IngestRequest(BaseModel):
    directory: str = Field(min_length=1)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(req: AskRequest, retriever: Annotated[Retriever, Depends(get_retriever)]):
    chunks = retriever.retrieve(req.query, top_k=req.top_k)
    if not chunks:
        return {"final_answer": "No relevant context found.", "retrieved_sources": []}

    gen = Generator(model_name=req.model) if req.model else get_generator()
    answer = gen.generate_answer(req.query, chunks)
    return answer


@app.post("/ingest")
def ingest_directory(
    req: IngestRequest,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    bm25_store: Annotated[BM25Store, Depends(get_bm25_store)],
):
    if not os.path.isdir(req.directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory}")

    pipeline = IngestionPipeline()
    chunks = pipeline.process_directory(req.directory)
    if chunks:
        vector_store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)
        return {"status": "success", "chunks_indexed": len(chunks)}
    return {"status": "success", "chunks_indexed": 0, "message": "No documents found."}
