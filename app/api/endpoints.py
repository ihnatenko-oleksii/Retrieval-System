from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.vector_store.chroma_store import VectorStore
from app.vector_store.bm25_store import BM25Store
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.core.config import settings
from app.ingestion.pipeline import IngestionPipeline

app = FastAPI(title="Production RAG System API")

store = VectorStore()
bm25_store = BM25Store()
retriever = Retriever(store, bm25_store)
generator = Generator()

class AskRequest(BaseModel):
    query: str
    top_k: Optional[int] = settings.retrieval_top_k
    model: Optional[str] = settings.llm_model

class IngestRequest(BaseModel):
    directory: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(req: AskRequest):
    chunks = retriever.retrieve(req.query, top_k=req.top_k)
    if not chunks:
        return {"final_answer": "No relevant context found.", "retrieved_sources": []}
    
    gen = Generator(model_name=req.model)
    answer = gen.generate_answer(req.query, chunks)
    return answer

@app.post("/ingest")
def ingest_directory(req: IngestRequest):
    pipeline = IngestionPipeline()
    chunks = pipeline.process_directory(req.directory)
    if chunks:
        store.add_chunks(chunks)
        bm25_store.add_chunks(chunks)
        return {"status": "success", "chunks_indexed": len(chunks)}
    return {"status": "success", "chunks_indexed": 0, "message": "No documents found."}
