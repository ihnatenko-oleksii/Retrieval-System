import typer
import logging
from app.ingestion.pipeline import IngestionPipeline
from app.vector_store.chroma_store import VectorStore
from app.vector_store.bm25_store import BM25Store
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = typer.Typer(help="Production RAG System CLI")

@app.command()
def ingest(directory: str):
    """
    Ingest documents from a given directory recursively.
    """
    typer.echo(f"Scanning and ingesting directory: {directory}")
    
    pipeline = IngestionPipeline()
    chunks = pipeline.process_directory(directory)
    
    typer.echo(f"Extracted {len(chunks)} chunks.")
    if chunks:
        typer.echo("Storing embeddings into ChromaDB and indexing BM25... This might take a moment.")
        
        # Dense vector store
        store = VectorStore()
        store.add_chunks(chunks)
        
        # Sparse vector store (BM25)
        bm25_store = BM25Store()
        bm25_store.add_chunks(chunks)
        
        typer.echo("Ingestion complete.")
    else:
        typer.echo("No chunks to store.")

@app.command()
def ask(query: str, top_k: int = settings.retrieval_top_k, model: str = settings.llm_model):
    """
    Ask a question over the indexed document knowledge base.
    """
    typer.echo(f"Query: '{query}'")
    
    store = VectorStore()
    bm25_store = BM25Store()
    retriever = Retriever(store, bm25_store)
    
    typer.echo("Retrieving and reranking chunks...")
    retrieved_chunks = retriever.retrieve(query, top_k=top_k)
    
    if not retrieved_chunks:
        typer.echo("No relevant information found.")
        return
        
    typer.echo(f"Found {len(retrieved_chunks)} relevant chunks. Generating answer...")
    
    generator = Generator(model_name=model)
    answer = generator.generate_answer(query, retrieved_chunks)
    
    typer.echo("\n" + "="*40)
    typer.echo("FINAL ANSWER")
    typer.echo("="*40)
    typer.echo(answer.get("final_answer", ""))
    
    typer.echo("\n" + "="*40)
    typer.echo("SOURCES")
    typer.echo("="*40)
    for src in answer.get("retrieved_sources", []):
        typer.echo(f"[{src['source_id']}] {src['file_name']} (Chunk {src['chunk_index']}, Score {src['distance']:.4f})")

@app.command()
def chat(top_k: int = settings.retrieval_top_k, model: str = settings.llm_model):
    """
    Start an interactive chat session with the RAG system.
    """
    typer.echo(f"Starting chat mode (model: {model}). Type 'exit' or 'quit' to stop.")
    
    store = VectorStore()
    bm25_store = BM25Store()
    retriever = Retriever(store, bm25_store)
    generator = Generator(model_name=model)
    
    chat_history = []
    
    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue
                
            retrieved_chunks = retriever.retrieve(query, top_k=top_k, chat_history=chat_history)
            answer_dict = generator.generate_answer(query, retrieved_chunks, chat_history=chat_history)
            answer_text = answer_dict.get("final_answer", "")
            
            typer.echo("\nAssistant: " + answer_text)
            
            sources = answer_dict.get("retrieved_sources", [])
            if sources:
                # remove duplicates for display
                unique_sources = list(set([s['file_name'] for s in sources]))
                typer.echo(f"\n[Sources: {', '.join(unique_sources)}]")
                
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": answer_text})
            
        except (KeyboardInterrupt, EOFError):
            break

@app.command()
def evals(jsonl_path: str, top_k: int = settings.retrieval_top_k):
    """
    Run an evaluation pipeline against a JSONL file of test cases.
    """
    from app.evals.evaluator import Evaluator
    typer.echo(f"Running evaluation against {jsonl_path} with top_k={top_k}...")
    evaluator = Evaluator(top_k=top_k)
    metrics = evaluator.evaluate_file(jsonl_path)
    
    if not metrics:
        typer.echo("Evaluation failed or no cases found.")
        return
        
    typer.echo("\nEvaluation Results:")
    typer.echo("-" * 40)
    for k, v in metrics.items():
        typer.echo(f"{k}: {v}")
    typer.echo("-" * 40)

@app.command()
def api(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.
    """
    import uvicorn
    typer.echo(f"Starting API server on {host}:{port}")
    uvicorn.run("app.api.endpoints:app", host=host, port=port, reload=True)

@app.command()
def ui():
    """
    Run the Gradio UI.
    """
    from app.ui.gradio_app import build_ui
    typer.echo("Starting Gradio UI...")
    build_ui().launch()

@app.command()
def evals_ui():
    """
    Run Gradio UI for evaluation metrics and charts.
    """
    from app.ui.evals_ui import build_evals_ui
    typer.echo("Starting Evals UI...")
    build_evals_ui().launch()


@app.command()
def embeddings_ui():
    """
    Run Gradio UI to visualize stored embeddings with t-SNE.
    """
    from app.ui.embeddings_ui import build_embeddings_ui

    typer.echo("Starting Embeddings UI...")
    build_embeddings_ui().launch()


@app.command()
def tuning_ui():
    """
    Run Gradio UI that sweeps runtime RAG parameters and finds the best configuration.
    """
    from app.ui.tuning_ui import build_tuning_ui

    typer.echo("Starting Tuning UI...")
    build_tuning_ui().launch()

if __name__ == "__main__":
    app()
