import gradio as gr
from app.vector_store.chroma_store import VectorStore
from app.vector_store.bm25_store import BM25Store
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.core.config import settings


def _to_text(value) -> str:
    """Best-effort conversion of Gradio message content to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # common shape: {"text": "...", "type": "text"}
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        return str(value)
    if isinstance(value, list):
        parts = [_to_text(item) for item in value]
        return " ".join([p for p in parts if p]).strip()
    return str(value)

def format_chat_history(history):
    """
    Normalize Gradio history into OpenAI-style messages.
    """
    formatted = []
    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = _to_text(item.get("content"))
            if role in {"user", "assistant"} and content:
                formatted.append({"role": role, "content": content})
            continue

        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, ast_msg = item
            user_text = _to_text(user_msg)
            assistant_text = _to_text(ast_msg)
            if user_text:
                formatted.append({"role": "user", "content": user_text})
            if assistant_text:
                formatted.append({"role": "assistant", "content": assistant_text})
    return formatted

def chat_interaction(message, history, n_results, model):
    if not message.strip():
        return "", history
        
    try:
        store = VectorStore()
        bm25_store = BM25Store()
        retriever = Retriever(store, bm25_store)
        generator = Generator(model_name=model)
        
        chat_history = format_chat_history(history)
        
        chunks = retriever.retrieve(message, top_k=n_results, chat_history=chat_history)
        answer = generator.generate_answer(message, chunks, chat_history=chat_history)
        
        final_answer = answer.get("final_answer", "")
        
        sources = answer.get("retrieved_sources", [])
        if sources:
            unique_sources = list(set([s['file_name'] for s in sources]))
            final_answer += f"\n\n*Sources: {', '.join(unique_sources)}*"
            
        # Return messages-format history to satisfy newer Chatbot expectations.
        messages = format_chat_history(history)
        messages.append({"role": "user", "content": message})
        messages.append({"role": "assistant", "content": final_answer})
        return "", messages
        
    except Exception as e:
        messages = format_chat_history(history)
        messages.append({"role": "user", "content": message})
        messages.append({"role": "assistant", "content": f"Error: {e}"})
        return "", messages

def clear_chat():
    return [], ""

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Production RAG System") as demo:
        gr.Markdown("## Conversational RAG (ChromaDB + BM25 + Ollama)")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="RAG Chatbot")
                with gr.Row():
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Ask something about the indexed documents...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat")
                
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                model = gr.Dropdown(
                    label="Ollama model",
                    choices=["llama3.2:latest", "llama3.2", "qwen3:8b", "deepseek-coder-v2:latest"],
                    value=settings.llm_model,
                    allow_custom_value=True,
                )
                n_results = gr.Slider(
                    label="Chunks to retrieve",
                    minimum=1,
                    maximum=10,
                    value=settings.retrieval_top_k,
                    step=1,
                )
                
        # Handle submit events
        msg.submit(
            fn=chat_interaction,
            inputs=[msg, chatbot, n_results, model],
            outputs=[msg, chatbot],
        )
        submit_btn.click(
            fn=chat_interaction,
            inputs=[msg, chatbot, n_results, model],
            outputs=[msg, chatbot],
        )
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, msg],
        )
        
    return demo
