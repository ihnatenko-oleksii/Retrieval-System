import gradio as gr

from app.core.config import settings
from app.core.runtime_config import RetrievalConfig
from app.generation.generator import Generator
from app.retrieval.retriever import Retriever
from app.vector_store.bm25_store import BM25Store
from app.vector_store.chroma_store import VectorStore


def _list_ollama_models() -> list[str]:
    """Best-effort fetch of locally available Ollama models."""
    try:
        import ollama

        resp = ollama.list()
        models = resp.get("models", []) if isinstance(resp, dict) else []
        names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else None
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        seen, out = set(), []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out or [settings.llm_model]
    except Exception:
        return [settings.llm_model]


def _to_text(value) -> str:
    """Best-effort conversion of Gradio message content to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
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


def chat_interaction(
    message,
    history,
    n_results,
    model,
    dense_weight,
    reranker_on,
    rerank_top_n,
    query_rewriting_on,
    query_expansion_on,
):
    if not message.strip():
        return "", history

    try:
        dense_w = float(dense_weight)
        dense_w = max(0.0, min(1.0, dense_w))
        sparse_w = round(1.0 - dense_w, 4)

        cfg = RetrievalConfig(
            top_k=int(n_results),
            dense_weight=dense_w,
            sparse_weight=sparse_w,
            reranker_on=bool(reranker_on),
            rerank_top_n=int(rerank_top_n),
            query_rewriting_on=bool(query_rewriting_on),
            query_expansion_on=bool(query_expansion_on),
            llm_model=model,
        )

        store = VectorStore()
        bm25_store = BM25Store()
        retriever = Retriever(store, bm25_store)
        generator = Generator(model_name=model)

        chat_history = format_chat_history(history)

        chunks = retriever.retrieve(
            message,
            top_k=cfg.top_k,
            chat_history=chat_history,
            config=cfg,
        )
        answer = generator.generate_answer(message, chunks, chat_history=chat_history)

        final_answer = answer.get("final_answer", "")

        sources = answer.get("retrieved_sources", [])
        if sources:
            unique_sources = list(set([s["file_name"] for s in sources]))
            final_answer += f"\n\n*Sources: {', '.join(unique_sources)}*"

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
                        show_label=False, placeholder="Ask something about the indexed documents...", lines=2, scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear Chat")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                model = gr.Dropdown(
                    label="Ollama model",
                    choices=_list_ollama_models(),
                    value=settings.llm_model,
                    allow_custom_value=True,
                )
                n_results = gr.Slider(
                    label="Chunks to retrieve (top_k)",
                    minimum=1,
                    maximum=10,
                    value=settings.retrieval_top_k,
                    step=1,
                )

                gr.Markdown("#### Hybrid search")
                dense_weight = gr.Slider(
                    label="Dense weight (sparse = 1 - dense)",
                    minimum=0.0,
                    maximum=1.0,
                    value=settings.hybrid_search_weights_dense,
                    step=0.05,
                )

                gr.Markdown("#### Reranking")
                reranker_on = gr.Checkbox(
                    label="Reranker on",
                    value=settings.reranker_on,
                )
                rerank_top_n = gr.Slider(
                    label="Rerank top_n",
                    minimum=1,
                    maximum=10,
                    value=settings.rerank_top_n,
                    step=1,
                )

                gr.Markdown("#### Query pre-processing")
                query_rewriting_on = gr.Checkbox(
                    label="Query rewriting on",
                    value=settings.query_rewriting_on,
                )
                query_expansion_on = gr.Checkbox(
                    label="Query expansion on",
                    value=settings.query_expansion_on,
                )

                gr.Markdown(
                    f"<sub>Embedding model: `{settings.embedding_model}`. Changing it requires re-ingest.</sub>"
                )

        chat_inputs = [
            msg,
            chatbot,
            n_results,
            model,
            dense_weight,
            reranker_on,
            rerank_top_n,
            query_rewriting_on,
            query_expansion_on,
        ]

        msg.submit(fn=chat_interaction, inputs=chat_inputs, outputs=[msg, chatbot])
        submit_btn.click(fn=chat_interaction, inputs=chat_inputs, outputs=[msg, chatbot])
        clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, msg])

    return demo
