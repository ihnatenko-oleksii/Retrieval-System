from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.core.config import settings
from app.core.runtime_config import RetrievalConfig
from app.evals.evaluator import Evaluator

CSS = """
:root {
  --accent: #7c3aed;
  --ok: #22c55e;
  --bad: #ef4444;
}

.gradio-container {
  max-width: 100% !important;
  padding: 0 24px !important;
}
"""


def _list_ollama_models() -> list[str]:
    """
    Best-effort fetch of locally available Ollama models.
    Falls back to a small default set if Ollama isn't running.
    """
    try:
        import ollama  # local dependency

        resp = ollama.list()
        models = resp.get("models", []) if isinstance(resp, dict) else []
        names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else None
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        # De-dup while preserving order
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out
    except Exception:
        return ["llama3.2:latest", "qwen3:8b", "deepseek-coder-v2:latest"]


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _build_metrics_bar(metrics_df: pd.DataFrame) -> go.Figure:
    if metrics_df.empty:
        return _empty_figure("Metrics Overview")
    fig = go.Figure(
        data=[
            go.Bar(
                x=metrics_df["metric"],
                y=metrics_df["value"],
                marker_color=["#60a5fa", "#818cf8", "#a78bfa", "#f472b6", "#22d3ee"],
                text=[f"{v:.4f}" for v in metrics_df["value"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Metrics Overview",
        yaxis=dict(range=[0, 1.05], title="score"),
        xaxis_title="metric",
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _build_hit_miss_donut(source_hits: int, source_misses: int) -> go.Figure:
    total = source_hits + source_misses
    if total == 0:
        return _empty_figure("Source Hit vs Miss")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["source_hit", "source_miss"],
                values=[source_hits, source_misses],
                hole=0.55,
                marker=dict(colors=["#22c55e", "#ef4444"]),
                textinfo="label+percent+value",
            )
        ]
    )
    fig.update_layout(
        title="Source Hit vs Miss",
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _build_rank_distribution(details_df: pd.DataFrame, top_k: int) -> go.Figure:
    if details_df.empty:
        return _empty_figure("First Relevant Rank Distribution")

    rank_counts = {f"rank_{i}": 0 for i in range(1, top_k + 1)}
    rank_counts["miss"] = 0

    for rank in details_df["first_relevant_rank"]:
        if pd.isna(rank):
            rank_counts["miss"] += 1
            continue
        r = int(rank)
        if 1 <= r <= top_k:
            rank_counts[f"rank_{r}"] += 1
        else:
            rank_counts["miss"] += 1

    labels = list(rank_counts.keys())
    values = list(rank_counts.values())
    colors = ["#34d399" if label != "miss" else "#f87171" for label in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="First Relevant Rank Distribution",
        template="plotly_dark",
        xaxis_title="bucket",
        yaxis_title="cases",
        yaxis=dict(rangemode="tozero"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _build_keyword_indicator(keyword_hit_rate: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=keyword_hit_rate * 100,
            number={"suffix": "%"},
            title={"text": "Keyword Hit Rate"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22d3ee"},
                "steps": [
                    {"range": [0, 40], "color": "#7f1d1d"},
                    {"range": [40, 70], "color": "#78350f"},
                    {"range": [70, 100], "color": "#14532d"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def run_evaluation(
    jsonl_path: str,
    top_k: int,
    model_name: str,
    dense_weight: float,
    reranker_on: bool,
    rerank_top_n: int,
    query_rewriting_on: bool,
    query_expansion_on: bool,
):
    eval_path = Path(jsonl_path).expanduser()
    if not eval_path.exists() and eval_path.suffix == ".json":
        alt_path = eval_path.with_suffix(".jsonl")
        if alt_path.exists():
            eval_path = alt_path
    if not eval_path.exists():
        return (
            f"❌ File not found: `{eval_path}`",
            pd.DataFrame(columns=["metric", "value"]),
            _empty_figure("Metrics Overview"),
            pd.DataFrame(
                columns=[
                    "question",
                    "expected_source",
                    "source_hit",
                    "first_relevant_rank",
                    "retrieved_sources",
                    "keyword_hit_score",
                ]
            ),
            _empty_figure("Source Hit vs Miss"),
            _empty_figure("First Relevant Rank Distribution"),
            _empty_figure("Keyword Hit Rate"),
        )

    dense_w = max(0.0, min(1.0, float(dense_weight)))
    sparse_w = round(1.0 - dense_w, 4)
    cfg = RetrievalConfig(
        top_k=int(top_k),
        dense_weight=dense_w,
        sparse_weight=sparse_w,
        reranker_on=bool(reranker_on),
        rerank_top_n=int(rerank_top_n),
        query_rewriting_on=bool(query_rewriting_on),
        query_expansion_on=bool(query_expansion_on),
        llm_model=model_name,
    )
    evaluator = Evaluator(config=cfg, model_name=model_name)
    metrics, case_details = evaluator.evaluate_cases(str(eval_path))

    if not metrics:
        return (
            "❌ No metrics produced. Verify your JSONL file format and content.",
            pd.DataFrame(columns=["metric", "value"]),
            _empty_figure("Metrics Overview"),
            pd.DataFrame(
                columns=[
                    "question",
                    "expected_source",
                    "source_hit",
                    "first_relevant_rank",
                    "retrieved_sources",
                    "keyword_hit_score",
                ]
            ),
            _empty_figure("Source Hit vs Miss"),
            _empty_figure("First Relevant Rank Distribution"),
            _empty_figure("Keyword Hit Rate"),
        )

    metrics_df = pd.DataFrame([{"metric": metric, "value": value} for metric, value in metrics.items()])
    details_df = pd.DataFrame(case_details)

    source_hits = int(details_df["source_hit"].sum()) if not details_df.empty else 0
    total_cases = len(details_df)
    source_misses = max(total_cases - source_hits, 0)
    metrics_plot = _build_metrics_bar(metrics_df)
    hit_plot = _build_hit_miss_donut(source_hits, source_misses)
    rank_plot = _build_rank_distribution(details_df, int(top_k))
    keyword_indicator = _build_keyword_indicator(float(metrics.get("keyword_hit_rate", 0.0)))

    status = (
        f"✅ Evaluated **{total_cases}** cases from `{eval_path}` with `top_k={int(top_k)}`.\n"
        f"Source hit cases: **{source_hits}/{total_cases}**"
    )
    return status, metrics_df, metrics_plot, details_df, hit_plot, rank_plot, keyword_indicator


def build_evals_ui() -> gr.Blocks:
    with gr.Blocks(css=CSS, theme=gr.themes.Monochrome(), title="RAG Evals Dashboard") as demo:
        gr.Markdown("## RAG Evals Dashboard")
        gr.Markdown("Run JSONL evaluations and inspect metrics with course-style visual dashboards.")

        with gr.Row():
            jsonl_path = gr.Textbox(
                label="Eval JSONL path",
                value="./evals-json.jsonl",
                placeholder="e.g. ./data/evals.jsonl",
                scale=4,
            )
            model_name = gr.Dropdown(
                label="Ollama model",
                choices=_list_ollama_models(),
                value=settings.llm_model,
                allow_custom_value=True,
                scale=2,
            )
            top_k = gr.Slider(
                label="Top-K",
                minimum=1,
                maximum=10,
                value=settings.retrieval_top_k,
                step=1,
                scale=2,
            )
            refresh_models = gr.Button("↻", variant="secondary", scale=0)
            run_btn = gr.Button("Run Evaluation", variant="primary", scale=1)

        with gr.Row():
            dense_weight = gr.Slider(
                label="Dense weight (sparse = 1 - dense)",
                minimum=0.0,
                maximum=1.0,
                value=settings.hybrid_search_weights_dense,
                step=0.05,
            )
            rerank_top_n = gr.Slider(
                label="Rerank top_n",
                minimum=1,
                maximum=10,
                value=settings.rerank_top_n,
                step=1,
            )
            reranker_on = gr.Checkbox(label="Reranker on", value=settings.reranker_on)
            query_rewriting_on = gr.Checkbox(label="Query rewriting on", value=settings.query_rewriting_on)
            query_expansion_on = gr.Checkbox(label="Query expansion on", value=settings.query_expansion_on)

        status_md = gr.Markdown()

        with gr.Row():
            metrics_table = gr.Dataframe(
                label="Metrics",
                headers=["metric", "value"],
                interactive=False,
                wrap=True,
            )
            metrics_plot = gr.Plot(label="Metrics Chart")

        with gr.Row():
            hit_plot = gr.Plot(label="Source Hit vs Miss")
            rank_plot = gr.Plot(label="First Relevant Rank Distribution")
            keyword_indicator = gr.Plot(label="Keyword Hit Rate")

        case_table = gr.Dataframe(
            label="Per-case results",
            headers=[
                "question",
                "expected_source",
                "source_hit",
                "first_relevant_rank",
                "retrieved_sources",
                "keyword_hit_score",
            ],
            interactive=False,
            wrap=True,
        )

        refresh_models.click(fn=_list_ollama_models, inputs=[], outputs=[model_name])

        run_btn.click(
            fn=run_evaluation,
            inputs=[
                jsonl_path,
                top_k,
                model_name,
                dense_weight,
                reranker_on,
                rerank_top_n,
                query_rewriting_on,
                query_expansion_on,
            ],
            outputs=[status_md, metrics_table, metrics_plot, case_table, hit_plot, rank_plot, keyword_indicator],
        )

    return demo
