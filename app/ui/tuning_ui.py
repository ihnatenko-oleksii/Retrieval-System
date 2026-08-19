import itertools
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from app.core.config import settings
from app.core.runtime_config import RetrievalConfig
from app.evals.evaluator import Evaluator
from app.retrieval.reranker import Reranker
from app.vector_store.bm25_store import BM25Store
from app.vector_store.chroma_store import VectorStore

# Composite score weights (confirmed with user).
W_KEYWORD = 0.6
W_MRR = 0.25
W_RECALL = 0.15

DEFAULT_TOP_K = [3, 5, 7]
DEFAULT_DENSE = [0.5, 0.7]
DEFAULT_BOOL = [False, True]
MAX_TRIALS_DEFAULT = 48
AVG_SECONDS_PER_TRIAL_DEFAULT = 62.5


def _list_ollama_models() -> list[str]:
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


def _parse_int_list(text: str, fallback: list[int]) -> list[int]:
    if not text:
        return list(fallback)
    values = []
    for tok in str(text).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            values.append(int(float(tok)))
        except ValueError:
            continue
    seen, out = set(), []
    for v in values:
        if v not in seen and v > 0:
            out.append(v)
            seen.add(v)
    return out or list(fallback)


def _parse_float_list(text: str, fallback: list[float]) -> list[float]:
    if not text:
        return list(fallback)
    values = []
    for tok in str(text).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            f = float(tok)
        except ValueError:
            continue
        if 0.0 <= f <= 1.0:
            values.append(round(f, 4))
    seen, out = set(), []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out or list(fallback)


def _parse_bool_list(selected: list[str]) -> list[bool]:
    if not selected:
        return [False, True]
    mapping = {"on": True, "off": False, "true": True, "false": False}
    values = []
    for item in selected:
        v = mapping.get(str(item).strip().lower())
        if v is not None and v not in values:
            values.append(v)
    return values or [False, True]


def _format_duration(seconds: float) -> str:
    sec = max(0, int(round(seconds)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def estimate_runtime(
    top_k_text: str,
    dense_text: str,
    reranker_choices: list[str],
    rewriting_choices: list[str],
    expansion_choices: list[str],
    llm_models: list[str],
    max_trials: int,
    avg_seconds_per_trial: float,
) -> str:
    top_ks = _parse_int_list(top_k_text, DEFAULT_TOP_K)
    dense_ws = _parse_float_list(dense_text, DEFAULT_DENSE)
    reranker_vals = _parse_bool_list(reranker_choices)
    rewriting_vals = _parse_bool_list(rewriting_choices)
    expansion_vals = _parse_bool_list(expansion_choices)
    models = list(llm_models) if llm_models else [settings.llm_model]

    total = len(top_ks) * len(dense_ws) * len(reranker_vals) * len(rewriting_vals) * len(expansion_vals) * len(models)

    est_per_trial = max(1.0, float(avg_seconds_per_trial))
    est_total = total * est_per_trial
    cap = int(max_trials) if max_trials and int(max_trials) > 0 else MAX_TRIALS_DEFAULT
    over_cap = total > cap

    lines = [
        "### Runtime estimate",
        f"- Trials to run: **{total}**",
        f"- Estimated time per trial: **~{_format_duration(est_per_trial)}**",
        f"- Estimated total runtime: **~{_format_duration(est_total)}**",
        f"- Max trials cap: **{cap}**",
    ]
    if over_cap:
        lines.append(f"- ⚠️ Current selection exceeds cap by **{total - cap}** trials.")
    lines.append(
        "- Combination count = `len(top_k) * len(dense_weight) * len(reranker_on) * "
        "len(query_rewriting_on) * len(query_expansion_on) * len(llm_models)`"
    )
    return "\n".join(lines)


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=340,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _composite(metrics: dict, top_k: int) -> float:
    recall = float(metrics.get(f"recall@{top_k}", 0.0))
    mrr = float(metrics.get("mrr", 0.0))
    kw = float(metrics.get("keyword_hit_rate", 0.0))
    return round(W_KEYWORD * kw + W_MRR * mrr + W_RECALL * recall, 4)


def _best_config_md(row: dict) -> str:
    lines = ["### Best configuration", ""]
    lines.append(f"- Composite score: **{row['composite']:.4f}**")
    lines.append(f"- top_k: `{row['top_k']}`")
    lines.append(f"- dense_weight: `{row['dense_weight']}` (sparse = `{round(1 - row['dense_weight'], 4)}`)")
    lines.append(f"- reranker_on: `{row['reranker_on']}`")
    lines.append(f"- query_rewriting_on: `{row['query_rewriting_on']}`")
    lines.append(f"- query_expansion_on: `{row['query_expansion_on']}`")
    lines.append(f"- llm_model: `{row['llm_model']}`")
    lines.append("")
    lines.append(
        f"Metrics: recall@k=**{row['recall']:.4f}**, mrr=**{row['mrr']:.4f}**, "
        f"keyword_hit_rate=**{row['keyword_hit_rate']:.4f}**, "
        f"precision@k=**{row['precision']:.4f}**, ndcg=**{row['ndcg']:.4f}**"
    )
    return "\n".join(lines)


def _top_trials_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Top trials by composite score")
    top = df.head(10).copy()
    top["label"] = top.apply(
        lambda r: (
            f"k={r['top_k']} d={r['dense_weight']} rr={int(bool(r['reranker_on']))} "
            f"qr={int(bool(r['query_rewriting_on']))} qe={int(bool(r['query_expansion_on']))}"
        ),
        axis=1,
    )
    fig = go.Figure(
        data=[
            go.Bar(
                x=top["composite"],
                y=top["label"],
                orientation="h",
                marker_color="#7c3aed",
                text=[f"{v:.4f}" for v in top["composite"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Top trials by composite score",
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="composite",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _heatmap_topk_vs_dense(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _empty_figure("Composite heatmap: top_k vs dense_weight")
    pivot = (
        df.groupby(["top_k", "dense_weight"])["composite"]
        .mean()
        .reset_index()
        .pivot(index="top_k", columns="dense_weight", values="composite")
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return _empty_figure("Composite heatmap: top_k vs dense_weight")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(r) for r in pivot.index],
            colorscale="Viridis",
            text=[[f"{v:.3f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title="composite"),
        )
    )
    fig.update_layout(
        title="Composite heatmap: top_k vs dense_weight (mean over other params)",
        template="plotly_dark",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="dense_weight",
        yaxis_title="top_k",
    )
    return fig


def run_sweep(
    jsonl_path: str,
    top_k_text: str,
    dense_text: str,
    reranker_choices: list[str],
    rewriting_choices: list[str],
    expansion_choices: list[str],
    llm_models: list[str],
    rerank_top_n: int,
    max_trials: int,
):
    eval_path = Path(jsonl_path).expanduser()
    if not eval_path.exists():
        return (
            f"❌ File not found: `{eval_path}`",
            "",
            pd.DataFrame(),
            _empty_figure("Top trials by composite score"),
            _empty_figure("Composite heatmap: top_k vs dense_weight"),
        )

    top_ks = _parse_int_list(top_k_text, DEFAULT_TOP_K)
    dense_ws = _parse_float_list(dense_text, DEFAULT_DENSE)
    reranker_vals = _parse_bool_list(reranker_choices)
    rewriting_vals = _parse_bool_list(rewriting_choices)
    expansion_vals = _parse_bool_list(expansion_choices)
    models = list(llm_models) if llm_models else [settings.llm_model]

    combos = list(itertools.product(top_ks, dense_ws, reranker_vals, rewriting_vals, expansion_vals, models))
    total = len(combos)

    if total == 0:
        return (
            "❌ Empty candidate set.",
            "",
            pd.DataFrame(),
            _empty_figure("Top trials by composite score"),
            _empty_figure("Composite heatmap: top_k vs dense_weight"),
        )

    cap = int(max_trials) if max_trials and int(max_trials) > 0 else MAX_TRIALS_DEFAULT
    if total > cap:
        return (
            f"❌ {total} trials exceed max_trials={cap}. Narrow the candidate sets or raise the cap.",
            "",
            pd.DataFrame(),
            _empty_figure("Top trials by composite score"),
            _empty_figure("Composite heatmap: top_k vs dense_weight"),
        )

    # Share heavy components across trials.
    vector_store = VectorStore()
    bm25_store = BM25Store()
    needs_reranker = any(reranker_vals)
    reranker = Reranker(force_load=needs_reranker) if needs_reranker else Reranker(force_load=False)

    rows = []
    t0 = time.time()
    for top_k, dense_w, rr_on, qr_on, qe_on, model in combos:
        sparse_w = round(1.0 - float(dense_w), 4)
        cfg = RetrievalConfig(
            top_k=int(top_k),
            dense_weight=float(dense_w),
            sparse_weight=sparse_w,
            reranker_on=bool(rr_on),
            rerank_top_n=int(rerank_top_n),
            query_rewriting_on=bool(qr_on),
            query_expansion_on=bool(qe_on),
            llm_model=model,
        )
        evaluator = Evaluator(
            config=cfg,
            vector_store=vector_store,
            bm25_store=bm25_store,
            reranker=reranker,
        )
        metrics, _ = evaluator.evaluate_cases(str(eval_path))
        if not metrics:
            continue

        recall = float(metrics.get(f"recall@{int(top_k)}", 0.0))
        precision = float(metrics.get(f"precision@{int(top_k)}", 0.0))
        mrr = float(metrics.get("mrr", 0.0))
        ndcg = float(metrics.get("ndcg", 0.0))
        kw = float(metrics.get("keyword_hit_rate", 0.0))
        composite = _composite(metrics, int(top_k))

        rows.append(
            {
                "top_k": int(top_k),
                "dense_weight": float(dense_w),
                "sparse_weight": sparse_w,
                "reranker_on": bool(rr_on),
                "query_rewriting_on": bool(qr_on),
                "query_expansion_on": bool(qe_on),
                "llm_model": model,
                "composite": composite,
                "keyword_hit_rate": round(kw, 4),
                "mrr": round(mrr, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "ndcg": round(ndcg, 4),
            }
        )

    elapsed = time.time() - t0
    if not rows:
        return (
            "❌ No trials produced metrics. Check the JSONL file.",
            "",
            pd.DataFrame(),
            _empty_figure("Top trials by composite score"),
            _empty_figure("Composite heatmap: top_k vs dense_weight"),
        )

    df = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    best_md = _best_config_md(best)

    status = (
        f"✅ Ran **{len(df)}/{total}** trials in **{elapsed:.1f}s** "
        f"on `{eval_path}`. Metric: **{W_KEYWORD} * keyword_hit_rate "
        f"+ {W_MRR} * MRR + {W_RECALL} * recall@K**."
    )
    return status, best_md, df, _top_trials_bar(df), _heatmap_topk_vs_dense(df)


def build_tuning_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Monochrome(), title="RAG Tuning Dashboard") as demo:
        gr.Markdown("## RAG Tuning Dashboard")
        gr.Markdown(
            "Sweep runtime retrieval parameters across an evals JSONL and pick the best config by composite score."
        )
        gr.Markdown(f"**Composite** = {W_KEYWORD} * keyword_hit_rate + {W_MRR} * MRR + {W_RECALL} * recall@K")

        with gr.Row():
            jsonl_path = gr.Textbox(
                label="Eval JSONL path",
                value="./evals-json.jsonl",
                scale=4,
            )
            max_trials = gr.Slider(
                label="Max trials",
                minimum=1,
                maximum=256,
                value=MAX_TRIALS_DEFAULT,
                step=1,
                scale=1,
            )
            run_btn = gr.Button("Run Sweep", variant="primary", scale=1)

        with gr.Row():
            top_k_text = gr.Textbox(
                label="top_k candidates (comma-separated)",
                value=", ".join(str(v) for v in DEFAULT_TOP_K),
            )
            dense_text = gr.Textbox(
                label="dense_weight candidates (0.0-1.0)",
                value=", ".join(str(v) for v in DEFAULT_DENSE),
            )

        with gr.Row():
            reranker_choices = gr.CheckboxGroup(
                label="reranker_on candidates",
                choices=["off", "on"],
                value=["off", "on"],
            )
            rewriting_choices = gr.CheckboxGroup(
                label="query_rewriting_on candidates",
                choices=["off", "on"],
                value=["off", "on"],
            )
            expansion_choices = gr.CheckboxGroup(
                label="query_expansion_on candidates",
                choices=["off", "on"],
                value=["off", "on"],
            )

        with gr.Row():
            llm_models = gr.Dropdown(
                label="LLM models to try",
                choices=_list_ollama_models(),
                value=[settings.llm_model],
                multiselect=True,
                allow_custom_value=True,
                scale=3,
            )
            rerank_top_n = gr.Slider(
                label="rerank_top_n (fixed)",
                minimum=1,
                maximum=10,
                value=settings.rerank_top_n,
                step=1,
                scale=1,
            )

        with gr.Row():
            avg_seconds_per_trial = gr.Number(
                label="Estimated seconds per trial",
                value=AVG_SECONDS_PER_TRIAL_DEFAULT,
                minimum=1,
                precision=1,
            )

        estimate_md = gr.Markdown()
        status_md = gr.Markdown()
        best_md = gr.Markdown()

        with gr.Row():
            bar_plot = gr.Plot(label="Top trials by composite")
            heatmap_plot = gr.Plot(label="Composite heatmap")

        results_table = gr.Dataframe(
            label="All trials (sorted by composite)",
            interactive=False,
            wrap=True,
        )

        run_btn.click(
            fn=run_sweep,
            inputs=[
                jsonl_path,
                top_k_text,
                dense_text,
                reranker_choices,
                rewriting_choices,
                expansion_choices,
                llm_models,
                rerank_top_n,
                max_trials,
            ],
            outputs=[status_md, best_md, results_table, bar_plot, heatmap_plot],
        )

        estimate_inputs = [
            top_k_text,
            dense_text,
            reranker_choices,
            rewriting_choices,
            expansion_choices,
            llm_models,
            max_trials,
            avg_seconds_per_trial,
        ]

        demo.load(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        top_k_text.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        dense_text.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        reranker_choices.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        rewriting_choices.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        expansion_choices.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        llm_models.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        max_trials.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])
        avg_seconds_per_trial.change(fn=estimate_runtime, inputs=estimate_inputs, outputs=[estimate_md])

    return demo
