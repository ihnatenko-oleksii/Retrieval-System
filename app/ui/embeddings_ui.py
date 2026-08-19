from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from app.vector_store.chroma_store import VectorStore


def _normalize_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _truncate(text: str, length: int = 120) -> str:
    clean = (text or "").strip().replace("\n", " ")
    return clean if len(clean) <= length else clean[:length] + "..."


def _group_from_metadata(metadata: dict | None, color_by: str) -> str:
    if not isinstance(metadata, dict):
        return "unknown"

    file_name = str(metadata.get("file_name") or "").strip()
    extension = str(metadata.get("extension") or "").strip().lstrip(".").lower()
    file_path = str(metadata.get("file_path") or "").strip()
    loader_type = str(metadata.get("loader_type") or "").strip()

    if color_by == "file_name":
        return file_name or "unknown_file"
    if color_by == "folder":
        if file_path:
            folder = Path(file_path).parent.name
            return folder or "root"
        return "unknown_folder"
    if color_by == "loader_type":
        return loader_type or "unknown_loader"
    return extension or "unknown_ext"


def _collapse_rare_groups(values: list[str], top_groups: int) -> list[str]:
    counts = Counter(values)
    keep = {name for name, _ in counts.most_common(max(1, int(top_groups)))}
    return [value if value in keep else "other" for value in values]


def _auto_perplexity(sample_size: int) -> int:
    return max(5, min(50, int(np.sqrt(sample_size))))


def _auto_category_count(sample_size: int) -> int:
    return max(2, min(12, int(np.sqrt(sample_size / 2))))


def _semantic_category_groups(vectors: np.ndarray, documents: list, category_count: int, seed: int) -> list[str]:
    n_points = len(vectors)
    if n_points < 2:
        return ["category_1"] * n_points

    k = max(2, min(int(category_count), n_points))
    kmeans = KMeans(n_clusters=k, random_state=int(seed), n_init="auto")
    labels = kmeans.fit_predict(vectors)

    docs = [str(doc or "") for doc in documents]
    label_names = {label: f"category_{label + 1}" for label in range(k)}

    # Best-effort naming using top tf-idf terms per cluster.
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        tfidf = vectorizer.fit_transform(docs)
        terms = vectorizer.get_feature_names_out()
        for label in range(k):
            idx = np.where(labels == label)[0]
            if len(idx) == 0:
                continue
            cluster_mean = np.asarray(tfidf[idx].mean(axis=0)).ravel()
            top_term_ids = cluster_mean.argsort()[-3:][::-1]
            top_terms = [terms[t] for t in top_term_ids if cluster_mean[t] > 0]
            if top_terms:
                label_names[label] = f"cat_{label + 1}: " + ", ".join(top_terms)
    except Exception:
        pass

    return [label_names[int(label)] for label in labels]


def build_embedding_plot(
    perplexity: int,
    max_points: int,
    color_by: str,
    top_groups: int,
    seed: int,
    category_count: int,
):
    try:
        store = VectorStore()
        data = store.collection.get(include=["embeddings", "documents", "metadatas"])
    except Exception as exc:
        return f"❌ Failed to load embeddings: `{exc}`", None, pd.DataFrame(columns=["group", "count", "percent"])

    embeddings = _normalize_list(data.get("embeddings"))
    documents = _normalize_list(data.get("documents"))
    metadatas = _normalize_list(data.get("metadatas"))

    if not embeddings:
        return "❌ No embeddings found in ChromaDB.", None, pd.DataFrame(columns=["group", "count", "percent"])

    vectors = np.array(embeddings)
    total_points = len(vectors)
    sample_size = min(int(max_points), total_points)

    if sample_size < 2:
        return "❌ Need at least 2 vectors for t-SNE.", None, pd.DataFrame(columns=["group", "count", "percent"])

    if sample_size < total_points:
        rng = np.random.default_rng(seed=int(seed))
        indices = rng.choice(total_points, size=sample_size, replace=False)
        vectors = vectors[indices]
        documents = [documents[i] for i in indices]
        metadatas = [metadatas[i] for i in indices]

    effective_perplexity = _auto_perplexity(sample_size) if int(perplexity) <= 0 else int(perplexity)
    effective_perplexity = max(1, min(effective_perplexity, sample_size - 1))

    # Stabilize t-SNE for high-dimensional sentence embeddings.
    if vectors.ndim == 2 and vectors.shape[1] > 50 and sample_size > 50:
        pca_dims = min(50, vectors.shape[1], sample_size - 1)
        vectors = PCA(n_components=pca_dims, random_state=42).fit_transform(vectors)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=effective_perplexity,
        metric="cosine",
        learning_rate="auto",
        init="pca",
    )
    reduced_vectors = tsne.fit_transform(vectors)

    if color_by == "semantic_category":
        effective_categories = _auto_category_count(sample_size) if int(category_count) <= 0 else int(category_count)
        groups = _semantic_category_groups(vectors, documents, effective_categories, int(seed))
    else:
        groups = [_group_from_metadata(meta, color_by) for meta in metadatas]
        groups = _collapse_rare_groups(groups, int(top_groups))
    unique_groups = sorted(set(groups))
    palette = [
        "#3b82f6",
        "#10b981",
        "#f97316",
        "#a855f7",
        "#eab308",
        "#06b6d4",
        "#ef4444",
        "#8b5cf6",
        "#84cc16",
        "#ec4899",
    ]
    color_map = {group: palette[i % len(palette)] for i, group in enumerate(unique_groups)}

    fig = go.Figure()
    for group in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == group]
        fig.add_trace(
            go.Scatter(
                x=reduced_vectors[idx, 0],
                y=reduced_vectors[idx, 1],
                mode="markers",
                name=group,
                marker=dict(size=6, color=color_map[group], opacity=0.82),
                text=[
                    (
                        f"Group: {groups[i]}<br>"
                        f"File: {metadatas[i].get('file_name', 'n/a') if isinstance(metadatas[i], dict) else 'n/a'}<br>"
                        f"Chunk: {metadatas[i].get('chunk_index', 'n/a') if isinstance(metadatas[i], dict) else 'n/a'}<br>"
                        f"Text: {_truncate(documents[i])}"
                    )
                    for i in idx
                ],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="2D Chroma Vector Store Visualization",
        width=900,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40),
        xaxis_title="t-SNE x",
        yaxis_title="t-SNE y",
        legend_title=f"Grouped by: {color_by}",
    )

    distribution = Counter(groups)
    total_visible = max(1, len(groups))
    dist_df = pd.DataFrame(
        [
            {"group": group, "count": count, "percent": round(100.0 * count / total_visible, 2)}
            for group, count in sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        ]
    )

    status = (
        f"✅ Visualized **{sample_size}** embeddings"
        f" (from **{total_points}** total). "
        f"Perplexity: **{effective_perplexity}**. "
        f"Coloring: **{color_by}**. "
        f"Groups: **{len(unique_groups)}**."
    )
    return status, fig, dist_df


def build_embeddings_ui() -> gr.Blocks:
    with gr.Blocks(title="Embeddings Explorer") as demo:
        gr.Markdown("## Embeddings Explorer")
        gr.Markdown("Visualize your stored ChromaDB embeddings with 2D t-SNE.")

        with gr.Row():
            perplexity = gr.Slider(
                label="t-SNE perplexity",
                minimum=0,
                maximum=80,
                value=0,
                step=1,
                info="0 = auto (recommended)",
            )
            max_points = gr.Slider(
                label="Max points to visualize",
                minimum=100,
                maximum=5000,
                value=1500,
                step=100,
            )
            color_by = gr.Dropdown(
                label="Color by",
                choices=["semantic_category", "extension", "file_name", "folder", "loader_type"],
                value="semantic_category",
            )
            category_count = gr.Slider(
                label="Semantic categories",
                minimum=0,
                maximum=20,
                value=0,
                step=1,
                info="0 = auto (only for semantic_category)",
            )
            top_groups = gr.Slider(
                label="Show top groups",
                minimum=3,
                maximum=30,
                value=12,
                step=1,
                info="Pozostałe grupy trafiają do 'other'",
            )
            seed = gr.Slider(
                label="Sampling seed",
                minimum=1,
                maximum=9999,
                value=42,
                step=1,
            )
            refresh_btn = gr.Button("Refresh Visualization", variant="primary")

        status = gr.Markdown()
        plot = gr.Plot(label="Embedding scatter plot")
        type_table = gr.Dataframe(
            label="Group distribution",
            headers=["group", "count", "percent"],
            interactive=False,
            wrap=True,
        )

        refresh_btn.click(
            fn=build_embedding_plot,
            inputs=[perplexity, max_points, color_by, top_groups, seed, category_count],
            outputs=[status, plot, type_table],
        )
        demo.load(
            fn=build_embedding_plot,
            inputs=[perplexity, max_points, color_by, top_groups, seed, category_count],
            outputs=[status, plot, type_table],
        )

    return demo
