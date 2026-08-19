import numpy as np

from app.core.models import Chunk, ChunkMetadata
from app.retrieval.ltr import FEATURE_NAMES, LTRFeatureExtractor
from app.retrieval.phase4 import FieldAwareBM25, Phase4Config, Phase4Index, Phase4Retriever, _tokens


def make_chunks() -> list[Chunk]:
    values = [
        ("atlas/api-retries.md", "Atlas API > Retryable responses", "408 429 500 502 503 504 are retryable."),
        ("atlas/access-service-accounts.md", "Atlas Access > API key storage and prefixes", "Keys use atk_."),
        ("atlas/api-pagination.md", "Atlas API > Invalid cursors", "An invalid cursor must be discarded."),
    ]
    return [
        Chunk(
            content=content,
            document_id=f"doc-{index}",
            metadata=ChunkMetadata(
                document_id=f"doc-{index}",
                chunk_id=f"{file_name}::{index}",
                file_path=file_name,
                file_name=file_name,
                extension=".md",
                loader_type="TextLoader",
                chunk_index=index,
                source_char_start=index * 100,
                source_char_end=index * 100 + len(content),
                heading_path=heading,
            ),
        )
        for index, (file_name, heading, content) in enumerate(values)
    ]


class FakeStream:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)
        self.representations = []

    def query(self, query, *, representation, instruction_mode):
        del query, instruction_mode
        self.representations.append(representation)
        return self.scores


def make_retriever() -> Phase4Retriever:
    chunks = make_chunks()
    records = [{"content": chunk.content, "metadata": chunk.metadata.model_dump()} for chunk in chunks]
    index = type("FakeIndex", (), {})()
    index.records = records
    index.streams = {
        "qwen": FakeStream([0.95, 0.20, 0.30]),
        "bge": FakeStream([0.70, 0.80, 0.25]),
        "e5": FakeStream([0.85, 0.40, 0.60]),
    }
    index.bm25 = FieldAwareBM25(records, field_aware=True)
    index.model_errors = {}
    return Phase4Retriever(index)


def test_field_aware_bm25_preserves_identifiers_and_boosts_headings():
    chunks = make_chunks()
    records = [{"content": chunk.content, "metadata": chunk.metadata.model_dump()} for chunk in chunks]
    bm25 = FieldAwareBM25(records, field_aware=True, title_weight=3.0, heading_weight=2.0)

    results = bm25.query("API key storage atk_", n_results=3)

    assert results
    assert results[0][0] == 1
    assert "atk_" in _tokens("atk_")


def test_phase4_index_builds_the_requested_sparse_variant_per_trial():
    index = Phase4Index.from_vector_backends(make_chunks(), dense_backends={})

    plain = index.bm25_for(Phase4Config(field_aware_bm25=False))
    fielded = index.bm25_for(Phase4Config(field_aware_bm25=True))

    assert plain is not fielded
    assert plain.field_weights == {"body": 1.0}
    assert fielded.field_weights["title"] == 3.0


def test_phase4_fusion_exposes_all_independent_stream_ranks_and_raw_content():
    retriever = make_retriever()
    config = Phase4Config(
        top_k=2,
        candidate_depth=3,
        stream_weights=(("qwen", 0.25), ("bge", 0.25), ("e5", 0.25), ("bm25", 0.25)),
        router_on=True,
        field_aware_bm25=True,
    )

    results = retriever.retrieve("Which exact retry status is supported?", config=config)

    assert len(results) == 2
    assert all("source_char_start" in metadata for _, metadata, _ in results)
    assert all("heading_path" in metadata for _, metadata, _ in results)
    trace = retriever.last_trace["candidate_features"]
    assert {"qwen_rank", "bge_rank", "e5_rank", "bm25_rank"}.issubset(trace[0])
    assert retriever.last_trace["routing"]["signals"]["has_lexical_signal"] is False


def test_phase4_instruction_router_is_query_only_and_has_distinct_modes():
    retriever = make_retriever()
    routed = Phase4Config(qwen_instruction_mode="routed", qwen_instruction_routing=True)

    assert retriever._instruction_modes("Which error code is 429?", routed) == ("precision",)
    assert retriever._instruction_modes("What does it mean across the platform?", routed) == ("ambiguity",)
    assert retriever._instruction_modes("How do triggers and approvals interact in a long workflow run?", routed) == (
        "multiple",
    )
    assert retriever._instruction_modes(
        "Explain semantic retrieval behavior for paraphrased technical questions in Atlas.", routed
    ) == (
        "semantic",
    )


def test_phase4_context_and_hierarchy_are_explicit_second_stage_signals():
    retriever = make_retriever()
    config = Phase4Config(
        top_k=2,
        candidate_depth=3,
        stream_weights=(("qwen", 1.0),),
        context_aware=True,
        hierarchical_on=True,
    )

    retriever.retrieve("retry status", config=config)

    assert retriever.index.streams["qwen"].representations == ["context"]
    assert all("hierarchy_score" in record for record in retriever.last_trace["candidate_features"])


def test_phase4_prf_and_hyde_preserve_original_path():
    retriever = make_retriever()
    config = Phase4Config(
        top_k=2,
        candidate_depth=3,
        stream_weights=(("qwen", 1.0),),
        prf_on=True,
        prf_confidence_threshold=1.0,
        hyde_on=True,
    )

    results = retriever.retrieve("retry status", config=config)

    assert results
    assert retriever.last_trace["hyde_status"].startswith("blocked:")
    assert retriever.last_trace["prf_applied"] is True
    assert retriever.last_trace["prf_terms"]


def test_phase4_ltr_features_cover_streams_fields_and_presence():
    features = LTRFeatureExtractor.extract(
        "Which exact API key is used?",
        {
            "content": "atk_123 API key",
            "metadata": {"heading_path": "Atlas > API key storage", "file_name": "a.md"},
            "qwen_score": 0.9,
            "qwen_rank": 1,
            "bge_score": 0.7,
            "bge_rank": 2,
            "e5_score": 0.6,
            "e5_rank": 3,
            "bm25_score": 5.0,
            "bm25_rank": 1,
            "stream_count": 4,
        },
    )

    assert set(("qwen_score", "bge_rank", "e5_score", "bm25_rank", "title_lexical_overlap")) <= set(features)
    assert features["stream_presence"] == 4.0
    assert LTRFeatureExtractor.matrix([features]).shape[1] == len(FEATURE_NAMES)
