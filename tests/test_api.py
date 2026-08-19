from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api import endpoints
from tests.conftest import make_meta


@pytest.fixture
def client():
    # Default every store dependency to a cheap mock so no test can
    # accidentally trigger a real embedding-model download or Chroma/BM25
    # disk access; individual tests override further as needed. FastAPI
    # resolves Depends() providers even when body validation ultimately
    # fails, so this must cover validation-error tests too.
    endpoints.app.dependency_overrides[endpoints.get_vector_store] = lambda: MagicMock()
    endpoints.app.dependency_overrides[endpoints.get_bm25_store] = lambda: MagicMock()
    endpoints.app.dependency_overrides[endpoints.get_retriever] = lambda: MagicMock()
    with TestClient(endpoints.app) as c:
        yield c
    endpoints.app.dependency_overrides.clear()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class TestAskEndpoint:
    def test_blank_query_is_rejected(self, client):
        response = client.post("/ask", json={"query": "   "})
        assert response.status_code == 422

    def test_missing_query_is_rejected(self, client):
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_top_k_below_one_is_rejected(self, client):
        response = client.post("/ask", json={"query": "hello", "top_k": 0})
        assert response.status_code == 422

    def test_no_chunks_found_short_circuits_without_calling_generator(self, client, monkeypatch):
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = []
        endpoints.app.dependency_overrides[endpoints.get_retriever] = lambda: fake_retriever

        def fail_if_called(*args, **kwargs):
            raise AssertionError("ollama.chat should not be called when there are no chunks")

        monkeypatch.setattr("app.generation.generator.ollama.chat", fail_if_called)

        response = client.post("/ask", json={"query": "hello"})

        assert response.status_code == 200
        assert response.json() == {"final_answer": "No relevant context found.", "retrieved_sources": []}

    def test_successful_ask_returns_answer_and_sources(self, client, monkeypatch):
        meta = make_meta("docs/a.md", 0, file_name="a.md")
        fake_retriever = MagicMock()
        fake_retriever.retrieve.return_value = [("chunk content", meta, 0.9)]
        endpoints.app.dependency_overrides[endpoints.get_retriever] = lambda: fake_retriever

        monkeypatch.setattr(
            "app.generation.generator.ollama.chat",
            lambda **kw: {"message": {"content": "Grounded answer [1]"}},
        )

        response = client.post("/ask", json={"query": "hello", "top_k": 1})

        assert response.status_code == 200
        body = response.json()
        assert body["final_answer"] == "Grounded answer [1]"
        assert body["retrieved_sources"][0]["file_name"] == "a.md"


class TestIngestEndpoint:
    def test_missing_directory_returns_400(self, client):
        response = client.post("/ingest", json={"directory": "/no/such/directory"})
        assert response.status_code == 400

    def test_blank_directory_is_rejected(self, client):
        response = client.post("/ingest", json={"directory": ""})
        assert response.status_code == 422

    def test_empty_directory_reports_zero_chunks(self, client, tmp_path):
        fake_vector_store = MagicMock()
        fake_bm25_store = MagicMock()
        endpoints.app.dependency_overrides[endpoints.get_vector_store] = lambda: fake_vector_store
        endpoints.app.dependency_overrides[endpoints.get_bm25_store] = lambda: fake_bm25_store

        response = client.post("/ingest", json={"directory": str(tmp_path)})

        assert response.status_code == 200
        assert response.json() == {"status": "success", "chunks_indexed": 0, "message": "No documents found."}
        fake_vector_store.add_chunks.assert_not_called()
        fake_bm25_store.add_chunks.assert_not_called()

    def test_directory_with_file_indexes_chunks_into_both_stores(self, client, tmp_path):
        (tmp_path / "note.md").write_text("Some content to ingest for testing.", encoding="utf-8")
        fake_vector_store = MagicMock()
        fake_bm25_store = MagicMock()
        endpoints.app.dependency_overrides[endpoints.get_vector_store] = lambda: fake_vector_store
        endpoints.app.dependency_overrides[endpoints.get_bm25_store] = lambda: fake_bm25_store

        response = client.post("/ingest", json={"directory": str(tmp_path)})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["chunks_indexed"] > 0
        fake_vector_store.add_chunks.assert_called_once()
        fake_bm25_store.add_chunks.assert_called_once()
