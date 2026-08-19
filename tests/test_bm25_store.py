from app.vector_store.bm25_store import BM25Store
from tests.conftest import make_chunk


def make_store(tmp_path) -> BM25Store:
    return BM25Store(persist_dir=str(tmp_path))


class TestTokenize:
    def test_strips_punctuation_and_lowercases(self, tmp_path):
        store = make_store(tmp_path)
        # "what" and "is" are both in the stopword list, so only the
        # meaningful acronym token survives.
        assert store._tokenize("What is SOW?") == ["sow"]

    def test_removes_stopwords_and_single_char_tokens(self, tmp_path):
        store = make_store(tmp_path)
        tokens = store._tokenize("the a of in for and CI")
        assert tokens == ["ci"]

    def test_empty_text_returns_no_tokens(self, tmp_path):
        store = make_store(tmp_path)
        assert store._tokenize("") == []
        assert store._tokenize(None) == []


class TestAddChunksDedup:
    def test_duplicate_chunk_is_not_added_twice(self, tmp_path):
        store = make_store(tmp_path)
        chunk = make_chunk("Hello world", file_path="a.md", chunk_index=0)

        store.add_chunks([chunk])
        store.add_chunks([chunk])

        assert len(store.chunks) == 1

    def test_distinct_chunks_are_both_kept(self, tmp_path):
        store = make_store(tmp_path)
        chunk_a = make_chunk("same text", file_path="a.md", chunk_index=0)
        chunk_b = make_chunk("same text", file_path="b.md", chunk_index=0)

        store.add_chunks([chunk_a, chunk_b])

        assert len(store.chunks) == 2

    def test_empty_chunk_list_is_a_noop(self, tmp_path):
        store = make_store(tmp_path)
        store.add_chunks([])
        assert store.chunks == []
        assert store.bm25 is None


class TestQuery:
    def test_empty_index_returns_no_results(self, tmp_path):
        store = make_store(tmp_path)
        assert store.query("anything") == []

    def test_query_with_no_matching_tokens_returns_empty(self, tmp_path):
        store = make_store(tmp_path)
        store.add_chunks([make_chunk("Dependency injection pattern", file_path="a.md", chunk_index=0)])
        # "the" and "a" are stopwords -> nothing to search on.
        assert store.query("the a") == []

    def test_query_returns_relevant_chunk_first(self, tmp_path):
        # BM25 idf collapses to ~0 in tiny corpora where a term appears in
        # most documents, so this needs enough distractor docs for the
        # signal to be meaningful.
        store = make_store(tmp_path)
        chunks = [make_chunk("Dependency injection is a design pattern", file_path="a.md", chunk_index=0)]
        chunks += [
            make_chunk(f"unrelated cooking recipe number {i} about food", file_path=f"x{i}.md", chunk_index=0)
            for i in range(4)
        ]
        store.add_chunks(chunks)

        results = store.query("dependency injection")

        assert len(results) == 1
        assert results[0][0] == "Dependency injection is a design pattern"

    def test_query_respects_n_results_limit(self, tmp_path):
        store = make_store(tmp_path)
        target_chunks = [
            make_chunk(f"topic keyword number {i} variant", file_path=f"target{i}.md", chunk_index=0) for i in range(3)
        ]
        distractor_chunks = [
            make_chunk(f"unrelated filler content {i}", file_path=f"noise{i}.md", chunk_index=0) for i in range(10)
        ]
        store.add_chunks(target_chunks + distractor_chunks)

        results = store.query("topic keyword", n_results=2)

        assert len(results) == 2


class TestPersistence:
    def test_chunks_survive_save_and_reload(self, tmp_path):
        store = make_store(tmp_path)
        chunks = [make_chunk("Persisted unique searchable content", file_path="a.md", chunk_index=0)]
        chunks += [
            make_chunk(f"unrelated filler content {i}", file_path=f"noise{i}.md", chunk_index=0) for i in range(4)
        ]
        store.add_chunks(chunks)

        reloaded = BM25Store(persist_dir=str(tmp_path))

        assert len(reloaded.chunks) == 5
        assert reloaded.chunks[0]["content"] == "Persisted unique searchable content"
        assert reloaded.query("persisted searchable") != []
