from app.chunking.splitter import RecursiveCharacterTextSplitter, get_splitter
from app.core.models import Document, DocumentMetadata


def make_document(content: str) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(file_path="docs/a.md", file_name="a.md", extension=".md", loader_type="TextLoader"),
    )


class TestSplitTextEdgeCases:
    def test_empty_text_returns_no_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        assert splitter.split_text("") == []

    def test_text_shorter_than_chunk_size_is_a_single_chunk(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        assert splitter.split_text("Hello world") == ["Hello world"]

    def test_long_text_without_separators_splits_by_char_budget(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        chunks = splitter.split_text("x" * 50)
        assert len(chunks) == 5
        assert all(len(c) == 10 for c in chunks)
        assert "".join(chunks) == "x" * 50

    def test_chunks_respect_configured_overlap(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "aaaa bbbb cccc dddd eeee ffff"
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        # Consecutive chunks must share at least one overlapping word.
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        assert first_words & second_words


class TestSplitDocument:
    def test_chunk_index_is_sequential_and_metadata_is_propagated(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        document = make_document("x" * 30)

        chunks = splitter.split_document(document)

        assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.document_id == document.id for c in chunks)
        assert all(c.metadata.file_path == "docs/a.md" for c in chunks)
        assert all(c.metadata.file_name == "a.md" for c in chunks)

    def test_empty_document_produces_no_chunks(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document = make_document("")

        assert splitter.split_document(document) == []

    def test_chunks_expose_source_offsets_and_heading_context(self):
        content = "# Guide\n## Retries\nUse the original key.\n## Limits\nWait for reset."
        splitter = RecursiveCharacterTextSplitter(chunk_size=24, chunk_overlap=0)
        chunks = splitter.split_document(make_document(content))

        assert chunks
        for chunk in chunks:
            start = chunk.metadata.source_char_start
            end = chunk.metadata.source_char_end
            assert start is not None
            assert end == start + len(chunk.content)
            assert content[start:end] == chunk.content
            assert chunk.metadata.heading_path.startswith("Guide")


def test_get_splitter_returns_recursive_character_splitter():
    assert isinstance(get_splitter(), RecursiveCharacterTextSplitter)
