import logging
import os

from app.chunking.splitter import get_splitter
from app.core.models import Chunk
from app.ingestion.loaders import get_loader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.splitter = get_splitter()

    def process_directory(self, directory_path: str) -> list[Chunk]:
        base_dir = os.path.abspath(directory_path)
        all_chunks = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.startswith("."):
                    continue
                file_path = os.path.join(root, file)
                loader = get_loader(file_path)
                if not loader:
                    logger.info(f"Skipping unsupported file: {file_path}")
                    continue

                try:
                    documents = loader.load(file_path)
                    for doc in documents:
                        # Store a stable, eval-friendly identifier in `file_name`.
                        # Basename-only breaks sources like "EPL/Analiza/.../file.md".
                        abs_path = os.path.abspath(doc.metadata.file_path)
                        rel_path = os.path.relpath(abs_path, base_dir)
                        doc.metadata.file_name = rel_path.replace(os.sep, "/")
                        chunks = self.splitter.split_document(doc)
                        all_chunks.extend(chunks)
                    logger.info(f"Successfully processed: {file_path}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")

        return all_chunks
