# FILE: customer_support_rag/ingestion/pipeline.py
"""End-to-end ingestion pipeline orchestrator.

Coordinates the full ingestion workflow:
    source_dir → load (MD + PDF) → chunk → embed → store in ChromaDB

The `IngestionPipeline` is the single entry point for ingestion work.
It is called by:
    - `POST /ingest` endpoint (api/routes/ingest.py) as a background task
    - `make ingest` Makefile target for CLI-based ingestion

The actual embed+store step (Phase 1B) is stubbed here with a detailed
TODO so it can be wired up once `retrieval.vector_store` is implemented.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from llama_index.core import Document

from core.exceptions import IngestionError, LoaderError
from core.logging import get_logger
from core.types import DocumentChunk, IngestionResult
from ingestion.chunkers import SemanticChunker
from ingestion.loaders import MarkdownLoader, PDFLoader, TextLoader

logger = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates the end-to-end document ingestion workflow.

    Stages:
        1. LOAD   — MarkdownLoader + PDFLoader read source files into Documents.
        2. CHUNK  — SemanticChunker splits Documents into DocumentChunks.
        3. EMBED & STORE — (Phase 1B) VectorStoreManager embeds and upserts chunks.

    This class is intentionally stateless — all state lives in the returned
    IngestionResult. This makes it safe to call concurrently from multiple
    background tasks.

    Example:
        pipeline = IngestionPipeline()
        result = await pipeline.run(Path("./data/docs"))
        print(result.total_chunks)  # e.g. 312
    """

    def __init__(self) -> None:
        self.markdown_loader: MarkdownLoader = MarkdownLoader()
        self.pdf_loader: PDFLoader = PDFLoader()
        self.text_loader: TextLoader = TextLoader()
        self.chunker: SemanticChunker = SemanticChunker()

    async def run(self, source_dir: Path) -> IngestionResult:
        """Execute the full ingestion pipeline for all documents in source_dir.

        Discovers .md, .txt and .pdf files recursively under source_dir. Files that
        fail loading are recorded in `IngestionResult.failed_documents` and
        skipped — the pipeline does not abort on single-file failures.

        Args:
            source_dir: Root directory containing source support documents.
                Must exist and be readable.

        Returns:
            IngestionResult with total counts and any individual file failures.

        Raises:
            IngestionError: If the directory does not exist or if the chunking
                or embedding stages fail entirely (not per-file failures).

        # TODO — PHASE 1A (LOAD + CHUNK):
        #   1. Validate source_dir: if not source_dir.exists(), raise IngestionError.
        #   2. Start timer: `start = time.monotonic()`
        #
        #   LOAD PHASE:
        #   3. Load Markdown: `md_docs = await self.markdown_loader.load_directory(source_dir)`
        #      Catch LoaderError per file using load_directory's internal skip logic.
        #   4. Find PDFs: `pdf_paths = list(source_dir.rglob("*.pdf"))`
        #   5. For each pdf_path, call `await self.pdf_loader.load(pdf_path)`.
        #      Wrap in try/except LoaderError:
        #        - On success: extend pdf_docs list.
        #        - On LoaderError: append str(pdf_path) to `failed_documents`.
        #   6. Combine: `all_docs = md_docs + pdf_docs`
        #   7. Log: logger.info("load_phase_complete", doc_count=len(all_docs))
        #
        #   CHUNK PHASE:
        #   8. `chunks = await self.chunker.chunk(all_docs)`
        #   9. Log: logger.info("chunk_phase_complete", chunk_count=len(chunks))
        #
        # TODO — PHASE 1B (EMBED + STORE — implement after retrieval/ is built):
        #   10. Import VectorStoreManager from retrieval.vector_store.
        #   11. Convert DocumentChunk list to LlamaIndex TextNode list:
        #         from llama_index.core.schema import TextNode
        #         nodes = [
        #             TextNode(
        #                 text=chunk.content,
        #                 id_=chunk.chunk_id,
        #                 metadata=chunk.metadata,
        #             )
        #             for chunk in chunks
        #         ]
        #   12. Call: `await VectorStoreManager().add_nodes(nodes)`
        #       This triggers embedding generation + ChromaDB upsert.
        #
        #   BUILD RESULT:
        #   13. `duration = time.monotonic() - start`
        #   14. Return IngestionResult(
        #           total_documents=len(all_docs),
        #           total_chunks=len(chunks),
        #           failed_documents=failed_documents,
        #           success=len(failed_documents) == 0,
        #           duration_seconds=round(duration, 3),
        #       )
        #   15. Log summary: logger.info("ingestion_complete",
        #           total_docs=len(all_docs), total_chunks=len(chunks),
        #           failures=len(failed_documents), duration_s=duration)
        """
        if not source_dir.exists():
            raise IngestionError(
                f"Source directory does not exist: {source_dir}",
                context={"directory": str(source_dir)},
            ) from None
        start = time.monotonic()
        successful_docs, failed_docs = await self._load_all_documents(source_dir)
        logger.info("load_phase_complete", doc_count=len(successful_docs))
        chunks = await self.chunker.chunk(successful_docs)
        logger.info("chunk_phase_complete", chunk_count=len(chunks))


    async def _load_all_documents(
        self,
        source_dir: Path,
    ) -> tuple[list[Document], list[str]]:
        """Load all MD and PDF documents from source_dir.

        Returns a tuple of (all_docs, failed_paths) where all_docs are
        successfully loaded Document objects and failed_paths are string
        paths of files that could not be loaded.

        Args:
            source_dir: Root directory to scan.

        Returns:
            Tuple of (list[Document], list[str failed file paths]).

        # TODO:
        #   1. Call `await self.markdown_loader.load_directory(source_dir)`.
        #   2. Iterate `source_dir.rglob("*.pdf")` and call `await self.pdf_loader.load(path)`
        #      for each, collecting failures in `failed_paths`.
        #   3. Return (md_docs + pdf_docs, failed_paths).
        """
        failed_paths = []
        docs = []
        try:
            docs.extend(await self.markdown_loader.load_directory(source_dir))
            docs.extend(await self.text_loader.load_directory(source_dir))
        except LoaderError as e:
            failed_paths.append(str(source_dir))
            logger.error("loader_error", file_path=str(source_dir), error=str(e))
        for pdf_path in source_dir.rglob("*.pdf"):
            try:
                docs.extend(await self.pdf_loader.load(pdf_path))
            except LoaderError as e:
                failed_paths.append(str(pdf_path))
                logger.error("loader_error", file_path=str(pdf_path), error=str(e))
        return docs, failed_paths