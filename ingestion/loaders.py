# FILE: customer_support_rag/ingestion/loaders.py
"""Document loaders for Markdown, Text and PDF source files.

Loaders are responsible for reading raw files from disk and converting them
into LlamaIndex `Document` objects. Each loader enriches `doc.metadata` with
standardised keys so that downstream chunkers and the vector store have
consistent metadata to work with.

All loader methods are async to allow concurrent loading of many files without
blocking the event loop. The actual I/O (SimpleDirectoryReader / PDFReader) is
synchronous under the hood; wrap with `asyncio.to_thread` in the TODO body.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

from core.exceptions import LoaderError
from core.logging import get_logger

logger = get_logger(__name__)


# ── Loader Protocol ────────────────────────────────────────────────────────────

@runtime_checkable
class BaseLoader(Protocol):
    """Structural interface that all document loaders must satisfy.

    Any class implementing `async load(path: Path) -> list[Document]`
    is automatically a valid BaseLoader without explicit inheritance.
    """

    async def load(self, path: Path) -> list[Document]:
        """Load documents from a file path.

        Args:
            path: Absolute or relative path to a file or directory.

        Returns:
            List of LlamaIndex Document objects.
        """
        ...


# ── Markdown Loader ────────────────────────────────────────────────────────────

class MarkdownLoader:
    """Loads Markdown (.md) files into LlamaIndex Document objects.

    Uses LlamaIndex's `SimpleDirectoryReader` with `required_exts=[".md"]`
    to handle both single files and recursive directory scans.

    Example:
        loader = MarkdownLoader()
        docs = await loader.load(Path("./data/docs/refund_policy.md"))
        all_docs = await loader.load_directory(Path("./data/docs/"))
    """

    async def load(self, path: Path) -> list[Document]:
        """Load a single Markdown file and return a list of Documents.

        Args:
            path: Absolute path to a .md file.

        Returns:
            List of LlamaIndex Document objects with enriched metadata.

        Raises:
            LoaderError: If the file does not exist, cannot be read, or
                SimpleDirectoryReader raises an exception.
        """
        if not path.exists() or path.suffix != ".md":
            raise LoaderError(
                f"Invalid file: {path}",
                context={"file_path": str(path)}
            )
        try:
            reader = SimpleDirectoryReader(input_files=[path])
            docs = await asyncio.to_thread(reader.load_data)
            for doc in docs:
                doc.metadata["source_path"] = str(path)
                doc.metadata["file_type"] = "markdown"
                doc.metadata["loaded_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("markdown_loaded", path=str(path), doc_count=len(docs))
            return docs
        except Exception as exc:
            raise LoaderError(
                f"Failed to load markdown: {path}",
                context={"file_path": str(path), "original_error": str(exc)},
            ) from exc
        return []

    async def load_directory(self, directory: Path) -> list[Document]:
        """Load all Markdown files from a directory tree recursively.

        Uses `rglob("*.md")` to discover every Markdown file under `directory`.
        Files that fail individually are skipped and logged, so a single
        corrupt file does not abort the entire directory load.

        Args:
            directory: Root directory to scan for .md files recursively.

        Returns:
            Flat list of all Document objects from all discovered files.

        Raises:
            LoaderError: Only if the directory itself does not exist.
        """
        if not directory.exists() or not directory.is_dir():
            raise LoaderError(
                f"Invalid directory: {directory}",
                context={"directory": str(directory)},
            ) from None
        
        md_files = list(directory.rglob("*.md"))
        all_docs = []
        for file_path in md_files:
            try:
                docs = await self.load(file_path)
                all_docs.extend(docs)
            except LoaderError as e:
                logger.warning("markdown_file_skipped", path=str(file_path), error=str(e))
        logger.info("directory_loaded", directory=str(directory), file_count=len(md_files), doc_count=len(all_docs))
        return all_docs
        


# ── PDF Loader ─────────────────────────────────────────────────────────────────

class PDFLoader:
    """Loads PDF files into LlamaIndex Document objects (one Document per page).

    Uses `llama_index.readers.file.PDFReader` from the
    `llama-index-readers-file` package for page-level extraction.

    Example:
        loader = PDFLoader()
        docs = await loader.load(Path("./data/docs/return_policy.pdf"))
    """

    async def load(self, path: Path) -> list[Document]:
        """Load a PDF file and return one LlamaIndex Document per page.

        Args:
            path: Absolute path to a .pdf file.

        Returns:
            List of LlamaIndex Document objects, one per PDF page,
            with page-level metadata.

        Raises:
            LoaderError: If the file does not exist, is not a PDF, is
                corrupted, or PDFReader raises an exception.
        """
        if not path.exists() or path.suffix.lower() != ".pdf":
            raise LoaderError(
                f"Invalid file: {path}",
                context={"file_path": str(path)},
            ) from None
        
        try:
            reader = PDFReader()
            docs = await asyncio.to_thread(reader.load_data, file=path)
            for i, doc in enumerate(docs):
                doc.metadata["source_path"] = str(path)
                doc.metadata["file_type"] = "pdf"
                doc.metadata["page_number"] = i + 1
                doc.metadata["loaded_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("pdf_loaded", path=str(path), page_count=len(docs))
            return docs
        except Exception as exc:
            raise LoaderError(
                f"Failed to load PDF: {path}",
                context={
                    "file_path": str(path),
                    "file_type": "pdf",
                    "original_error": str(exc),
                },
            ) from exc
        return []

class TextLoader:
    """Loads plain Text (.txt) files into LlamaIndex Document objects."""

    async def load(self, path: Path) -> list[Document]:
        """Load a single Text file and return a list of Documents.

        Args:
            path: Absolute path to a .txt file.

        Returns:
            List of LlamaIndex Document objects with enriched metadata.

        Raises:
            LoaderError: If the file does not exist, cannot be read, or
                SimpleDirectoryReader raises an exception.
        """
        if not path.exists() or path.suffix != ".txt":
            raise LoaderError(
                f"Invalid file: {path}",
                context={"file_path": str(path)}
            )
        try:
            reader = SimpleDirectoryReader(input_files=[path])
            docs = await asyncio.to_thread(reader.load_data)
            for doc in docs:
                doc.metadata["source_path"] = str(path)
                doc.metadata["file_type"] = "text"
                doc.metadata["loaded_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("text_loaded", path=str(path), doc_count=len(docs))
            return docs
        except Exception as exc:
            raise LoaderError(
                f"Failed to load text: {path}",
                context={"file_path": str(path), "original_error": str(exc)},
            ) from exc
        return []

    async def load_directory(self, directory: Path) -> list[Document]:
        """Load all Text files from a directory tree recursively.

        Uses `rglob("*.txt")` to discover every Text file under `directory`.
        Files that fail individually are skipped and logged, so a single
        corrupt file does not abort the entire directory load.

        Args:
            directory: Root directory to scan for .txt files recursively.

        Returns:
            Flat list of all Document objects from all discovered files.

        Raises:
            LoaderError: Only if the directory itself does not exist.
        """
        if not directory.exists() or not directory.is_dir():
            raise LoaderError(
                f"Invalid directory: {directory}",
                context={"directory": str(directory)},
            ) from None
        
        md_files = list(directory.rglob("*.txt"))
        all_docs = []
        for file_path in md_files:
            try:
                docs = await self.load(file_path)
                all_docs.extend(docs)
            except LoaderError as e:
                logger.warning("text_file_skipped", path=str(file_path), error=str(e))
        logger.info("directory_loaded", directory=str(directory), file_count=len(md_files), doc_count=len(all_docs))
        return all_docs