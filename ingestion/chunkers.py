# FILE: customer_support_rag/ingestion/chunkers.py
"""Document chunking strategies for the ingestion pipeline.

Chunkers take raw LlamaIndex `Document` objects (from loaders) and split
them into smaller, overlapping `DocumentChunk` units suitable for embedding
and vector storage.

The `SemanticChunker` uses LlamaIndex's `SentenceSplitter` to produce
token-aware chunks with configurable size and overlap, driven by
`config.settings` values.
"""

from __future__ import annotations

import uuid
import asyncio
import tiktoken
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from config.settings import settings
from core.exceptions import ChunkerError
from core.logging import get_logger
from core.types import DocumentChunk

logger = get_logger(__name__)


class SemanticChunker:
    """Splits LlamaIndex Documents into overlapping DocumentChunk units.

    Uses `SentenceSplitter` (sentence-aware boundary detection) to avoid
    splitting mid-sentence. Chunk size and overlap are configurable via
    `settings.chunk_size` and `settings.chunk_overlap`.

    Args:
        chunk_size: Target token count per chunk. Defaults to settings.chunk_size.
        chunk_overlap: Token overlap between consecutive chunks for context
            continuity. Defaults to settings.chunk_overlap.

    Example:
        chunker = SemanticChunker()
        chunks = await chunker.chunk(documents)
        # chunks: list[DocumentChunk]
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size: int = chunk_size or settings.chunk_size
        self.chunk_overlap: int = chunk_overlap or settings.chunk_overlap

    async def chunk(self, documents: list[Document]) -> list[DocumentChunk]:
        """Split a list of Documents into overlapping DocumentChunk objects.

        Args:
            documents: Raw LlamaIndex Document objects from a loader.
                Should have `metadata["source_path"]` and `metadata["file_type"]`
                set by the loader.

        Returns:
            Ordered list of DocumentChunk dataclasses ready for embedding.
            Preserves the relative order of documents and chunks within
            each document.

        Raises:
            ChunkerError: If SentenceSplitter or TextNode extraction fails.
        """
        try:
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            nodes = await asyncio.to_thread(
                splitter.get_nodes_from_documents,
                documents,
            )
            chunks = []
            for node in nodes:
                chunks.append(DocumentChunk(
                    chunk_id=node.node_id,
                    source_path=node.metadata.get("source_path", "unknown"),
                    content=node.text,
                    metadata=dict(node.metadata),
                    token_count=_estimate_tokens(node.text),
                ))
            logger.debug("chunking_complete",
                doc_count=len(documents), chunk_count=len(chunks))
            return chunks
        except Exception as exc:
            raise ChunkerError(
                "Chunking failed",
                context={"doc_count": len(documents), "original_error": str(exc)},
            ) from exc

    async def chunk_single(self, document: Document) -> list[DocumentChunk]:
        """Chunk a single document. Convenience wrapper around `chunk`.

        Args:
            document: A single LlamaIndex Document to chunk.

        Returns:
            List of DocumentChunk objects for this document.

        Raises:
            ChunkerError: Propagated from `chunk`.
        """
        return await self.chunk([document])


# ── Private Helpers ────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Uses whitespace splitting as a fast approximation. For more accurate
    counts, replace with tiktoken:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))

    Args:
        text: The text to estimate tokens for.

    Returns:
        Approximate token count.

    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))
