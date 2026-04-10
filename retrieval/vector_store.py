# FILE: customer_support_rag/retrieval/vector_store.py
"""ChromaDB vector store connection and node indexing.

This module is the **only** place that talks directly to ChromaDB.
All other retrieval components call VectorStoreManager rather than
importing chromadb directly, keeping the storage backend swappable.

Classes:
    VectorStoreManager: Manages ChromaDB connection, LlamaIndex
        VectorStoreIndex lifecycle, and node upsert operations.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)
import logging
from config.settings import settings
from core.exceptions import RAGBaseException
from core.logging import get_logger

logger = get_logger(__name__)
_stdlib_logger = logging.getLogger(__name__)


class VectorStoreError(RAGBaseException):
    """Raised when a vector store operation fails."""


class VectorStoreManager:
    """Manages the ChromaDB-backed LlamaIndex VectorStoreIndex.

    Responsibilities:
        - Open and maintain the chromadb.HttpClient connection.
        - Wrap the ChromaDB collection in a LlamaIndex ChromaVectorStore
          so that LlamaIndex can drive embedding generation and storage.
        - Expose ``add_nodes`` for the ingestion pipeline and
          ``get_index`` for the retriever.

    Example::

        manager = VectorStoreManager()
        await manager.connect()
        await manager.add_nodes(text_nodes)
        index = await manager.get_index()
    """

    def __init__(self) -> None:
        """Declare instance attributes; call ``connect()`` before use."""
        self._client: chromadb.HttpClient | None = None
        self._collection = None
        self._vector_store: ChromaVectorStore | None = None
        self._storage_context: StorageContext | None = None
        self._index: VectorStoreIndex | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialise ChromaDB client, collection, and LlamaIndex wrappers.

        Must be called once before any other method. Wraps the synchronous
        chromadb client calls in ``asyncio.to_thread`` to keep the event
        loop unblocked.

        Retries up to 5 times with exponential back-off (2 s → 30 s). This
        handles Docker Compose startup races where the ChromaDB container is
        healthy but not yet accepting connections when the API container starts.

        Raises:
            VectorStoreError: If the ChromaDB server is unreachable after all
                retries or the collection cannot be created.
        """
        try:
            await self._connect_with_retry()
        except RetryError as e:
            raise VectorStoreError(
                f"ChromaDB unreachable after retries at "
                f"{settings.chroma_host}:{settings.chroma_port}",
                context={
                    "host": settings.chroma_host,
                    "port": settings.chroma_port,
                    "collection": settings.chroma_collection_name,
                    "original_error": str(e),
                },
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f"Failed to connect to ChromaDB at {settings.chroma_host}:{settings.chroma_port}",
                context={
                    "host": settings.chroma_host,
                    "port": settings.chroma_port,
                    "collection": settings.chroma_collection_name,
                    "original_error": str(e),
                },
            ) from e

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(_stdlib_logger, logging.WARNING),
        reraise=True,
    )
    async def _connect_with_retry(self) -> None:
        """Inner connect logic with tenacity retry — called by connect()."""
        self._client = await asyncio.to_thread(
            chromadb.HttpClient,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

        self._collection = await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=settings.chroma_collection_name,
        )

        self._vector_store = ChromaVectorStore(
            chroma_collection=self._collection,
        )

        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store,
        )

        logger.info(
            "chroma_connected",
            host=settings.chroma_host,
            port=settings.chroma_port,
            collection=settings.chroma_collection_name,
        )

    async def add_nodes(self, nodes: Sequence[TextNode]) -> None:
        """Embed and upsert a list of LlamaIndex TextNodes into ChromaDB.

        Uses the OpenAI embedding model configured in ``settings`` to embed
        the nodes, then stores them via the LlamaIndex VectorStoreIndex.
        Calling this does **not** rebuild the retriever — call
        ``get_index()`` after ingestion to get a fresh index.

        Args:
            nodes: Sequence of ``llama_index.core.schema.TextNode`` objects
                produced by the ingestion chunker.

        Raises:
            VectorStoreError: If the connection has not been established
                (``connect()`` not called) or if the upsert fails.
        """
        try:
            if self._storage_context is None:
                raise VectorStoreError(
                    "connect() must be called before add_nodes()",
                    context={
                        "node_count": len(nodes),
                    },
                )
            embed_model = OpenAIEmbedding(
                model=settings.embed_model_name,
                api_key=settings.openai_api_key,
                dimensions=settings.embed_dimensions,
            )
            self._index = await asyncio.to_thread(
                VectorStoreIndex,
                nodes=nodes,
                storage_context=self._storage_context,
                embed_model=embed_model,
                show_progress=True,
            )
            logger.info("nodes_upserted", count=len(nodes))
        except Exception as e:
            raise VectorStoreError(
                "Failed to upsert nodes into ChromaDB",
                context={
                    "node_count": len(nodes),
                    "original_error": str(e),
                },
            ) from e

    async def get_index(self) -> VectorStoreIndex:
        """Return a VectorStoreIndex loaded from the existing ChromaDB collection.

        Used by HybridRetriever to build its dense retriever against the
        already-populated vector store, without re-embedding any nodes.

        Returns:
            A ``VectorStoreIndex`` backed by the live ChromaDB collection.

        Raises:
            VectorStoreError: If ``connect()`` has not been called first.

        """
        try:
            if self._vector_store is None:
                raise VectorStoreError(
                    "connect() must be called before get_index()",
                    context={
                        "vector_store": self._vector_store,
                    },
                )
            if self._index is not None:
                return self._index
            embed_model = OpenAIEmbedding(
                model=settings.embed_model_name,
                api_key=settings.openai_api_key,
                dimensions=settings.embed_dimensions,
            )
            self._index = await asyncio.to_thread(
                VectorStoreIndex.from_vector_store,
                self._vector_store,
                embed_model=embed_model,
            )
            return self._index
        except Exception as e:
            raise VectorStoreError(
                "Failed to load index from ChromaDB",
                context={
                    "vector_store": self._vector_store,
                    "original_error": str(e),
                },
            ) from e

    async def delete_collection(self) -> None:
        """Drop and recreate the ChromaDB collection for a full re-ingest.

        Calling this before a new ``add_nodes`` run prevents duplicate
        chunks from accumulating across ingestion runs.

        Raises:
            VectorStoreError: If the client is not connected.
        """
        try:
            if self._client is None:
                raise VectorStoreError(
                    "connect() must be called before delete_collection()",
                    context={
                        "client": self._client,
                    },
                )
            await asyncio.to_thread(
                self._client.delete_collection,
                name=settings.chroma_collection_name,
            )
            await self.connect()
            self._index = None
            logger.info(
                "collection_deleted",
                collection=settings.chroma_collection_name,
            )
        except Exception as e:
            raise VectorStoreError(
                "Failed to delete collection",
                context={
                    "collection": settings.chroma_collection_name,
                    "original_error": str(e),
                },
            ) from e

    async def get_all_nodes(self) -> list[TextNode]:
        """Fetch all nodes sequentially from the ChromaDB collection."""
        if self._collection is None:
            raise VectorStoreError("connect() must be called before get_all_nodes()")
        
        try:
            data = await asyncio.to_thread(
                self._collection.get,
                include=["metadatas", "documents"]
            )
            nodes = []
            if data and data.get("ids"):
                for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
                    nodes.append(TextNode(id_=id_, text=doc, metadata=meta or {}))
            return nodes
        except Exception as e:
            raise VectorStoreError(
                "Failed to fetch all nodes from ChromaDB",
                context={"original_error": str(e)}
            ) from e
