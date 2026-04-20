# FILE: customer_support_rag/retrieval/retriever.py
"""Hybrid dense + sparse retrieval using LlamaIndex.

Implements a two-path retrieval strategy:
  - **Dense path**: ``VectorIndexRetriever`` using cosine similarity over
    OpenAI embeddings stored in ChromaDB.
  - **Sparse path**: ``BM25Retriever`` for exact keyword matching.
  - **Fusion**: ``QueryFusionRetriever`` with ``mode="reciprocal_rerank"``
    (Reciprocal Rank Fusion, RRF) to merge both ranked lists.

The public interface deliberately converts all LlamaIndex ``NodeWithScore``
objects into ``core.types.RetrievalResult`` before returning, so no
LlamaIndex types leak into the agent layer.

Classes:
    HybridRetriever: The single retrieval entry point for the agent DAG.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from config.settings import settings
from core.exceptions import RAGBaseException
from core.logging import get_logger
from core.types import DocumentChunk, RetrievalResult
from retrieval.vector_store import VectorStoreManager

logger = get_logger(__name__)


class RetrieverError(RAGBaseException):
    """Raised when a retrieval operation fails."""


class HybridRetriever:
    """Hybrid dense+sparse retriever backed by ChromaDB and BM25.

    Usage::

        retriever = HybridRetriever()
        await retriever.build_index()
        results = await retriever.retrieve("How do I get a refund?", top_k=5)
        # results: list[RetrievalResult], sorted by fused score descending
    """

    def __init__(self) -> None:
        """Declare instance attributes; call ``build_index()`` before ``retrieve()``."""
        self._vector_store_manager: VectorStoreManager = VectorStoreManager()
        self._index: Optional[VectorStoreIndex] = None
        self._fusion_retriever: Optional[QueryFusionRetriever] = None

    def invalidate(self) -> None:
        """Invalidate ALL cached state and replace the VectorStoreManager.

        Creates a brand-new VectorStoreManager so that every internal cache
        (ChromaDB client, collection reference, vector store wrapper, and index)
        is discarded. The next ``retrieve()`` call will rebuild everything from
        scratch against the current collection in ChromaDB.

        This must be called after any operation that deletes and recreates the
        ChromaDB collection (i.e. reset ingestion), because ChromaDB assigns a
        new UUID to the recreated collection, making all previous references stale.
        """
        self._vector_store_manager = VectorStoreManager()   # fresh — no stale refs
        self._index = None
        self._fusion_retriever = None
        logger.info(
            "hybrid_retriever.invalidated",
            reason="collection reset — VectorStoreManager replaced, will rebuild on next retrieve call",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_index(self, top_k: int = 5) -> None:
        """Connect to ChromaDB and build the fusion retriever.

        Must be called once (e.g. at application startup or inside the
        LangGraph ``retrieve`` node) before ``retrieve()`` can be used.

        Raises:
            RetrieverError: If the vector store connection or index
                construction fails.
        """
        try:
            # 1. Connect to ChromaDB
            await self._vector_store_manager.connect()

            # 2. Load the VectorStoreIndex from the existing collection
            self._index = await self._vector_store_manager.get_index()
            
            if self._index is None:
                raise RetrieverError(
                    "Failed to load VectorStoreIndex. Index is None.",
                    context={"vector_store_manager": str(self._vector_store_manager)}
                )

            # 3. Build the DENSE retriever
            dense_retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k * 2,   # oversample before fusion
            )

            # 4. Build the SPARSE retriever (BM25)
            # Ensure docstore exists and contains documents
            nodes = []
            if hasattr(self._index, 'docstore') and self._index.docstore.docs:
                nodes = list(self._index.docstore.docs.values())
            else:
                nodes = await self._vector_store_manager.get_all_nodes()
                logger.info(f"Total nodes in docstore: {len(nodes)}")
                
            if not nodes:
                raise RetrieverError("No documents found in docstore or ChromaDB. Cannot build BM25 retriever.")
                
            sparse_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=top_k * 2,
            )

            # 5. Combine with QueryFusionRetriever (RRF)
            self._fusion_retriever = QueryFusionRetriever(
                retrievers=[dense_retriever, sparse_retriever],
                similarity_top_k=top_k,
                num_queries=1,          # no query expansion for now
                mode="reciprocal_rerank",
                use_async=True,
                verbose=False,
            )

            # 6. Log success
            collection_name = getattr(settings, "chroma_collection_name", "unknown_collection")
            logger.info("hybrid_index_built", collection=collection_name)

        except Exception as e:
            # 7. Wrap in try/except, re-raise as RetrieverError with context
            logger.error(f"Failed to build hybrid index: {e}")
            raise RetrieverError(
                "Error occurred while connecting to ChromaDB or building retrievers.",
                context={"error_details": str(e)}
            ) from e


    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Execute hybrid retrieval for a natural-language query.

        Runs the QueryFusionRetriever (dense + sparse, RRF) and converts
        the ranked ``NodeWithScore`` objects into ``RetrievalResult``
        instances using ``core.types`` — no LlamaIndex types are returned.

        Args:
            query: The natural-language search query.
            top_k: Maximum number of results to return after fusion.
                Defaults to 5 (overridden by ``settings.reranker_top_n``
                if passed from the orchestrator).

        Returns:
            A list of ``RetrievalResult`` objects, sorted by descending
            relevance score. Empty list if no relevant chunks are found.

        Raises:
            RetrieverError: If ``build_index()`` has not been called, or
                if the underlying retrieval fails.
        """
        # 1. Guard clause
        if self._fusion_retriever is None:
            raise RetrieverError("build_index() must be called before retrieve()")

        try:
            # 2. Run the fusion retriever asynchronously
            # Using aretrieve since use_async=True was set during initialization
            nodes_with_scores = await self._fusion_retriever.aretrieve(query)
            
            results: list[RetrievalResult] = []

            # 3. Convert NodeWithScore → RetrievalResult
            for node_score in nodes_with_scores:
                node = node_score.node
                
                # a) Extract content
                content = node.get_content()
                
                # b) Extract metadata
                metadata = node.metadata or {}
                
                # c) Build DocumentChunk
                chunk = DocumentChunk(
                    chunk_id=node.node_id,
                    source_path=metadata.get("source_path", ""),
                    content=content,
                    metadata=metadata,
                    token_count=metadata.get("token_count", 0),
                )
                
                # d) Build RetrievalResult
                result = RetrievalResult(
                    chunk=chunk,
                    score=float(node_score.score or 0.0),
                    retrieval_method="hybrid",
                )
                results.append(result)

            # 4. Sort by score descending (Explicit guarantee)
            results.sort(key=lambda r: r.score, reverse=True)

            # 5. Slice to top_k
            top_results = results[:top_k]

            # 6. Log completion
            logger.info(
                "retrieval_complete",
                query_preview=query[:60],
                result_count=len(top_results),
                retrieved_contexts=[
                    {
                        "score": round(r.score, 4),
                        "source": r.chunk.source_path,
                        "content_preview": (r.chunk.content[:300] + "...") if len(r.chunk.content) > 300 else r.chunk.content
                    }
                    for r in top_results
                ]
            )

            # 7. Return results
            return top_results

        except Exception as e:
            logger.error(f"Retrieval execution failed: {e}")
            raise RetrieverError(
                "Error occurred during hybrid retrieve operation.",
                context={"query": query, "error_details": str(e)}
            ) from e