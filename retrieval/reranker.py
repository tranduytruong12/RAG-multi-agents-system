# FILE: customer_support_rag/retrieval/reranker.py
"""Relevance re-ranking for retrieved document chunks.

After the HybridRetriever produces a fused result list via RRF, the
ContextRanker provides a *second-pass* re-scoring using either:
  - A **Cohere Rerank API** call (cross-encoder quality, no local GPU needed).
  - Future: a local cross-encoder model (e.g. sentence-transformers).

The ranker only re-orders an existing list; it never fetches new chunks.
The output list is identical in structure to the input but sorted by the
ranker's relevance score.

Classes:
    ContextRanker: Second-pass re-ranker, defaults to Cohere Rerank v3.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from config.settings import settings
from core.exceptions import RAGBaseException
from core.logging import get_logger
from core.types import RetrievalResult
import cohere

logger = get_logger(__name__)


class RankerError(RAGBaseException):
    """Raised when a re-ranking operation fails."""


class ContextRanker:
    """Second-pass relevance ranker using Cohere Rerank.

    Takes the output of ``HybridRetriever.retrieve()`` and re-scores every
    chunk against the original query using Cohere's cross-encoder model,
    returning the top-N results.

    Cohere Rerank v3 (``rerank-english-v3.0``) is the default because it
    significantly outperforms BM25 or cosine similarity alone on domain-
    specific customer-support text.

    Usage::

        ranker = ContextRanker()
        reranked = await ranker.rank(query, retrieval_results, top_n=5)
    """

    def __init__(self) -> None:
        """Declare Cohere client; lazy-initialised on first ``rank()`` call."""
        self._client = None   # cohere.AsyncClient — set in _ensure_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """Re-rank a list of retrieval results against the query.

        Args:
            query: The original user query string used for re-scoring.
            results: Retrieved chunks from ``HybridRetriever.retrieve()``.
                The list is not modified in place.
            top_n: Number of results to return after re-ranking.
                Defaults to ``settings.reranker_top_n`` (5).

        Returns:
            A new list of ``RetrievalResult`` objects re-ordered by Cohere
            relevance score (descending), capped at ``top_n`` items.
            If ``results`` is empty or Cohere returns no scores, the
            original list is returned unchanged.

        Raises:
            RankerError: If the Cohere API call fails after retries, or if
                the API key is missing.
        """
        try:
            if not results:
                return []
            
            top_n = top_n or settings.reranker_top_n

            await self._ensure_client()
            
            documents = [r.chunk.content for r in results]
            
            response = await self._client.rerank(
                query=query,
                documents=documents,
                model="rerank-english-v3.0",
                top_n=top_n,
                return_documents=False,
            )
            
            reranked = []
            for r in response.results:
                original = results[r.index]
                reranked.append(
                    RetrievalResult(
                        chunk=original.chunk,
                        score=float(r.relevance_score),
                        retrieval_method="reranked",
                    )
                )
            
            logger.info(
                "rerank_complete",
                query_preview=query[:60],
                input_count=len(results),
                output_count=len(reranked),
            )
            
            return reranked
        
        except cohere.APIError as e:
            raise RankerError(
                f"Cohere API error: {e}",
                context={"query": query[:60]},
            ) from e
        except Exception as e:
            raise RankerError(
                f"Unexpected error during rerank: {e}",
                context={"query": query[:60]},
            ) from e

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _ensure_client(self) -> None:
        """Lazily initialise the Cohere async client if not already done.

        # TODO (Sprint 2):
        #   1. Guard: if self._client is not None, return immediately.
        #
        #   2. Check API key:
        #          if not settings.cohere_api_key:
        #              raise RankerError(
        #                  "COHERE_API_KEY is not set in environment",
        #                  context={"setting": "cohere_api_key"},
        #              )
        #
        #   3. Import and initialise:
        #          import cohere
        #          self._client = cohere.AsyncClient(
        #              api_key=settings.cohere_api_key,
        #          )
        #
        #   4. Log:
        #          logger.info("cohere_client_initialised")
        """
        if self._client is not None:
            return

        if not settings.cohere_api_key:
            raise RankerError(
                "COHERE_API_KEY is not set in environment",
                context={"setting": "cohere_api_key"},
            )

        self._client = cohere.AsyncClient(
            api_key=settings.cohere_api_key,
        )

        logger.info("cohere_client_initialised")
