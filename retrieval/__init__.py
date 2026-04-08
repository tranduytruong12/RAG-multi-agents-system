# FILE: customer_support_rag/retrieval/__init__.py
"""Retrieval & Vector Search package.

Public surface (import from here, not from sub-modules):

    from retrieval import VectorStoreManager, HybridRetriever, ContextRanker
"""

from retrieval.vector_store import VectorStoreManager, VectorStoreError
from retrieval.retriever import HybridRetriever, RetrieverError
from retrieval.reranker import ContextRanker, RankerError

__all__ = [
    "VectorStoreManager",
    "VectorStoreError",
    "HybridRetriever",
    "RetrieverError",
    "ContextRanker",
    "RankerError",
]
