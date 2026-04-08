import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from retrieval.reranker import ContextRanker, RankerError
from core.types import RetrievalResult, DocumentChunk
from config.settings import settings

@pytest.fixture
def ranker():
    # Make sure we have a mock api key, otherwise ensure_client will throw
    settings.cohere_api_key = "dummy_key"
    return ContextRanker()

@pytest.fixture
def sample_results():
    chunk1 = DocumentChunk(chunk_id="1", content="Answer 1", metadata={}, source_path="test", token_count=0)
    chunk2 = DocumentChunk(chunk_id="2", content="Answer 2", metadata={}, source_path="test", token_count=0)
    return [
        RetrievalResult(chunk=chunk1, score=0.5, retrieval_method="hybrid"),
        RetrievalResult(chunk=chunk2, score=0.4, retrieval_method="hybrid"),
    ]

@pytest.mark.asyncio
@patch("retrieval.reranker.cohere.AsyncClient")
async def test_rerank(mock_cohere_client, ranker, sample_results):
    # Setup mock cohere client
    mock_client = AsyncMock()
    
    mock_response = MagicMock()
    mock_r1 = MagicMock()
    mock_r1.index = 1
    mock_r1.relevance_score = 0.99
    
    mock_r2 = MagicMock()
    mock_r2.index = 0
    mock_r2.relevance_score = 0.88
    
    mock_response.results = [mock_r1, mock_r2]
    mock_client.rerank = AsyncMock(return_value=mock_response)
    
    ranker._client = mock_client
    
    reranked = await ranker.rerank("Query", sample_results, top_n=2)
    
    assert len(reranked) == 2
    assert reranked[0].score == 0.99
    assert reranked[0].chunk.content == "Answer 2"  # index 1 from sample_results
    assert reranked[1].score == 0.88
    assert reranked[1].chunk.content == "Answer 1"  # index 0 from sample_results

@pytest.mark.asyncio
async def test_rerank_empty(ranker):
    reranked = await ranker.rerank("Query", [], top_n=2)
    assert reranked == []
