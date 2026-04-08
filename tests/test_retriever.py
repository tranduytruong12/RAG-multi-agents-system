import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from retrieval.retriever import HybridRetriever, RetrieverError

@pytest.fixture
def hybrid_retriever():
    return HybridRetriever()

@pytest.mark.asyncio
@patch("retrieval.retriever.VectorIndexRetriever")
@patch("retrieval.retriever.BM25Retriever.from_defaults")
@patch("retrieval.retriever.QueryFusionRetriever")
async def test_build_index(mock_fusion, mock_bm25_from_defaults, mock_dense, hybrid_retriever):
    # Mock VectorStoreManager
    hybrid_retriever._vector_store_manager = MagicMock()
    hybrid_retriever._vector_store_manager.connect = AsyncMock()
    
    mock_index = MagicMock()
    mock_index.docstore = MagicMock()
    mock_index.docstore.docs = {"id": "doc_here"}
    hybrid_retriever._vector_store_manager.get_index = AsyncMock(return_value=mock_index)
    
    await hybrid_retriever.build_index(top_k=2)
    
    hybrid_retriever._vector_store_manager.connect.assert_awaited_once()
    hybrid_retriever._vector_store_manager.get_index.assert_awaited_once()
    assert hybrid_retriever._fusion_retriever is not None
    mock_bm25_from_defaults.assert_called_once()
    mock_fusion.assert_called_once()

@pytest.mark.asyncio
async def test_retrieve_without_build():
    hybrid_retriever = HybridRetriever()
    with pytest.raises(RetrieverError):
        await hybrid_retriever.retrieve("query")

@pytest.mark.asyncio
async def test_retrieve(hybrid_retriever):
    mock_fusion = MagicMock()
    
    # mock nodes_with_scores
    mock_node_score = MagicMock()
    mock_node_score.node.get_content.return_value = "This is a test answer."
    mock_node_score.node.node_id = "test_id"
    mock_node_score.node.metadata = {"source": "test"}
    mock_node_score.score = 0.95
    
    mock_fusion.aretrieve = AsyncMock(return_value=[mock_node_score])
    hybrid_retriever._fusion_retriever = mock_fusion
    
    results = await hybrid_retriever.retrieve("How are you?", top_k=1)
    
    assert len(results) == 1
    assert results[0].chunk.content == "This is a test answer."
    assert results[0].score == 0.95
    assert results[0].chunk.chunk_id == "test_id"
