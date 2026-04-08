import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from retrieval.vector_store import VectorStoreManager, VectorStoreError
from llama_index.core.schema import TextNode

@pytest.fixture
def manager():
    return VectorStoreManager()

@pytest.mark.asyncio
@patch("retrieval.vector_store.chromadb.HttpClient")
@patch("retrieval.vector_store.ChromaVectorStore")
@patch("retrieval.vector_store.StorageContext.from_defaults")
async def test_connect(mock_storage_context, mock_chroma_vs, mock_http_client, manager):
    # Setup mocks
    mock_client_instance = MagicMock()
    mock_http_client.return_value = mock_client_instance
    mock_collection = MagicMock()
    mock_client_instance.get_or_create_collection.return_value = mock_collection
    
    await manager.connect()
    
    # Assertions
    assert manager._client is not None
    assert manager._collection is not None
    assert manager._vector_store is not None
    assert manager._storage_context is not None

@pytest.mark.asyncio
@patch("retrieval.vector_store.OpenAIEmbedding")
@patch("retrieval.vector_store.VectorStoreIndex", autospec=True)
async def test_add_nodes(mock_index, mock_embed, manager):
    # Simulate a connected manager
    manager._storage_context = object()  # dummy
    
    nodes = [TextNode(text="Test node 1", id_="1")]
    await manager.add_nodes(nodes)
    
    # Assertions
    assert manager._index is not None
    mock_index.assert_called_once()
    mock_embed.assert_called_once()

@pytest.mark.asyncio
async def test_add_nodes_error(manager):
    # Should raise error since it's not connected
    with pytest.raises(VectorStoreError):
        await manager.add_nodes([])

@pytest.mark.asyncio
async def test_get_all_nodes(manager):
    # Setup mock collection
    manager._collection = MagicMock()
    manager._collection.get.return_value = {
        "ids": ["1", "2"],
        "documents": ["doc1", "doc2"],
        "metadatas": [{"key": "val1"}, {"key": "val2"}]
    }
    
    nodes = await manager.get_all_nodes()
    
    assert len(nodes) == 2
    assert nodes[0].id_ == "1"
    assert nodes[0].text == "doc1"
    assert nodes[1].metadata["key"] == "val2"

@pytest.mark.asyncio
async def test_get_index(manager):
    manager._vector_store = MagicMock()
    manager._index = MagicMock()
    
    # Should simply return the existing index
    idx = await manager.get_index()
    assert idx is manager._index
