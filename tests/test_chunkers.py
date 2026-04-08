import pytest
from llama_index.core import Document
from ingestion.chunkers import SemanticChunker, _estimate_tokens

@pytest.mark.asyncio
async def test_semantic_chunker():
    chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
    
    # Create dummy documents
    text1 = "This is the first sentence. " * 10
    text2 = "This is the second document's sentence. " * 8
    
    doc1 = Document(text=text1, metadata={"source_path": "test1.md", "file_type": "markdown"})
    doc2 = Document(text=text2, metadata={"source_path": "test2.md", "file_type": "markdown"})
    
    chunks = await chunker.chunk([doc1, doc2])
    
    # Assert semantic chunks were created
    assert len(chunks) > 0
    
    # Check typing and metadata of DocumentChunk
    first_chunk = chunks[0]
    assert hasattr(first_chunk, "chunk_id")
    assert first_chunk.source_path == "test1.md"
    assert first_chunk.metadata.get("file_type") == "markdown"
    assert first_chunk.token_count > 0


@pytest.mark.asyncio
async def test_semantic_chunker_single():
    chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
    text = "This is a simple single document test case. " * 10
    doc = Document(text=text, metadata={"source_path": "single.md"})
    
    chunks = await chunker.chunk_single(doc)
    assert len(chunks) > 0
    assert chunks[0].source_path == "single.md"


def test_estimate_tokens():
    text = "This is a test document with several words."
    token_count = _estimate_tokens(text)
    
    # Using cl100k_base, the estimation should be close to the word count or higher
    assert token_count >= 5
