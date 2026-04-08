import pytest
import asyncio
from pathlib import Path
from llama_index.core import Document
from core.exceptions import LoaderError
from ingestion.loaders import MarkdownLoader, TextLoader, PDFLoader

# We assume data/docs/medical_paper_1.pdf exists in the project root
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "docs" / "medical_paper_1.pdf"


@pytest.mark.asyncio
async def test_markdown_loader_load(tmp_path):
    loader = MarkdownLoader()
    md_file = tmp_path / "test.md"
    md_file.write_text("# Hello World\nThis is a test markdown file.", encoding="utf-8")
    
    docs = await loader.load(md_file)
    assert len(docs) == 1
    assert "Hello World" in docs[0].text
    assert docs[0].metadata["file_type"] == "markdown"
    assert docs[0].metadata["source_path"] == str(md_file)


@pytest.mark.asyncio
async def test_markdown_loader_invalid_file(tmp_path):
    loader = MarkdownLoader()
    with pytest.raises(LoaderError):
        await loader.load(tmp_path / "does_not_exist.md")
        
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("Not a markdown", encoding="utf-8")
    with pytest.raises(LoaderError):
        await loader.load(invalid_file)


@pytest.mark.asyncio
async def test_text_loader_load(tmp_path):
    loader = TextLoader()
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello Text Loader.", encoding="utf-8")
    
    docs = await loader.load(txt_file)
    assert len(docs) == 1
    assert "Hello Text Loader." in docs[0].text
    assert docs[0].metadata["file_type"] == "text"
    assert docs[0].metadata["source_path"] == str(txt_file)


@pytest.mark.asyncio
async def test_markdown_loader_directory(tmp_path):
    loader = MarkdownLoader()
    (tmp_path / "file1.md").write_text("File 1")
    (tmp_path / "file2.md").write_text("File 2")
    
    # Should ignore this
    (tmp_path / "file3.txt").write_text("File 3")
    
    docs = await loader.load_directory(tmp_path)
    assert len(docs) == 2


@pytest.mark.asyncio
async def test_pdf_loader_load():
    loader = PDFLoader()
    if not SAMPLE_PDF.exists():
        pytest.skip("Sample PDF not found, skipping PDFLoader test.")
        
    docs = await loader.load(SAMPLE_PDF)
    assert len(docs) > 0  # Assuming it has at least 1 page
    assert docs[0].metadata["file_type"] == "pdf"
    assert docs[0].metadata["page_number"] == 1
    assert docs[0].metadata["source_path"] == str(SAMPLE_PDF)
