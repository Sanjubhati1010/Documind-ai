
import fitz
import pytest
from langchain_core.documents import Document
from src.pdf_processor import (
    extract_pdf_pages,
    process_pdf,
    split_documents,
)


def make_pdf(*page_texts: str) -> bytes:
    """Generate an in-memory PDF with one short text per page."""
    pdf = fitz.open()

    for text in page_texts:
        page = pdf.new_page()
        page.insert_text((72, 72), text, fontsize=11)

    content = pdf.tobytes()
    pdf.close()
    return content


def test_extract_pdf_pages_preserves_page_numbers():
    content = make_pdf(
        "Database introduction.",
        "Normalization concepts.",
    )

    docs = extract_pdf_pages(content, "dbms.pdf")

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "dbms.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
    assert "Database introduction" in docs[0].page_content
    assert "Normalization concepts" in docs[1].page_content


def test_invalid_pdf_is_rejected():
    with pytest.raises(ValueError, match="Invalid PDF"):
        extract_pdf_pages(b"not a PDF")


def test_empty_pdf_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        extract_pdf_pages(b"")


def test_textless_pdf_is_rejected():
    content = make_pdf("   ")

    with pytest.raises(ValueError, match="No extractable text"):
        extract_pdf_pages(content)


def test_chunks_preserve_metadata():
    # Test chunking directly without squeezing long text into one PDF page.
    long_text = " ".join(
        f"Database normalization concept number {i} reduces redundancy."
        for i in range(100)
    )

    docs = [
        Document(
            page_content=long_text,
            metadata={"source": "notes.pdf", "page": 1},
        )
    ]

    chunks = split_documents(
        docs,
        chunk_size=150,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "notes.pdf" for c in chunks)
    assert all(c.metadata["page"] == 1 for c in chunks)
    assert all("start_index" in c.metadata for c in chunks)
    assert all(len(c.page_content) <= 150 for c in chunks)


def test_process_pdf_returns_chunks():
    content = make_pdf(
        "Artificial intelligence helps process information."
    )

    chunks = process_pdf(content, "ai.pdf")

    assert chunks
    assert "artificial intelligence" in chunks[0].page_content.lower()
    assert chunks[0].metadata["source"] == "ai.pdf"
    assert chunks[0].metadata["page"] == 1


def test_invalid_chunk_settings_are_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        split_documents([], chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents(
            [],
            chunk_size=100,
            chunk_overlap=100,
        )
