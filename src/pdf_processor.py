
import re
from io import BytesIO

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_page_text(text: str) -> str:
    """Clean extracted PDF text while preserving paragraphs and bullet lists."""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unwanted spaces around line breaks.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Normalize excessive spaces.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Convert flattened PDF bullet symbols into Markdown list items.
    text = re.sub(
        r"(?<!\n)[ \t]*[•▪◦‣][ \t]*",
        "\n- ",
        text,
    )

    # Normalize numbered list spacing.
    text = re.sub(
        r"(?<!\n)[ \t]+(\d+[.)][ \t]+)",
        r"\n\1",
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_pdf_pages(
    pdf_bytes: bytes,
    source_name: str = "uploaded.pdf",
) -> list[Document]:
    """
    Extract text from every PDF page.

    Preserves original filename and 1-based page number.
    Rejects empty, invalid, and textless PDF files.
    """

    # 1. Reject empty input.
    if not pdf_bytes:
        raise ValueError(
            "The uploaded PDF is empty: no data found."
        )

    # 2. Validate PDF header.
    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError(
            "Invalid PDF: uploaded file is not a valid PDF."
        )

    # 3. Open PDF safely.
    try:
        pdf = fitz.open(
            stream=BytesIO(pdf_bytes),
            filetype="pdf",
        )

    except Exception as exc:
        raise ValueError(
            f"Invalid PDF: could not open the uploaded file. {exc}"
        ) from exc

    documents = []

    try:
        # 4. Reject PDFs containing zero pages.
        if pdf.page_count == 0:
            raise ValueError(
                "The uploaded PDF is empty: it contains no pages."
            )

        # 5. Extract text page by page.
        for page_number, page in enumerate(pdf, start=1):

            raw_text = page.get_text(
                "text",
                sort=True,
            )

            cleaned_text = clean_page_text(raw_text)

            if not cleaned_text:
                continue

            documents.append(
                Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": source_name,
                        "page": page_number,
                    },
                )
            )

    finally:
        pdf.close()

    # 6. Reject scanned or image-only PDFs without extractable text.
    if not documents:
        raise ValueError(
            "No extractable text was found in this PDF. "
            "It may be scanned or image-only. OCR is not included yet."
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split documents into overlapping chunks.

    Preserves original source and page metadata.
    Adds character offsets for source excerpt reconstruction.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size."
        )

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents(documents)


def process_pdf(
    pdf_bytes: bytes,
    source_name: str = "uploaded.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Complete PDF processing pipeline:

    1. Validate PDF.
    2. Extract page-wise text.
    3. Clean extracted text.
    4. Split text into overlapping chunks.
    5. Preserve source and page metadata.
    """

    pages = extract_pdf_pages(
        pdf_bytes=pdf_bytes,
        source_name=source_name,
    )

    chunks = split_documents(
        documents=pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            "No extractable text chunks were generated from this PDF."
        )

    return chunks
