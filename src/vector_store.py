
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL, VECTORSTORE_DIR


def get_embeddings(
    model_name: str = EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Load the Hugging Face embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(
    documents: list[Document],
    embeddings=None,
) -> FAISS:
    """Create a FAISS index from document chunks."""
    if not documents:
        raise ValueError("Cannot create a vector store from empty documents.")

    if embeddings is None:
        embeddings = get_embeddings()

    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )


def save_vector_store(
    vector_store: FAISS,
    directory: str | Path = VECTORSTORE_DIR,
    index_name: str = "documind_index",
) -> Path:
    """Save the FAISS index locally."""
    if not index_name or Path(index_name).name != index_name:
        raise ValueError("index_name must be a simple filename.")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(
        folder_path=str(directory),
        index_name=index_name,
    )

    return directory


def load_vector_store(
    directory: str | Path = VECTORSTORE_DIR,
    index_name: str = "documind_index",
    embeddings=None,
) -> FAISS:
    """Load a previously saved local FAISS index."""
    if not index_name or Path(index_name).name != index_name:
        raise ValueError("index_name must be a simple filename.")

    directory = Path(directory)

    if not (directory / f"{index_name}.faiss").is_file():
        raise FileNotFoundError(
            f"FAISS index not found in {directory}. "
            "Process a PDF and create the index first."
        )

    if embeddings is None:
        embeddings = get_embeddings()

    # Only load index files created by and trusted by this application.
    return FAISS.load_local(
        folder_path=str(directory),
        embeddings=embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )


def search_documents(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> list[Document]:
    """Retrieve the most relevant chunks for a question."""
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    return vector_store.similarity_search(
        query=query.strip(),
        k=k,
    )
