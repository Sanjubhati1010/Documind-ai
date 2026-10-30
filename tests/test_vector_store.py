
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
    search_documents,
)


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for offline unit tests."""

    def _embed(self, text: str) -> list[float]:
        text = text.lower()

        if "database" in text or "normalization" in text:
            return [1.0, 0.0, 0.0]

        if "python" in text or "programming" in text:
            return [0.0, 1.0, 0.0]

        return [0.0, 0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Database normalization reduces data redundancy.",
            metadata={"source": "dbms.pdf", "page": 3},
        ),
        Document(
            page_content="Python is used for programming and automation.",
            metadata={"source": "python.pdf", "page": 5},
        ),
    ]


def test_empty_documents_are_rejected():
    with pytest.raises(ValueError, match="empty documents"):
        create_vector_store([], embeddings=FakeEmbeddings())


def test_create_vector_store(sample_documents):
    store = create_vector_store(
        sample_documents,
        embeddings=FakeEmbeddings(),
    )

    assert store.index.ntotal == 2


def test_search_returns_source_metadata(sample_documents):
    store = create_vector_store(
        sample_documents,
        embeddings=FakeEmbeddings(),
    )

    results = search_documents(
        store,
        query="database normalization",
        k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "dbms.pdf"
    assert results[0].metadata["page"] == 3


def test_empty_query_is_rejected(sample_documents):
    store = create_vector_store(
        sample_documents,
        embeddings=FakeEmbeddings(),
    )

    with pytest.raises(ValueError, match="query cannot be empty"):
        search_documents(store, query="  ")


def test_invalid_k_is_rejected(sample_documents):
    store = create_vector_store(
        sample_documents,
        embeddings=FakeEmbeddings(),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        search_documents(store, query="database", k=0)


def test_save_and_load_vector_store(tmp_path, sample_documents):
    embeddings = FakeEmbeddings()

    store = create_vector_store(
        sample_documents,
        embeddings=embeddings,
    )

    save_vector_store(
        store,
        directory=tmp_path,
        index_name="test_index",
    )

    loaded_store = load_vector_store(
        directory=tmp_path,
        index_name="test_index",
        embeddings=embeddings,
    )

    results = search_documents(
        loaded_store,
        query="database normalization",
        k=1,
    )

    assert loaded_store.index.ntotal == 2
    assert results[0].metadata["source"] == "dbms.pdf"
    assert results[0].metadata["page"] == 3


def test_loading_missing_index_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="FAISS index not found"):
        load_vector_store(
            directory=tmp_path,
            embeddings=FakeEmbeddings(),
        )
