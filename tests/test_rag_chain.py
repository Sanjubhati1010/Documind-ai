
import pytest
from langchain_core.documents import Document
from src.rag_chain import (
    answer_question,
    format_context,
    format_sources,
)


class FakeVectorStore:
    """A deterministic vector store for unit tests."""

    def __init__(self, documents):
        self.documents = documents

    def similarity_search(self, query: str, k: int = 4):
        return self.documents[:k]


class FakeLLM:
    """A fake LLM that records the prompt."""

    def __init__(self, answer="Normalization reduces redundancy."):
        self.answer = answer
        self.last_prompt = ""

    def invoke(self, prompt):
        self.last_prompt = prompt

        class Response:
            def __init__(self, content):
                self.content = content

        return Response(self.answer)


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Normalization reduces data redundancy.",
            metadata={"source": "dbms.pdf", "page": 3},
        ),
        Document(
            page_content="Normalization improves database consistency.",
            metadata={"source": "dbms.pdf", "page": 3},
        ),
    ]


def test_format_context_includes_source_and_page(sample_documents):
    context = format_context(sample_documents)

    assert "dbms.pdf" in context
    assert "Page 3" in context
    assert "Normalization reduces data redundancy." in context


def test_format_sources_removes_duplicate_pages(sample_documents):
    sources = format_sources(sample_documents)

    assert len(sources) == 1
    assert sources[0]["source"] == "dbms.pdf"
    assert sources[0]["page"] == 3
    assert sources[0]["excerpt"]


def test_answer_question_returns_answer_and_sources(sample_documents):
    store = FakeVectorStore(sample_documents)
    llm = FakeLLM()

    result = answer_question(
        store,
        "What is normalization?",
        llm=llm,
    )

    assert result["answer"] == "Normalization reduces redundancy."
    assert result["sources"][0]["source"] == "dbms.pdf"
    assert result["sources"][0]["page"] == 3
    assert "What is normalization?" in llm.last_prompt
    assert "Normalization reduces data redundancy." in llm.last_prompt


def test_empty_question_is_rejected(sample_documents):
    store = FakeVectorStore(sample_documents)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question(store, "   ", llm=FakeLLM())


def test_no_retrieved_documents_returns_fallback():
    store = FakeVectorStore([])
    llm = FakeLLM()

    result = answer_question(
        store,
        "What is normalization?",
        llm=llm,
    )

    assert "could not find relevant information" in result["answer"]
    assert result["sources"] == []
