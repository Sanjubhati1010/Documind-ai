
import streamlit as st
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, RETRIEVAL_K
from src.pdf_processor import process_pdf
from src.rag_chain import answer_question
from src.vector_store import create_vector_store

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session() -> None:
    """Initialize Streamlit session state."""
    defaults = {
        "vector_store": None,
        "messages": [],
        "processed_files": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def page_sort_key(source: dict) -> tuple:
    """Sort source references by filename and numeric page number."""
    filename = str(source.get("source", "Unknown"))
    page = source.get("page", "Unknown")

    try:
        page_number = int(page)
    except (TypeError, ValueError):
        page_number = float("inf")

    return filename.casefold(), page_number


def render_sources(sources: list[dict]) -> None:
    """Display sorted source references and readable excerpts."""
    if not sources:
        return

    sorted_sources = sorted(
        sources,
        key=lambda item: (
            str(item.get("source", "Unknown")).casefold(),
            int(item.get("page", 0))
            if str(item.get("page", "")).isdigit()
            else float("inf"),
        ),
    )

    with st.expander("📖 View source documents", expanded=False):
        for index, source in enumerate(sorted_sources, start=1):
            filename = source.get("source", "Unknown")
            page = source.get("page", "Unknown")
            excerpt = str(source.get("excerpt", "")).strip()
            labels = source.get("labels", [])

            st.markdown(f"**{filename} — Page {page}**")

            if labels:
                st.caption(
                    "Referenced as: "
                    + ", ".join(f"[{label}]" for label in labels)
                )

            if excerpt:
                with st.expander(
                    f"Read source excerpt {index}",
                    expanded=False,
                ):
                    st.markdown(excerpt)

            if index < len(sorted_sources):
                st.divider()

def process_uploaded_pdfs(uploaded_files) -> None:
    """Extract PDF text, create chunks, and build the FAISS index."""
    all_chunks = []
    failed_files = []

    progress = st.progress(0, text="Starting PDF processing...")

    try:
        for index, uploaded_file in enumerate(uploaded_files):
            try:
                chunks = process_pdf(
                    pdf_bytes=uploaded_file.getvalue(),
                    source_name=uploaded_file.name,
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                all_chunks.extend(chunks)

            except Exception as exc:
                failed_files.append((uploaded_file.name, exc))

            progress.progress(
                (index + 1) / len(uploaded_files),
                text=f"Processed {index + 1}/{len(uploaded_files)} PDFs",
            )

        if all_chunks:
            try:
                with st.spinner(
                    "Creating FAISS index with Hugging Face embeddings..."
                ):
                    vector_store = create_vector_store(all_chunks)

                st.session_state.vector_store = vector_store
                st.session_state.processed_files = sorted(
                    {
                        chunk.metadata.get("source", "Unknown")
                        for chunk in all_chunks
                    }
                )
                st.session_state.messages = []

                st.success(
                    f"Success! Indexed {len(all_chunks)} text chunks from "
                    f"{len(st.session_state.processed_files)} document(s)."
                )

            except Exception as exc:
                st.session_state.vector_store = None
                st.session_state.processed_files = []
                st.error("Could not create the FAISS search index.")
                st.code(f"{type(exc).__name__}: {exc}")

        for filename, exc in failed_files:
            st.warning(
                f"Could not process {filename}: "
                f"{type(exc).__name__}: {exc}"
            )

        if not all_chunks and not failed_files:
            st.warning("No text could be extracted from the uploaded PDFs.")

        if failed_files and all_chunks:
            st.info(
                "Some PDFs could not be processed. "
                "The search index contains the successfully processed PDFs."
            )

    finally:
        progress.empty()


def render_chat() -> None:
    """Render chat history and handle new questions."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your PDFs...")

    if not question or not question.strip():
        return

    question = question.strip()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = answer_question(
                    vector_store=st.session_state.vector_store,
                    question=question,
                    k=RETRIEVAL_K,
                )

                answer = result.get("answer", "")
                sources = result.get("sources", [])

                if not answer:
                    answer = "No answer was returned. Please try again."

                st.markdown(answer)
                render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                st.error(
                    "I couldn't generate an answer. "
                    "The exact error is shown below."
                )
                st.code(f"{type(exc).__name__}: {exc}")

                st.caption(
                    "Check your Gemini API key, model availability, "
                    "quota, and internet connection."
                )


initialize_session()

st.title("📚 DocuMind AI")
st.caption(
    "Your AI-powered PDF assistant — upload documents, ask questions, "
    "and explore answers grounded in your files."
)

with st.sidebar:
    st.header("📄 Your documents")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more text-based PDFs.",
    )

    process_button = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_button and uploaded_files:
        process_uploaded_pdfs(uploaded_files)

    if st.session_state.processed_files:
        st.divider()
        st.subheader("Indexed documents")

        for filename in st.session_state.processed_files:
            st.markdown(f"- `{filename}`")

        if st.button(
            "Clear documents and chat",
            use_container_width=True,
        ):
            st.session_state.vector_store = None
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Built with Streamlit, LangChain, FAISS and Gemini.")

if st.session_state.vector_store is None:
    st.info(
        "👈 Upload one or more PDFs in the sidebar and click "
        "**Process documents** to get started."
    )

    st.markdown("### What you can do")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📤 Upload")
        st.write(
            "Add your study notes, reports, or other text-based PDFs."
        )

    with col2:
        st.subheader("💬 Ask")
        st.write(
            "Ask questions in natural language about your documents."
        )

    with col3:
        st.subheader("🔎 Verify")
        st.write(
            "Inspect source excerpts and page numbers for answers."
        )

else:
    st.success(
        f"{len(st.session_state.processed_files)} document(s) ready for questions."
    )

    render_chat()
