import streamlit as st

from pdf_processor import (
    extract_text_from_pdf,
    create_chunks
)

from vector_store import VectorStore

from agent import create_agent


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="PDF Q&A Agent",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0e1117;
    }

    h1, h2, h3 {
        color: white;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #1b1f27;
        border: 1px solid #444;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📄 PDF Question & Answer Agent")

st.write(
    "Upload a PDF and ask questions about its contents."
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "vector_store" not in st.session_state:

    st.session_state.vector_store = None


if "agent" not in st.session_state:

    st.session_state.agent = None


if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    st.divider()

    st.info(
        "Upload a PDF first, then ask questions "
        "about the document."
    )


# --------------------------------------------------
# PDF PROCESSING
# --------------------------------------------------

if uploaded_file is not None:

    if (
        st.session_state.pdf_name
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                # Extract text
                pages = extract_text_from_pdf(
                    uploaded_file
                )

                if not pages:

                    st.error(
                        "No readable text was found "
                        "in this PDF."
                    )

                else:

                    # Create chunks
                    chunks = create_chunks(
                        pages
                    )

                    # Create vector store
                    vector_store = VectorStore()

                    vector_store.create_index(
                        chunks
                    )

                    # Create agent
                    agent = create_agent(
                        vector_store
                    )

                    # Save in session
                    st.session_state.vector_store = (
                        vector_store
                    )

                    st.session_state.agent = agent

                    st.session_state.pdf_name = (
                        uploaded_file.name
                    )

                    st.success(
                        f"PDF processed successfully!"
                    )

                    st.write(
                        f"📄 Pages: {len(pages)}"
                    )

                    st.write(
                        f"🧩 Text chunks: {len(chunks)}"
                    )

            except Exception as e:

                st.error(
                    f"Error processing PDF: {e}"
                )


# --------------------------------------------------
# QUESTION AREA
# --------------------------------------------------

if st.session_state.agent is not None:

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the main conclusion of this PDF?"
    )

    ask_button = st.button(
        "🔍 Ask Agent",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Agent is searching the PDF..."
            ):

                try:

                    result = (
                        st.session_state.agent.invoke(
                            {
                                "question": question,
                                "context": "",
                                "sources": [],
                                "answer": ""
                            }
                        )
                    )

                    answer = result["answer"]

                    sources = result.get(
                        "sources",
                        []
                    )

                    # --------------------------------
                    # ANSWER
                    # --------------------------------

                    st.markdown(
                        "### 🤖 Answer"
                    )

                    st.markdown(answer)


                    # --------------------------------
                    # SOURCES
                    # --------------------------------

                    if sources:

                        st.markdown(
                            "### 📚 Sources"
                        )

                        for source in sources:

                            st.write(
                                f"📄 {source}"
                            )

                except Exception as e:

                    st.error(
                        f"Agent error: {e}"
                    )

else:

    st.info(
        "👈 Upload a PDF from the sidebar to begin."
    )