"""
Streamlit starter template for the LangChain RAG tutorial.
Interactive UI for querying the tutorial RAG chain.

Note: This template requires Streamlit dependencies.
Install with: pip install -r requirements.txt
"""

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from shared import format_docs, require_vector_store
from shared.config import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_VECTOR_STORE_PATH,
)
from shared.prompts import RAG_PROMPT_TEMPLATE

# Page configuration
st.set_page_config(
    page_title="LangChain RAG Tutorial",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_rag() -> tuple[Any | None, Any | None]:
    """Initialize RAG components (cached)"""
    try:
        # Load embeddings and vector store
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)

        # Initialize LLM
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)

        return vectorstore, llm

    except Exception as e:
        st.error(f"Failed to initialize RAG: {e}")
        return None, None


def main() -> None:
    """Main Streamlit application"""

    # Header
    st.markdown('<div class="main-header">LangChain RAG Tutorial</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        # Architecture selection
        architecture = st.selectbox(
            "RAG Architecture",
            ["Simple RAG", "Contextual RAG", "Fusion RAG"],
            help="Select the RAG architecture to use",
        )

        # Number of documents
        k = st.slider(
            "Documents to retrieve (k)",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of documents to retrieve from vector store",
        )

        # Temperature
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Controls randomness in responses",
        )

        st.divider()

        # Info
        st.subheader("About")
        st.markdown("""
        This is a tutorial RAG demo built with:
        - **LangChain** for RAG pipelines
        - **OpenAI** for embeddings and LLM
        - **FAISS** for vector search
        - **Streamlit** for UI

        [View on GitHub](https://github.com/gianlucamazza/langchain-rag-tutorial)
        """)

    # Initialize RAG
    with st.spinner("Loading RAG components..."):
        vectorstore, llm = initialize_rag()

    if vectorstore is None or llm is None:
        st.error("Failed to load RAG components. Check your configuration.")
        return

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Ask a Question")

        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="e.g., What is RAG and how does it work?",
            key="query_input",
        )

        # Submit button
        submit = st.button("Get Answer", type="primary", use_container_width=True)

    with col2:
        st.subheader("Statistics")

        # Placeholder for statistics
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Architecture", architecture.split()[0])
        with stat_col2:
            st.metric("Documents (k)", k)

    # Process query
    if submit and query:
        with st.spinner("Thinking..."):
            start_time = time.time()

            try:
                # Create retriever
                retriever = vectorstore.as_retriever(search_kwargs={"k": k})

                # Update LLM temperature
                llm.temperature = temperature

                # Build chain
                chain = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | RAG_PROMPT_TEMPLATE
                    | llm
                    | StrOutputParser()
                )

                # Get answer
                answer = chain.invoke(query)

                # Get source documents
                docs = retriever.invoke(query)

                # Calculate latency
                latency = (time.time() - start_time) * 1000

                # Display results
                st.divider()

                st.subheader("Answer")
                st.markdown(answer)

                st.divider()

                # Source documents
                with st.expander("Source Documents", expanded=False):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(
                            f"**Document {i}** (Source: {doc.metadata.get('source', 'unknown')})"
                        )
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Latency", f"{latency:.0f} ms")
                with col2:
                    st.metric("Sources", len(docs))
                with col3:
                    st.metric("Architecture", architecture.split()[0])

                # Success message
                st.success("Query completed successfully!")

            except Exception as e:
                st.error(f"Error: {e}")

    elif submit and not query:
        st.warning("Please enter a question.")

    # Sample queries
    st.divider()
    st.subheader("Try These Sample Queries")

    sample_queries = [
        "What is RAG and how does it work?",
        "How do I create a FAISS vector store?",
        "What's the difference between similarity and MMR retrieval?",
        "How does Adaptive RAG routing work?",
        "What are the cost optimization strategies for RAG?",
    ]

    cols = st.columns(len(sample_queries))
    for i, col in enumerate(cols):
        if col.button(f"Query {i + 1}", use_container_width=True):
            st.session_state.query_input = sample_queries[i]
            st.rerun()


if __name__ == "__main__":
    main()
