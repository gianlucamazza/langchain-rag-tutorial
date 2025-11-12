"""
Utility functions for LangChain RAG Tutorial
Provides reusable functions for document formatting, vector store management, and display.
"""

import warnings
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from .config import SECTION_WIDTH, PREVIEW_LENGTH


# ============================================================================
# WARNING FILTERS
# ============================================================================

def suppress_warnings():
    """
    Suppress common non-critical warnings for cleaner notebook output.

    Following modern best practices:
    - Suppress deprecation warnings from dependencies (Pydantic V1)
    - Keep critical warnings visible for debugging
    """
    # Suppress Pydantic V1 deprecation warnings (from langchain_core)
    warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core._api.deprecation")
    warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

    # Suppress other common non-critical warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# Auto-suppress warnings on import
suppress_warnings()


def format_docs(docs: List[Document]) -> str:
    """
    Format a list of documents into a single string for use in prompts.

    Args:
        docs: List of LangChain Document objects

    Returns:
        str: Concatenated document content separated by newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)


def load_vector_store(
    path: str | Path,
    embeddings: Embeddings,
    verbose: bool = True
) -> FAISS:
    """
    Load a FAISS vector store from disk.

    Args:
        path: Path to the saved vector store directory
        embeddings: Embeddings instance (must match the one used to create the store)
        verbose: Whether to print status messages

    Returns:
        FAISS: Loaded vector store

    Example:
        >>> from langchain_openai import OpenAIEmbeddings
        >>> embeddings = OpenAIEmbeddings()
        >>> vectorstore = load_vector_store("data/vector_stores/openai", embeddings)
    """
    try:
        vectorstore = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True  # Required for pickle files
        )
        if verbose:
            print(f"✓ Loaded vector store from {path}")
        return vectorstore
    except Exception as e:
        if verbose:
            print(f"✗ Error loading vector store from {path}: {e}")
        raise


def save_vector_store(
    vectorstore: FAISS,
    path: str | Path,
    verbose: bool = True
) -> None:
    """
    Save a FAISS vector store to disk.

    Args:
        vectorstore: FAISS vector store to save
        path: Path where to save the vector store
        verbose: Whether to print status messages

    Example:
        >>> save_vector_store(vectorstore, "data/vector_stores/openai")
    """
    try:
        # Ensure parent directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        vectorstore.save_local(str(path))
        if verbose:
            print(f"✓ Saved vector store to {path}")
    except Exception as e:
        if verbose:
            print(f"✗ Error saving vector store to {path}: {e}")
        raise


def print_section_header(title: str, width: int = SECTION_WIDTH) -> None:
    """
    Print a formatted section header.

    Args:
        title: Section title
        width: Total width of the header

    Example:
        >>> print_section_header("Document Loading")
        ================================================================================
        DOCUMENT LOADING
        ================================================================================
    """
    print("\n" + "=" * width)
    print(title.upper())
    print("=" * width + "\n")


def print_results(
    docs: List[Document],
    title: str = "Retrieved Documents",
    max_docs: Optional[int] = None,
    preview_length: int = PREVIEW_LENGTH
) -> None:
    """
    Print formatted results from document retrieval.

    Args:
        docs: List of retrieved documents
        title: Title for the results section
        max_docs: Maximum number of documents to display (None = all)
        preview_length: Number of characters to show in content preview

    Example:
        >>> docs = retriever.invoke("What is RAG?")
        >>> print_results(docs, "Similarity Search Results", max_docs=3)
    """
    print(f"\n{title}")
    print("-" * SECTION_WIDTH)

    display_docs = docs[:max_docs] if max_docs else docs

    for i, doc in enumerate(display_docs, 1):
        print(f"\n{i}. Source: {doc.metadata.get('source', 'N/A')}")

        # Show additional metadata if available
        if 'source_type' in doc.metadata:
            print(f"   Type: {doc.metadata['source_type']}")
        if 'process_date' in doc.metadata:
            print(f"   Date: {doc.metadata['process_date']}")

        # Show content preview
        content = doc.page_content[:preview_length]
        if len(doc.page_content) > preview_length:
            content += "..."
        print(f"   Content: {content}")

    if max_docs and len(docs) > max_docs:
        print(f"\n... and {len(docs) - max_docs} more documents")


def print_comparison_table(
    data: List[List[str]],
    headers: Optional[List[str]] = None
) -> None:
    """
    Print a formatted comparison table.

    Args:
        data: Table data as list of rows (each row is a list of strings)
        headers: Optional header row

    Example:
        >>> data = [
        ...     ["Feature", "OpenAI", "HuggingFace"],
        ...     ["Dimension", "1536", "384"],
        ...     ["Cost", "Paid", "Free"]
        ... ]
        >>> print_comparison_table(data)
    """
    if not data:
        return

    # Use first row as headers if not provided
    if headers is None and data:
        headers = data[0]
        data = data[1:]

    # Calculate column widths
    all_rows = [headers] + data if headers else data
    col_widths = [max(len(str(row[i])) for row in all_rows) + 2
                  for i in range(len(all_rows[0]))]

    # Print headers
    if headers:
        print("".join(str(item).ljust(col_widths[j])
                      for j, item in enumerate(headers)))
        print("-" * sum(col_widths))

    # Print data rows
    for row in data:
        print("".join(str(item).ljust(col_widths[j])
                      for j, item in enumerate(row)))


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate the number of tokens in a text string.

    Args:
        text: Text to tokenize
        model: Model name for tokenizer (default: gpt-3.5-turbo)

    Returns:
        int: Estimated number of tokens

    Note:
        Requires tiktoken package. This is an approximation.
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4
    except Exception:
        return len(text) // 4


def estimate_embedding_cost(
    texts: List[str],
    model: str = "text-embedding-3-small",
    cost_per_million: float = 0.02
) -> tuple[int, float]:
    """
    Estimate the cost of embedding a list of texts with OpenAI.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name
        cost_per_million: Cost per million tokens (default for text-embedding-3-small)

    Returns:
        tuple: (total_tokens, estimated_cost_usd)

    Example:
        >>> texts = [doc.page_content for doc in chunks]
        >>> tokens, cost = estimate_embedding_cost(texts)
        >>> print(f"Estimated cost: ${cost:.4f} for {tokens:,} tokens")
    """
    total_tokens = sum(estimate_tokens(text) for text in texts)
    estimated_cost = (total_tokens / 1_000_000) * cost_per_million
    return total_tokens, estimated_cost


if __name__ == "__main__":
    # Test utilities
    print_section_header("Testing Utilities")

    # Test comparison table
    data = [
        ["Feature", "OpenAI", "HuggingFace"],
        ["Dimension", "1536", "384"],
        ["Cost", "Paid", "Free"],
        ["Speed", "Fast", "Medium"]
    ]
    print_comparison_table(data)

    # Test token estimation
    test_text = "This is a test string for token estimation."
    tokens = estimate_tokens(test_text)
    print(f"\nTest text tokens: {tokens}")
