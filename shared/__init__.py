"""
Shared utilities for LangChain RAG Tutorial
Provides reusable functions, configurations, and prompts across all notebooks.
"""

__version__ = "1.0.0"

from .config import (
    OPENAI_API_KEY,
    VECTOR_STORE_DIR,
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_K,
)

from .utils import (
    format_docs,
    load_vector_store,
    save_vector_store,
    print_section_header,
    print_results,
)

from .loaders import (
    load_langchain_docs,
    split_documents,
)

from .prompts import (
    RAG_PROMPT_TEMPLATE,
    RAG_WITH_METADATA_PROMPT,
    RELEVANCE_GRADER_PROMPT,
    HYDE_PROMPT,
)

__all__ = [
    # Config
    "OPENAI_API_KEY",
    "VECTOR_STORE_DIR",
    "CACHE_DIR",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_K",
    # Utils
    "format_docs",
    "load_vector_store",
    "save_vector_store",
    "print_section_header",
    "print_results",
    # Loaders
    "load_langchain_docs",
    "split_documents",
    # Prompts
    "RAG_PROMPT_TEMPLATE",
    "RAG_WITH_METADATA_PROMPT",
    "RELEVANCE_GRADER_PROMPT",
    "HYDE_PROMPT",
]
