"""
Shared utilities for LangChain RAG Tutorial
Provides reusable functions, configurations, and prompts across all notebooks.
"""

# ============================================================================
# EARLY WARNING SUPPRESSION
# Must run BEFORE any langchain/pydantic imports to prevent warnings
# ============================================================================
import logging
import warnings

# Suppress Pydantic V1 compatibility warnings (preventive, in case of Python 3.14+)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.v1")
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")

# Suppress USER_AGENT warning from langchain_community (loads before .env)
logging.getLogger("langchain_community.utils.user_agent").setLevel(logging.ERROR)

# Suppress other common deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__version__ = "1.4.0"

from .config import (  # noqa: E402
    CACHE_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_K,
    OPENAI_API_KEY,
    VECTOR_STORE_DIR,
)
from .loaders import (  # noqa: E402
    load_and_split,
    load_langchain_docs,
    split_documents,
)
from .prompts import (  # noqa: E402
    CHUNK_SUMMARY_PROMPT,
    ENTITY_DISAMBIGUATION_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    GRAPH_SUMMARIZATION_PROMPT,
    GRAPHRAG_ANSWER_PROMPT,
    HYDE_PROMPT,
    HYPOTHETICAL_QUESTIONS_PROMPT,
    RAG_PROMPT_TEMPLATE,
    RAG_WITH_METADATA_PROMPT,
    RELATIONSHIP_EXTRACTION_PROMPT,
    RELEVANCE_GRADER_PROMPT,
    SQL_ERROR_RECOVERY_PROMPT,
    SQL_RESULTS_INTERPRETATION_PROMPT,
    SQL_SCHEMA_SUMMARY_PROMPT,
    TEXT_TO_SQL_PROMPT,
)
from .retrievers import (  # noqa: E402
    bm25_tokenize,
    build_bm25_retriever,
    build_hybrid_retriever,
    build_reranking_retriever,
)
from .utils import (  # noqa: E402
    format_docs,
    load_vector_store,
    print_results,
    print_section_header,
    require_vector_store,
    save_vector_store,
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
    "require_vector_store",
    "save_vector_store",
    "print_section_header",
    "print_results",
    # Retrievers
    "bm25_tokenize",
    "build_bm25_retriever",
    "build_hybrid_retriever",
    "build_reranking_retriever",
    # Loaders
    "load_langchain_docs",
    "split_documents",
    "load_and_split",
    # Prompts
    "RAG_PROMPT_TEMPLATE",
    "RAG_WITH_METADATA_PROMPT",
    "RELEVANCE_GRADER_PROMPT",
    "HYDE_PROMPT",
    "SQL_SCHEMA_SUMMARY_PROMPT",
    "TEXT_TO_SQL_PROMPT",
    "SQL_RESULTS_INTERPRETATION_PROMPT",
    "SQL_ERROR_RECOVERY_PROMPT",
    "ENTITY_EXTRACTION_PROMPT",
    "RELATIONSHIP_EXTRACTION_PROMPT",
    "ENTITY_DISAMBIGUATION_PROMPT",
    "GRAPH_SUMMARIZATION_PROMPT",
    "GRAPHRAG_ANSWER_PROMPT",
    "CHUNK_SUMMARY_PROMPT",
    "HYPOTHETICAL_QUESTIONS_PROMPT",
]
