"""
Shared utilities for LangChain RAG Tutorial
Provides reusable functions, configurations, and prompts across all notebooks.
"""

# ============================================================================
# EARLY WARNING SUPPRESSION
# Must run BEFORE any langchain/pydantic imports to prevent warnings
# ============================================================================
import warnings
import logging

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

__version__ = "1.0.0"

from .config import (  # noqa: E402
    OPENAI_API_KEY,
    VECTOR_STORE_DIR,
    CACHE_DIR,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_K,
)

from .utils import (  # noqa: E402
    format_docs,
    load_vector_store,
    save_vector_store,
    print_section_header,
    print_results,
)

from .loaders import (  # noqa: E402
    load_langchain_docs,
    split_documents,
    load_and_split,
)

from .prompts import (  # noqa: E402
    RAG_PROMPT_TEMPLATE,
    RAG_WITH_METADATA_PROMPT,
    RELEVANCE_GRADER_PROMPT,
    HYDE_PROMPT,
    SQL_SCHEMA_SUMMARY_PROMPT,
    TEXT_TO_SQL_PROMPT,
    SQL_RESULTS_INTERPRETATION_PROMPT,
    SQL_ERROR_RECOVERY_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    RELATIONSHIP_EXTRACTION_PROMPT,
    ENTITY_DISAMBIGUATION_PROMPT,
    GRAPH_SUMMARIZATION_PROMPT,
    GRAPHRAG_ANSWER_PROMPT,
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
]
