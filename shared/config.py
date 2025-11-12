"""
Configuration module for LangChain RAG Tutorial
Centralizes API keys, paths, and default parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")

# Set User Agent for HTTP requests
USER_AGENT = os.getenv("USER_AGENT", "LangChain-RAG-Tutorial/1.0")
os.environ["USER_AGENT"] = USER_AGENT

# ============================================================================
# API KEYS
# ============================================================================

# Required: OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY not found in environment")
    print("  Please create .env file with: OPENAI_API_KEY=your-key-here")

# Set in environment for LangChain
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Optional: HuggingFace API Key (not needed for local embeddings)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if HUGGINGFACE_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

# Optional: Tavily API Key (for premium web search in CRAG)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Optional: LangSmith API Key (for tracing and monitoring)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    # Enable tracing if LANGSMITH_TRACING is set to true
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "langchain-rag-tutorial")

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# ============================================================================
# PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = DATA_DIR / "vector_stores"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Vector store paths
OPENAI_VECTOR_STORE_PATH = VECTOR_STORE_DIR / "openai_embeddings"
HF_VECTOR_STORE_PATH = VECTOR_STORE_DIR / "huggingface_embeddings"

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

# Text splitting
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "200"))

# Retrieval
DEFAULT_K = int(os.getenv("DEFAULT_K", "4"))  # Number of documents to retrieve
DEFAULT_MMR_FETCH_K = int(os.getenv("DEFAULT_MMR_FETCH_K", "20"))  # Documents to fetch before MMR filtering
DEFAULT_MMR_LAMBDA = float(os.getenv("DEFAULT_MMR_LAMBDA", "0.5"))  # Balance between relevance (1.0) and diversity (0.0)

# LLM
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0"))  # Deterministic responses

# Embeddings
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Document loading
DEFAULT_LANGCHAIN_URLS = [
    "https://python.langchain.com/docs/use_cases/question_answering/",
    "https://python.langchain.com/docs/modules/data_connection/retrievers/",
    "https://python.langchain.com/docs/modules/model_io/llms/",
    "https://python.langchain.com/docs/use_cases/chatbots/"
]

# ============================================================================
# DISPLAY SETTINGS
# ============================================================================

SECTION_WIDTH = int(os.getenv("SECTION_WIDTH", "80"))
PREVIEW_LENGTH = int(os.getenv("PREVIEW_LENGTH", "300"))  # Characters to show in document previews

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def verify_api_key() -> bool:
    """
    Verify that OpenAI API key is loaded.

    Returns:
        bool: True if key is loaded, False otherwise
    """
    if OPENAI_API_KEY:
        print("✓ OpenAI API Key: LOADED")
        print(f"  Preview: {OPENAI_API_KEY[:7]}...{OPENAI_API_KEY[-4:]}")
        return True
    else:
        print("✗ OpenAI API Key: NOT LOADED")
        print("\n⚠️  Setup instructions:")
        print("  1. Create .env file in project root")
        print("  2. Add: OPENAI_API_KEY=sk-proj-...")
        print("  3. Get key from: https://platform.openai.com/api-keys")
        print("  4. Restart kernel after updating .env")
        return False

def get_project_info() -> dict:
    """
    Get project configuration information.

    Returns:
        dict: Project configuration details
    """
    return {
        # Environment
        "environment": ENVIRONMENT,
        "debug_mode": DEBUG_MODE,
        "log_level": LOG_LEVEL,
        # Paths
        "project_root": str(PROJECT_ROOT),
        "vector_store_dir": str(VECTOR_STORE_DIR),
        "cache_dir": str(CACHE_DIR),
        # API Keys
        "openai_api_key_loaded": bool(OPENAI_API_KEY),
        "huggingface_api_key_loaded": bool(HUGGINGFACE_API_KEY),
        "tavily_api_key_loaded": bool(TAVILY_API_KEY),
        "langsmith_api_key_loaded": bool(LANGSMITH_API_KEY),
        # LLM Configuration
        "default_model": DEFAULT_MODEL,
        "default_temperature": DEFAULT_TEMPERATURE,
        "openai_embedding_model": OPENAI_EMBEDDING_MODEL,
        "hf_embedding_model": HF_EMBEDDING_MODEL,
        # Text Processing
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        # Retrieval
        "k": DEFAULT_K,
        "mmr_fetch_k": DEFAULT_MMR_FETCH_K,
        "mmr_lambda": DEFAULT_MMR_LAMBDA,
        # Display
        "section_width": SECTION_WIDTH,
        "preview_length": PREVIEW_LENGTH,
    }

if __name__ == "__main__":
    # Test configuration
    print("=" * SECTION_WIDTH)
    print("LANGCHAIN RAG - CONFIGURATION")
    print("=" * SECTION_WIDTH)

    info = get_project_info()
    for key, value in info.items():
        print(f"{key:.<30} {value}")

    print("\n" + "=" * SECTION_WIDTH)
    verify_api_key()
    print("=" * SECTION_WIDTH)
