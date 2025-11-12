"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    from langchain.schema import Document

    return [
        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "doc1.txt"}
        ),
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation.",
            metadata={"source": "doc2.txt"}
        ),
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return "This is a test response from the LLM."


@pytest.fixture
def test_query():
    """Sample test query"""
    return "What is LangChain?"
