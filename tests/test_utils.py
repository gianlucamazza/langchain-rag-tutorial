"""
Tests for shared/utils.py
"""

import pytest
from shared.utils import format_docs, estimate_tokens
from langchain_core.documents import Document


def test_format_docs():
    """Test format_docs function"""
    docs = [
        Document(page_content="First doc"),
        Document(page_content="Second doc"),
    ]

    result = format_docs(docs)
    
    assert "First doc" in result
    assert "Second doc" in result
    assert "\n\n" in result  # Check separator


def test_format_docs_empty():
    """Test format_docs with empty list"""
    result = format_docs([])
    assert result == ""


def test_estimate_tokens():
    """Test token estimation"""
    text = "This is a test sentence."
    
    tokens = estimate_tokens(text)
    
    assert isinstance(tokens, int)
    assert tokens > 0
    assert tokens < 100  # Reasonable range for short text


def test_estimate_tokens_empty():
    """Test token estimation with empty string"""
    tokens = estimate_tokens("")
    assert tokens == 0
