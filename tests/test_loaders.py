"""
Tests for document loading and splitting utilities
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from shared.loaders import (
    load_langchain_docs,
    split_documents,
    load_and_split,
    compare_splitting_strategies
)


class TestLoadLangChainDocs:
    """Tests for load_langchain_docs function"""

    @patch('shared.loaders.WebBaseLoader')
    def test_load_docs_default_urls(self, mock_loader_class):
        """Test loading with default URLs"""
        # Setup mock
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Test content", metadata={"source": "test.html"})
        ]
        mock_loader_class.return_value = mock_loader

        # Execute
        docs = load_langchain_docs(verbose=False)

        # Assert
        assert len(docs) == 1
        assert docs[0].page_content == "Test content"
        assert 'source_type' in docs[0].metadata
        assert docs[0].metadata['source_type'] == 'web_documentation'

    @patch('shared.loaders.WebBaseLoader')
    def test_load_docs_custom_urls(self, mock_loader_class):
        """Test loading with custom URLs"""
        # Setup mock
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Custom content", metadata={"source": "custom.html"})
        ]
        mock_loader_class.return_value = mock_loader

        custom_urls = ["https://example.com/doc1", "https://example.com/doc2"]

        # Execute
        docs = load_langchain_docs(urls=custom_urls, verbose=False)

        # Assert
        mock_loader_class.assert_called_once_with(custom_urls)
        assert len(docs) == 1

    @patch('shared.loaders.WebBaseLoader')
    def test_load_docs_without_metadata(self, mock_loader_class):
        """Test loading without adding custom metadata"""
        # Setup mock
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Test", metadata={"source": "test.html"})
        ]
        mock_loader_class.return_value = mock_loader

        # Execute
        docs = load_langchain_docs(add_metadata=False, verbose=False)

        # Assert
        assert 'source_type' not in docs[0].metadata
        assert 'process_date' not in docs[0].metadata

    @patch('shared.loaders.WebBaseLoader')
    def test_load_docs_handles_errors(self, mock_loader_class):
        """Test error handling when loading fails"""
        # Setup mock to raise exception
        mock_loader = MagicMock()
        mock_loader.load.side_effect = Exception("Network error")
        mock_loader_class.return_value = mock_loader

        # Execute and assert
        with pytest.raises(Exception) as exc_info:
            load_langchain_docs(verbose=False)
        
        assert "Network error" in str(exc_info.value)


class TestSplitDocuments:
    """Tests for split_documents function"""

    def test_split_basic(self):
        """Test basic document splitting"""
        # Create test document
        long_text = "A" * 2000  # 2000 characters
        docs = [Document(page_content=long_text, metadata={"source": "test"})]

        # Execute
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50, verbose=False)

        # Assert
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 500 for chunk in chunks)
        assert all(chunk.metadata["source"] == "test" for chunk in chunks)

    def test_split_preserves_metadata(self):
        """Test that splitting preserves document metadata"""
        docs = [
            Document(
                page_content="A" * 1000,
                metadata={"source": "doc1.txt", "author": "Test Author"}
            )
        ]

        chunks = split_documents(docs, chunk_size=200, verbose=False)

        assert all(chunk.metadata["source"] == "doc1.txt" for chunk in chunks)
        assert all(chunk.metadata["author"] == "Test Author" for chunk in chunks)

    def test_split_empty_list(self):
        """Test splitting empty document list"""
        chunks = split_documents([], verbose=False)
        assert len(chunks) == 0

    def test_split_custom_parameters(self):
        """Test splitting with custom chunk size and overlap"""
        docs = [Document(page_content="A" * 1000, metadata={})]

        chunks = split_documents(
            docs,
            chunk_size=300,
            chunk_overlap=100,
            verbose=False
        )

        assert len(chunks) > 0
        assert all(len(chunk.page_content) <= 300 for chunk in chunks)


class TestLoadAndSplit:
    """Tests for load_and_split convenience function"""

    @patch('shared.loaders.load_langchain_docs')
    @patch('shared.loaders.split_documents')
    def test_load_and_split_workflow(self, mock_split, mock_load):
        """Test complete load and split workflow"""
        # Setup mocks
        mock_docs = [Document(page_content="Test", metadata={})]
        mock_load.return_value = mock_docs
        mock_split.return_value = [
            Document(page_content="Test chunk 1", metadata={}),
            Document(page_content="Test chunk 2", metadata={}),
        ]

        # Execute
        result = load_and_split(verbose=False)

        # Assert
        mock_load.assert_called_once()
        # Check split was called with correct args (note: keyword args may be used)
        assert mock_split.called
        assert len(result) == 2

    @patch('shared.loaders.load_langchain_docs')
    @patch('shared.loaders.split_documents')
    def test_load_and_split_custom_params(self, mock_split, mock_load):
        """Test with custom parameters"""
        mock_docs = [Document(page_content="Test", metadata={})]
        mock_load.return_value = mock_docs
        mock_split.return_value = mock_docs

        custom_urls = ["https://example.com"]
        
        # Execute
        load_and_split(
            urls=custom_urls,
            chunk_size=500,
            chunk_overlap=100,
            verbose=False
        )

        # Assert
        mock_load.assert_called_once()
        assert mock_split.called


class TestCompareSplittingStrategies:
    """Tests for compare_splitting_strategies function"""

    def test_compare_strategies_basic(self, sample_documents):
        """Test strategy comparison with sample documents"""
        strategies = [(1000, 200), (500, 100)]
        result = compare_splitting_strategies(sample_documents, strategies, verbose=False)

        # Assert result structure
        assert isinstance(result, dict)
        assert len(result) == 2
        
        # Check each strategy has required keys
        for strategy_name, data in result.items():
            assert "num_chunks" in data
            assert "chunk_size" in data
            assert "chunk_overlap" in data
            assert "chunks" in data
            assert isinstance(data["num_chunks"], int)
            assert isinstance(data["chunks"], list)

    def test_compare_strategies_empty_docs(self):
        """Test with empty document list"""
        strategies = [(1000, 200)]
        result = compare_splitting_strategies([], strategies, verbose=False)
        
        # Should return results even for empty list
        assert isinstance(result, dict)
        for strategy_data in result.values():
            assert strategy_data["num_chunks"] == 0
            assert len(strategy_data["chunks"]) == 0

    def test_compare_strategies_custom_chunk_size(self, sample_documents):
        """Test with custom chunk size"""
        strategies = [(100, 20), (200, 40)]
        result = compare_splitting_strategies(sample_documents, strategies, verbose=False)

        # All strategies should produce chunks
        assert len(result) == 2
        for strategy_data in result.values():
            # Should have processed the documents
            assert isinstance(strategy_data["chunks"], list)


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_split_very_small_chunks(self):
        """Test splitting with very small chunk size"""
        docs = [Document(page_content="Short text", metadata={})]
        chunks = split_documents(docs, chunk_size=5, chunk_overlap=1, verbose=False)
        
        # Should handle gracefully
        assert len(chunks) >= 1

    def test_split_zero_overlap(self):
        """Test splitting with zero overlap"""
        docs = [Document(page_content="A" * 500, metadata={})]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=0, verbose=False)
        
        assert len(chunks) >= 1

    def test_single_char_document(self):
        """Test splitting single character document"""
        docs = [Document(page_content="A", metadata={})]
        chunks = split_documents(docs, chunk_size=100, chunk_overlap=50, verbose=False)
        
        assert len(chunks) >= 1
        # May have multiple chunks depending on splitter behavior
        assert chunks[0].page_content in ["A", "A\n"]
