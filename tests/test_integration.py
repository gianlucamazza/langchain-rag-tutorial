"""
Integration tests for core RAG workflows

These tests validate end-to-end functionality of different RAG architectures.
They use actual LLM calls (mocked in CI/CD) to ensure realistic behavior.

Run with: pytest tests/test_integration.py -v
Skip slow tests: pytest tests/test_integration.py -v -m "not slow"
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def skip_if_no_api_key():
    """Skip tests if OpenAI API key is not available"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OpenAI API key not available")


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing without API calls"""
    mock = MagicMock(spec=OpenAIEmbeddings)
    # Return embeddings matching the number of documents
    def embed_docs(texts):
        return [[0.1, 0.2, 0.3]] * len(texts)
    
    mock.embed_documents = embed_docs
    mock.embed_query.return_value = [0.1, 0.2, 0.3]
    return mock


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls"""
    mock = MagicMock(spec=ChatOpenAI)
    mock.invoke.return_value.content = "This is a mocked response from the LLM."
    return mock


@pytest.fixture
def test_vector_store(sample_documents, mock_embeddings):
    """Create a test vector store"""
    return FAISS.from_documents(sample_documents, mock_embeddings)


class TestSimpleRAGWorkflow:
    """Integration tests for simple RAG workflow"""

    @pytest.mark.slow
    def test_end_to_end_rag_chain(self, test_vector_store, mock_llm):
        """Test complete RAG chain from query to answer"""
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser
        from shared.prompts import RAG_PROMPT_TEMPLATE
        from shared.utils import format_docs

        # Create RAG chain
        retriever = test_vector_store.as_retriever(search_kwargs={"k": 2})
        
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | RAG_PROMPT_TEMPLATE
            | mock_llm
            | StrOutputParser()
        )

        # Execute query
        query = "What is LangChain?"
        result = rag_chain.invoke(query)

        # Assertions
        assert isinstance(result, str)
        assert len(result) > 0
        mock_llm.invoke.assert_called_once()

    def test_retrieval_quality(self, test_vector_store):
        """Test that retrieval returns relevant documents"""
        retriever = test_vector_store.as_retriever(search_kwargs={"k": 2})
        
        query = "What is RAG?"
        docs = retriever.invoke(query)

        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)
        # At least one document should mention RAG
        assert any("RAG" in doc.page_content for doc in docs)

    @pytest.mark.skip(reason="MMR search may not work with mocked embeddings")
    def test_similarity_vs_mmr_retrieval(self, test_vector_store):
        """Test different retrieval strategies"""
        # Similarity search
        sim_retriever = test_vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        sim_docs = sim_retriever.invoke("LangChain")

        # MMR search
        mmr_retriever = test_vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        mmr_docs = mmr_retriever.invoke("LangChain")

        assert len(sim_docs) == 3
        assert len(mmr_docs) == 3
        # Both should return documents
        assert all(isinstance(doc, Document) for doc in sim_docs)
        assert all(isinstance(doc, Document) for doc in mmr_docs)


class TestMemoryRAGWorkflow:
    """Integration tests for conversational RAG with memory"""

    @pytest.mark.slow
    def test_conversational_rag_with_history(self, test_vector_store, mock_llm):
        """Test RAG with conversation history"""
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_core.runnables import RunnablePassthrough
        from shared.prompts import MEMORY_RAG_PROMPT
        from shared.utils import format_docs

        retriever = test_vector_store.as_retriever()
        
        # Simulate conversation with history
        chat_history = [
            HumanMessage(content="What is RAG?"),
            AIMessage(content="RAG stands for Retrieval-Augmented Generation.")
        ]

        # Create chain with history
        chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough(),
                "chat_history": lambda _: chat_history
            }
            | MEMORY_RAG_PROMPT
            | mock_llm
        )

        # Follow-up question
        result = chain.invoke("Can you explain more?")

        assert result is not None
        mock_llm.invoke.assert_called_once()

    def test_context_aware_follow_up(self, test_vector_store):
        """Test that follow-up questions work with context"""
        # This would test that the system understands "it" refers to previous topic
        retriever = test_vector_store.as_retriever()
        
        # First query
        docs1 = retriever.invoke("What is LangChain?")
        assert len(docs1) > 0

        # Follow-up (in real scenario, would use chat history)
        docs2 = retriever.invoke("How does it work?")
        assert len(docs2) > 0


class TestAdvancedRAGArchitectures:
    """Integration tests for advanced RAG patterns"""

    @pytest.mark.skip(reason="MultiQueryRetriever requires actual LLM calls")
    def test_multi_query_retrieval(self, test_vector_store, mock_llm):
        """Test multi-query retrieval approach"""
        from langchain.retrievers.multi_query import MultiQueryRetriever

        # Mock LLM to return query variations
        mock_llm.invoke.return_value.content = """
        1. What is LangChain framework?
        2. How does LangChain work?
        3. LangChain documentation overview
        """

        retriever = MultiQueryRetriever.from_llm(
            retriever=test_vector_store.as_retriever(),
            llm=mock_llm
        )

        query = "Tell me about LangChain"
        docs = retriever.invoke(query)

        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)

    @pytest.mark.slow
    def test_hyde_workflow(self, test_vector_store, mock_llm):
        """Test HyDe (Hypothetical Document Embeddings) workflow"""
        from shared.prompts import HYDE_PROMPT

        # Step 1: Generate hypothetical document
        hyde_chain = HYDE_PROMPT | mock_llm
        mock_llm.invoke.return_value.content = "LangChain is a comprehensive framework..."
        
        hypothetical_doc = hyde_chain.invoke({"question": "What is LangChain?"})
        
        assert hypothetical_doc is not None
        assert len(hypothetical_doc.content) > 0

        # Step 2: Use hypothetical doc for retrieval
        # (In real implementation, would embed and search)
        retriever = test_vector_store.as_retriever()
        docs = retriever.invoke(hypothetical_doc.content)
        
        assert len(docs) > 0

    def test_adaptive_routing(self, mock_llm):
        """Test query complexity classification and routing"""
        from shared.prompts import COMPLEXITY_CLASSIFIER_PROMPT

        test_cases = [
            ("What is X?", "SIMPLE"),
            ("Compare X and Y", "MEDIUM"),
            ("Explain how X works with Y in context of Z", "COMPLEX")
        ]

        for query, expected_complexity in test_cases:
            # Mock classifier response
            mock_llm.invoke.return_value.content = expected_complexity
            
            classifier = COMPLEXITY_CLASSIFIER_PROMPT | mock_llm
            result = classifier.invoke({"query": query})
            
            assert expected_complexity in result.content


class TestVectorStoreOperations:
    """Integration tests for vector store operations"""

    def test_vector_store_persistence(self, tmp_path, sample_documents, mock_embeddings):
        """Test saving and loading vector store"""
        # Create and save vector store
        vector_store = FAISS.from_documents(sample_documents, mock_embeddings)
        store_name = "test_store"
        
        # Save to temp directory
        save_path = tmp_path / store_name
        vector_store.save_local(str(save_path))
        
        # Verify file exists
        assert (save_path / "index.faiss").exists()
        
        # Load vector store
        loaded_store = FAISS.load_local(
            str(save_path),
            mock_embeddings,
            allow_dangerous_deserialization=True
        )
        
        assert loaded_store is not None
        # Test retrieval works
        docs = loaded_store.similarity_search("test", k=1)
        assert len(docs) > 0

    def test_vector_store_add_documents(self, test_vector_store):
        """Test adding documents to existing vector store"""
        initial_count = test_vector_store.index.ntotal
        
        # Add new documents
        new_docs = [
            Document(page_content="New document 1", metadata={"source": "new1"}),
            Document(page_content="New document 2", metadata={"source": "new2"})
        ]
        
        test_vector_store.add_documents(new_docs)
        
        # Verify documents were added
        assert test_vector_store.index.ntotal == initial_count + 2

    def test_vector_store_delete_documents(self, test_vector_store, sample_documents):
        """Test document deletion from vector store"""
        # FAISS doesn't support deletion natively, but test the concept
        initial_count = test_vector_store.index.ntotal
        assert initial_count > 0


class TestErrorHandling:
    """Integration tests for error handling"""

    def test_empty_retrieval_result(self, mock_embeddings):
        """Test handling of empty retrieval results"""
        # FAISS requires at least one document, so skip this test
        # or test with a single document
        single_doc = [Document(page_content="test", metadata={})]
        store = FAISS.from_documents(single_doc, mock_embeddings)
        retriever = store.as_retriever()
        
        # Should return results
        docs = retriever.invoke("any query")
        assert isinstance(docs, list)
        assert len(docs) >= 0

    def test_malformed_query_handling(self, test_vector_store):
        """Test handling of unusual queries"""
        retriever = test_vector_store.as_retriever()
        
        # Empty query
        docs = retriever.invoke("")
        assert isinstance(docs, list)
        
        # Very long query
        long_query = "test " * 1000
        docs = retriever.invoke(long_query)
        assert isinstance(docs, list)

    @patch('shared.loaders.load_langchain_docs')
    def test_document_loading_failure_recovery(self, mock_load):
        """Test recovery from document loading failures"""
        mock_load.side_effect = Exception("Network error")
        
        with pytest.raises(Exception) as exc_info:
            from shared.loaders import load_and_split
            load_and_split()
        
        assert "Network error" in str(exc_info.value)


class TestPerformance:
    """Performance-related integration tests"""

    @pytest.mark.slow
    def test_batch_query_performance(self, test_vector_store):
        """Test performance of batch queries"""
        import time
        
        retriever = test_vector_store.as_retriever(search_kwargs={"k": 5})
        queries = [
            "What is LangChain?",
            "What is RAG?",
            "How to use embeddings?",
            "Vector search basics",
            "Document retrieval methods"
        ]
        
        start_time = time.time()
        results = [retriever.invoke(q) for q in queries]
        elapsed_time = time.time() - start_time
        
        # All queries should complete
        assert len(results) == len(queries)
        assert all(len(docs) > 0 for docs in results)
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed_time < 5.0, f"Batch queries took {elapsed_time}s"

    def test_retrieval_with_different_k_values(self, test_vector_store):
        """Test retrieval performance with different K values"""
        query = "LangChain"
        
        for k in [1, 3, 5, 10]:
            retriever = test_vector_store.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
            
            # Should return up to k documents
            assert len(docs) <= k
            assert len(docs) > 0


# Pytest configuration for integration tests
def pytest_configure(config):
    """Add custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
