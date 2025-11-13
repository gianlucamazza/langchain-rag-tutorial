"""
Tests for prompt templates
"""

import pytest
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from shared.prompts import (
    RAG_PROMPT_TEMPLATE,
    RAG_WITH_METADATA_PROMPT,
    MEMORY_RAG_PROMPT,
    HYDE_PROMPT,
    COMPLEXITY_CLASSIFIER_PROMPT,
    ADAPTIVE_RAG_PROMPT,
    RELEVANCE_GRADER_PROMPT,
    CRAG_PROMPT,
    RETRIEVAL_NEED_PROMPT,
    SELF_CRITIQUE_PROMPT,
    CITATION_CHECK_PROMPT,
    MULTI_QUERY_PROMPT,
    REACT_AGENT_PROMPT,
    CONTEXTUAL_RAG_ANSWER_PROMPT,
    FUSION_RAG_ANSWER_PROMPT,
    TEXT_TO_SQL_PROMPT,
    ENTITY_EXTRACTION_PROMPT,
    GRAPHRAG_ANSWER_PROMPT,
)


class TestSimpleRAGPrompts:
    """Tests for simple RAG prompt templates"""

    def test_rag_prompt_template_structure(self):
        """Test RAG_PROMPT_TEMPLATE has correct structure"""
        assert isinstance(RAG_PROMPT_TEMPLATE, ChatPromptTemplate)
        
        # Test input variables
        assert "context" in RAG_PROMPT_TEMPLATE.input_variables
        assert "input" in RAG_PROMPT_TEMPLATE.input_variables

    def test_rag_prompt_template_formatting(self):
        """Test RAG_PROMPT_TEMPLATE can be formatted"""
        formatted = RAG_PROMPT_TEMPLATE.format_messages(
            context="Test context about LangChain",
            input="What is LangChain?"
        )
        
        assert len(formatted) == 2  # system + user messages
        assert "Test context about LangChain" in str(formatted)
        assert "What is LangChain?" in str(formatted)

    def test_rag_with_metadata_prompt_structure(self):
        """Test RAG_WITH_METADATA_PROMPT has correct variables"""
        assert isinstance(RAG_WITH_METADATA_PROMPT, ChatPromptTemplate)
        
        required_vars = ["context", "input", "strategy", "num_docs"]
        for var in required_vars:
            assert var in RAG_WITH_METADATA_PROMPT.input_variables

    def test_rag_with_metadata_prompt_formatting(self):
        """Test RAG_WITH_METADATA_PROMPT formatting with metadata"""
        formatted = RAG_WITH_METADATA_PROMPT.format_messages(
            context="Test context",
            input="Test question",
            strategy="similarity",
            num_docs="5"
        )
        
        assert len(formatted) == 2
        assert "similarity" in str(formatted)
        assert "5" in str(formatted)


class TestMemoryRAGPrompts:
    """Tests for memory-based RAG prompts"""

    def test_memory_rag_prompt_structure(self):
        """Test MEMORY_RAG_PROMPT has chat_history placeholder"""
        assert isinstance(MEMORY_RAG_PROMPT, ChatPromptTemplate)
        assert "context" in MEMORY_RAG_PROMPT.input_variables
        assert "input" in MEMORY_RAG_PROMPT.input_variables

    def test_memory_rag_prompt_with_history(self):
        """Test MEMORY_RAG_PROMPT with chat history"""
        from langchain_core.messages import HumanMessage, AIMessage
        
        formatted = MEMORY_RAG_PROMPT.format_messages(
            context="LangChain documentation",
            input="Tell me more",
            chat_history=[
                HumanMessage(content="What is RAG?"),
                AIMessage(content="RAG stands for Retrieval-Augmented Generation.")
            ]
        )
        
        # Should include system, history messages, and user message
        assert len(formatted) >= 3


class TestHyDePrompts:
    """Tests for HyDe (Hypothetical Document Embeddings) prompts"""

    def test_hyde_prompt_structure(self):
        """Test HYDE_PROMPT structure"""
        assert isinstance(HYDE_PROMPT, ChatPromptTemplate)
        assert "question" in HYDE_PROMPT.input_variables

    def test_hyde_prompt_formatting(self):
        """Test HYDE_PROMPT formatting"""
        formatted = HYDE_PROMPT.format_messages(
            question="What is machine learning?"
        )
        
        assert len(formatted) == 2
        assert "hypothetical" in str(formatted).lower()
        assert "What is machine learning?" in str(formatted)


class TestAdaptiveRAGPrompts:
    """Tests for Adaptive RAG prompts"""

    def test_complexity_classifier_prompt(self):
        """Test COMPLEXITY_CLASSIFIER_PROMPT"""
        assert isinstance(COMPLEXITY_CLASSIFIER_PROMPT, ChatPromptTemplate)
        assert "query" in COMPLEXITY_CLASSIFIER_PROMPT.input_variables

    def test_complexity_classifier_formatting(self):
        """Test complexity classifier formatting"""
        formatted = COMPLEXITY_CLASSIFIER_PROMPT.format_messages(
            query="What is the capital of France?"
        )
        
        formatted_text = str(formatted)
        assert "SIMPLE" in formatted_text
        assert "MEDIUM" in formatted_text
        assert "COMPLEX" in formatted_text

    def test_adaptive_rag_prompt(self):
        """Test ADAPTIVE_RAG_PROMPT"""
        assert isinstance(ADAPTIVE_RAG_PROMPT, ChatPromptTemplate)
        
        formatted = ADAPTIVE_RAG_PROMPT.format_messages(
            context="Test context",
            input="Test query",
            strategy="SIMPLE"
        )
        
        assert "SIMPLE" in str(formatted)


class TestCorrectiveRAGPrompts:
    """Tests for Corrective RAG (CRAG) prompts"""

    def test_relevance_grader_prompt(self):
        """Test RELEVANCE_GRADER_PROMPT"""
        assert isinstance(RELEVANCE_GRADER_PROMPT, ChatPromptTemplate)
        
        required_vars = ["question", "document"]
        for var in required_vars:
            assert var in RELEVANCE_GRADER_PROMPT.input_variables

    def test_relevance_grader_formatting(self):
        """Test relevance grader formatting"""
        formatted = RELEVANCE_GRADER_PROMPT.format_messages(
            question="What is LangChain?",
            document="LangChain is a framework for building LLM applications."
        )
        
        formatted_text = str(formatted)
        assert "relevant" in formatted_text.lower() or "RELEVANT" in formatted_text

    def test_crag_prompt(self):
        """Test CRAG_PROMPT structure"""
        assert isinstance(CRAG_PROMPT, ChatPromptTemplate)
        assert "context" in CRAG_PROMPT.input_variables
        assert "input" in CRAG_PROMPT.input_variables


class TestSelfRAGPrompts:
    """Tests for Self-RAG prompts"""

    def test_retrieval_need_prompt(self):
        """Test RETRIEVAL_NEED_PROMPT"""
        assert isinstance(RETRIEVAL_NEED_PROMPT, ChatPromptTemplate)
        assert "query" in RETRIEVAL_NEED_PROMPT.input_variables

    def test_self_critique_prompt(self):
        """Test SELF_CRITIQUE_PROMPT"""
        assert isinstance(SELF_CRITIQUE_PROMPT, ChatPromptTemplate)
        
        # Updated to match actual implementation (query, context, response)
        required_vars = ["query", "context", "response"]
        for var in required_vars:
            assert var in SELF_CRITIQUE_PROMPT.input_variables

    def test_citation_check_prompt(self):
        """Test CITATION_CHECK_PROMPT"""
        assert isinstance(CITATION_CHECK_PROMPT, ChatPromptTemplate)
        
        # Check for correct variables (context and response, not answer)
        required_vars = ["response", "context"]
        for var in required_vars:
            assert var in CITATION_CHECK_PROMPT.input_variables

    def test_self_critique_formatting(self):
        """Test self-critique prompt formatting"""
        formatted = SELF_CRITIQUE_PROMPT.format_messages(
            query="What is AI?",
            context="AI context here",
            response="AI is artificial intelligence."
        )
        
        assert "What is AI?" in str(formatted)
        assert "AI is artificial intelligence." in str(formatted)


class TestMultiQueryPrompts:
    """Tests for multi-query prompts"""

    def test_multi_query_prompt(self):
        """Test MULTI_QUERY_PROMPT"""
        # May be ChatPromptTemplate or PromptTemplate
        assert hasattr(MULTI_QUERY_PROMPT, 'input_variables')
        assert "question" in MULTI_QUERY_PROMPT.input_variables

    def test_multi_query_formatting(self):
        """Test multi-query prompt asks for variations"""
        # Use format() instead of format_messages() for PromptTemplate
        if isinstance(MULTI_QUERY_PROMPT, ChatPromptTemplate):
            formatted = MULTI_QUERY_PROMPT.format_messages(question="What is RAG?")
            formatted_text = str(formatted)
        else:
            formatted_text = MULTI_QUERY_PROMPT.format(question="What is RAG?")
        
        # Should ask for multiple variations/perspectives
        assert "What is RAG?" in formatted_text


class TestAgenticPrompts:
    """Tests for agentic RAG prompts"""

    def test_react_agent_prompt(self):
        """Test REACT_AGENT_PROMPT"""
        # May be ChatPromptTemplate or PromptTemplate
        assert hasattr(REACT_AGENT_PROMPT, 'input_variables')
        
        # ReAct prompts typically have these variables
        assert "input" in REACT_AGENT_PROMPT.input_variables or \
               "question" in REACT_AGENT_PROMPT.input_variables


class TestContextualRAGPrompts:
    """Tests for Contextual RAG prompts"""

    def test_contextual_rag_answer_prompt(self):
        """Test CONTEXTUAL_RAG_ANSWER_PROMPT"""
        assert isinstance(CONTEXTUAL_RAG_ANSWER_PROMPT, ChatPromptTemplate)
        assert "context" in CONTEXTUAL_RAG_ANSWER_PROMPT.input_variables
        assert "input" in CONTEXTUAL_RAG_ANSWER_PROMPT.input_variables


class TestFusionRAGPrompts:
    """Tests for Fusion RAG prompts"""

    def test_fusion_rag_answer_prompt(self):
        """Test FUSION_RAG_ANSWER_PROMPT"""
        assert isinstance(FUSION_RAG_ANSWER_PROMPT, ChatPromptTemplate)
        assert "context" in FUSION_RAG_ANSWER_PROMPT.input_variables
        assert "input" in FUSION_RAG_ANSWER_PROMPT.input_variables


class TestSQLRAGPrompts:
    """Tests for SQL RAG prompts"""

    def test_text_to_sql_prompt(self):
        """Test TEXT_TO_SQL_PROMPT"""
        assert isinstance(TEXT_TO_SQL_PROMPT, ChatPromptTemplate)
        
        # Should have schema and question variables
        required_vars = ["schema", "question"]
        for var in required_vars:
            assert var in TEXT_TO_SQL_PROMPT.input_variables

    def test_text_to_sql_formatting(self):
        """Test SQL prompt formatting"""
        formatted = TEXT_TO_SQL_PROMPT.format_messages(
            schema="CREATE TABLE users (id INT, name VARCHAR);",
            question="How many users are there?"
        )
        
        formatted_text = str(formatted)
        assert "CREATE TABLE" in formatted_text
        assert "How many users are there?" in formatted_text


class TestGraphRAGPrompts:
    """Tests for GraphRAG prompts"""

    def test_entity_extraction_prompt(self):
        """Test ENTITY_EXTRACTION_PROMPT"""
        assert isinstance(ENTITY_EXTRACTION_PROMPT, ChatPromptTemplate)
        assert "text" in ENTITY_EXTRACTION_PROMPT.input_variables

    def test_graphrag_answer_prompt(self):
        """Test GRAPHRAG_ANSWER_PROMPT"""
        assert isinstance(GRAPHRAG_ANSWER_PROMPT, ChatPromptTemplate)
        
        # Should have graph context and query
        required_vars = ["context", "input"]
        for var in required_vars:
            assert var in GRAPHRAG_ANSWER_PROMPT.input_variables


class TestPromptConsistency:
    """Tests for consistency across prompts"""

    def test_all_prompts_are_chat_templates(self):
        """Test that all exported prompts are valid template instances"""
        prompts_to_check = [
            RAG_PROMPT_TEMPLATE,
            MEMORY_RAG_PROMPT,
            HYDE_PROMPT,
            ADAPTIVE_RAG_PROMPT,
            CRAG_PROMPT,
            SELF_CRITIQUE_PROMPT,
        ]
        
        for prompt in prompts_to_check:
            # Should be either ChatPromptTemplate or PromptTemplate
            assert isinstance(prompt, (ChatPromptTemplate, PromptTemplate)), \
                f"Prompt {prompt} should be a valid template"

    def test_prompts_have_required_variables(self):
        """Test that prompts define input variables"""
        prompts_with_vars = [
            (RAG_PROMPT_TEMPLATE, ["context", "input"]),
            (HYDE_PROMPT, ["question"]),
            (COMPLEXITY_CLASSIFIER_PROMPT, ["query"]),
        ]
        
        for prompt, expected_vars in prompts_with_vars:
            for var in expected_vars:
                assert var in prompt.input_variables, \
                    f"Prompt should have variable '{var}'"

    def test_prompts_can_be_invoked(self):
        """Test that prompts can be invoked with correct parameters"""
        # Simple test that prompts don't raise errors when formatted
        try:
            # ChatPromptTemplate uses format_messages, PromptTemplate uses format
            if isinstance(RAG_PROMPT_TEMPLATE, ChatPromptTemplate):
                RAG_PROMPT_TEMPLATE.format_messages(context="test", input="test")
            else:
                RAG_PROMPT_TEMPLATE.format(context="test", input="test")
            
            if isinstance(HYDE_PROMPT, ChatPromptTemplate):
                HYDE_PROMPT.format_messages(question="test")
            else:
                HYDE_PROMPT.format(question="test")
            
            if isinstance(COMPLEXITY_CLASSIFIER_PROMPT, ChatPromptTemplate):
                COMPLEXITY_CLASSIFIER_PROMPT.format_messages(query="test")
            else:
                COMPLEXITY_CLASSIFIER_PROMPT.format(query="test")
        except Exception as e:
            pytest.fail(f"Prompt formatting should not raise exception: {e}")
