"""
Prompt templates for LangChain RAG Tutorial
Provides reusable prompt templates for various RAG architectures.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate


# ============================================================================
# SIMPLE RAG PROMPTS
# ============================================================================

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question based on the context provided below.

If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which parts of the context you used to formulate your answer.

Context:
{context}"""),
    ("user", "{input}"),
])


RAG_WITH_METADATA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question based on the context provided below.

If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which parts of the context you used to formulate your answer.

Metadata:
- Retrieval strategy: {strategy}
- Number of documents: {num_docs}

Context:
{context}"""),
    ("user", "{input}"),
])


# ============================================================================
# MEMORY RAG PROMPTS
# ============================================================================

MEMORY_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question based on the context and conversation history.

If the context doesn't contain enough information to answer the question, say so clearly.
Use the conversation history to understand follow-up questions and references.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])


# ============================================================================
# HYDE PROMPTS
# ============================================================================

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant. Given a question, write a hypothetical document
that would perfectly answer this question. The document should be detailed and informative,
as if it came from a knowledge base.

Write the hypothetical document in a clear, factual style."""),
    ("user", "{question}"),
])


# ============================================================================
# ADAPTIVE RAG PROMPTS
# ============================================================================

COMPLEXITY_CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Analyze the complexity of this query and classify it as:
- SIMPLE: Direct factual question (1 concept)
- MEDIUM: Multi-concept or requires comparison
- COMPLEX: Multi-step reasoning, ambiguous, or specialized

Respond with only: SIMPLE, MEDIUM, or COMPLEX"""),
    ("user", "{query}"),
])


ADAPTIVE_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer based on context.

Strategy used: {strategy}

Context:
{context}"""),
    ("user", "{input}"),
])


# ============================================================================
# CORRECTIVE RAG (CRAG) PROMPTS
# ============================================================================

RELEVANCE_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing relevance of a document to a question.

If the document contains keyword(s) or semantic meaning related to the question,
grade it as relevant. Give a binary score 'yes' or 'no'.

Be strict but fair in your assessment."""),
    ("user", """Question: {question}

Document: {document}

Relevant (yes/no):"""),
])


CRAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer based on context.

Metadata:
- Used web search: {used_web}
- Relevance ratio: {relevance_ratio:.1%}

Context:
{context}"""),
    ("user", "{input}"),
])


# ============================================================================
# SELF-RAG PROMPTS
# ============================================================================

RETRIEVAL_NEED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant determining if external knowledge is needed.

Analyze the query and decide:
- YES: Query requires specific factual information from documents
- NO: Query can be answered with general knowledge

Respond with only YES or NO."""),
    ("user", "{query}"),
])


SELF_CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a critical evaluator. Assess this response for:
1. Factual accuracy (based on context)
2. Completeness
3. Relevance to question

Provide:
- SCORE: 1-5 (1=poor, 5=excellent)
- ISSUES: List any problems
- SHOULD_RETRY: yes/no (if score < 3)

Format:
SCORE: X
ISSUES: ...
SHOULD_RETRY: yes/no"""),
    ("user", """Question: {query}
Context: {context}
Response: {response}

Your evaluation:"""),
])


CITATION_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Check if the response properly cites the context.

Score YES if response uses information from context.
Score NO if response makes unsupported claims.

Respond: YES or NO"""),
    ("user", """Context: {context}
Response: {response}

Properly cited:"""),
])


# ============================================================================
# MULTI-QUERY PROMPTS (BRANCHED RAG)
# ============================================================================

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant. Generate 3 different versions of the user question
to retrieve relevant documents from a vector database. Provide these alternative questions
separated by newlines.

Original question: {question}

Alternative questions:""",
)


# ============================================================================
# AGENT PROMPTS (AGENTIC RAG)
# ============================================================================

REACT_AGENT_PROMPT = PromptTemplate.from_template("""Answer the following question as best you can.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Begin!

Question: {input}
Thought: {agent_scratchpad}""")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_prompt_by_name(name: str) -> ChatPromptTemplate | PromptTemplate:
    """
    Get a prompt template by name.

    Args:
        name: Prompt template name

    Returns:
        Prompt template

    Example:
        >>> prompt = get_prompt_by_name("RAG_PROMPT_TEMPLATE")
    """
    prompts = {
        "rag": RAG_PROMPT_TEMPLATE,
        "rag_metadata": RAG_WITH_METADATA_PROMPT,
        "memory": MEMORY_RAG_PROMPT,
        "hyde": HYDE_PROMPT,
        "complexity": COMPLEXITY_CLASSIFIER_PROMPT,
        "adaptive": ADAPTIVE_RAG_PROMPT,
        "relevance": RELEVANCE_GRADER_PROMPT,
        "crag": CRAG_PROMPT,
        "retrieval_need": RETRIEVAL_NEED_PROMPT,
        "self_critique": SELF_CRITIQUE_PROMPT,
        "citation": CITATION_CHECK_PROMPT,
        "multi_query": MULTI_QUERY_PROMPT,
        "react": REACT_AGENT_PROMPT,
    }

    if name.lower() not in prompts:
        raise ValueError(f"Unknown prompt name: {name}. Available: {list(prompts.keys())}")

    return prompts[name.lower()]


if __name__ == "__main__":
    # Test prompts
    print("=" * 80)
    print("TESTING PROMPT TEMPLATES")
    print("=" * 80)

    # Test RAG prompt
    print("\n1. RAG Prompt:")
    print(RAG_PROMPT_TEMPLATE.format(
        context="LangChain is a framework for building LLM applications.",
        input="What is LangChain?"
    ))

    # Test HyDe prompt
    print("\n2. HyDe Prompt:")
    print(HYDE_PROMPT.format(question="What is semantic search?"))

    # Test complexity classifier
    print("\n3. Complexity Classifier:")
    print(COMPLEXITY_CLASSIFIER_PROMPT.format(
        query="What is the capital of France?"
    ))

    print("\n✓ All prompts loaded successfully")
