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

MULTI_QUERY_PROMPT = PromptTemplate.from_template("""You are an AI assistant. Generate 3 different versions of the user question
to retrieve relevant documents from a vector database. Provide these alternative questions
separated by newlines.

Original question: {question}

Alternative questions:""")


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
# CONTEXTUAL RAG PROMPTS
# ============================================================================

DOCUMENT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Generate a concise summary of this document
that captures its main topics, purpose, and key information. This summary will be used to provide
context for individual chunks of the document.

Keep the summary to 2-3 sentences, focusing on what the document is about and its scope."""),
    ("user", """Document:
{document}

Summary:"""),
])


CONTEXTUAL_CHUNK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Given a document summary and a chunk of text,
create a brief contextual description that explains how this chunk relates to the overall document.

This context will be prepended to the chunk to improve retrieval. Keep it concise (1-2 sentences)."""),
    ("user", """Document Summary:
{doc_summary}

Chunk:
{chunk}

Contextual description:"""),
])


CONTEXTUAL_RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the user's question based on the
contextually-enriched documents provided below.

Each document has been augmented with contextual information to help you understand its
role within the larger document structure.

If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}"""),
    ("user", "{input}"),
])


# ============================================================================
# FUSION RAG PROMPTS
# ============================================================================

FUSION_QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping to improve search results. Generate {num_queries}
different versions of the user question to retrieve relevant documents from a vector database.

Create variations that:
1. Rephrase the question using different words
2. Break down complex questions into sub-questions
3. Add relevant context or expand abbreviations
4. Approach the question from different angles

Provide these alternative questions, one per line, numbered."""),
    ("user", """Original question: {question}

Alternative questions:"""),
])


FUSION_RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer based on the documents retrieved
using RAG-Fusion, which combines results from multiple query variations.

The documents have been ranked using Reciprocal Rank Fusion for optimal relevance.

Metadata:
- Original query: {original_query}
- Alternative queries generated: {num_queries}
- Unique documents retrieved: {num_docs}

Context:
{context}"""),
    ("user", "{input}"),
])


# ============================================================================
# SQL RAG PROMPTS
# ============================================================================

SQL_SCHEMA_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a database expert. Provide a clear, concise summary of this database
table that describes:
1. What data it contains
2. Key columns and their purposes
3. What kinds of questions it can answer

Keep it brief (2-3 sentences) and focus on the semantic meaning, not just technical details."""),
    ("user", """Table: {table_name}

Schema:
{schema}

Summary:"""),
])


TEXT_TO_SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL expert. Given a natural language question and database schema,
generate a valid SQL query to answer the question.

Rules:
1. Only use SELECT statements (no INSERT, UPDATE, DELETE, DROP)
2. Only query tables that exist in the schema
3. Use proper SQL syntax for the database type
4. Include LIMIT clause if appropriate
5. Use JOINs when needed to combine tables
6. Return ONLY the SQL query, no explanations

Available schema:
{schema}"""),
    ("user", """Question: {question}

SQL Query:"""),
])


SQL_RESULTS_INTERPRETATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Given a natural language question, the SQL query
that was executed, and its results, provide a clear, natural language answer.

Format the answer to be user-friendly and easy to understand. If the results are empty,
explain that no matching data was found."""),
    ("user", """Question: {question}

SQL Query:
{sql_query}

Results:
{results}

Answer:"""),
])


SQL_ERROR_RECOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL expert. A SQL query failed with an error. Analyze the error
and generate a corrected query.

Common issues:
- Table/column names not matching schema
- Syntax errors
- Invalid JOIN conditions
- Type mismatches

Return ONLY the corrected SQL query."""),
    ("user", """Original Question: {question}

Failed Query:
{failed_query}

Error:
{error}

Schema:
{schema}

Corrected Query:"""),
])


# ============================================================================
# GRAPHRAG PROMPTS
# ============================================================================

ENTITY_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant specialized in information extraction. Extract all
important entities from the text.

For each entity, provide:
- name: The entity name
- type: Entity type (PERSON, ORGANIZATION, LOCATION, CONCEPT, TECHNOLOGY, EVENT, etc.)
- description: Brief description (1 sentence)

Return ONLY valid JSON array format:
[
  {{"name": "...", "type": "...", "description": "..."}},
  ...
]"""),
    ("user", """Text:
{text}

Entities (JSON):"""),
])


RELATIONSHIP_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant specialized in relationship extraction. Given text and
a list of entities, identify relationships between them.

For each relationship, provide:
- source: Source entity name (must match entity list)
- relation: Relationship type (e.g., "created_by", "part_of", "used_in", "related_to")
- target: Target entity name (must match entity list)
- description: Brief description (1 sentence)

Return ONLY valid JSON array format:
[
  {{"source": "...", "relation": "...", "target": "...", "description": "..."}},
  ...
]

Entities:
{entities}"""),
    ("user", """Text:
{text}

Relationships (JSON):"""),
])


ENTITY_DISAMBIGUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping with entity disambiguation. Given a query
and a list of entities from a knowledge graph, identify which entities are most relevant
to the query.

Return the entity names that match or relate to the query, one per line."""),
    ("user", """Query: {query}

Available entities:
{entities}

Relevant entities:"""),
])


GRAPH_SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant. Summarize the information from this knowledge graph
subgraph in a clear, structured way.

Focus on:
1. Key entities and their relationships
2. Important connections and patterns
3. Relevant facts for answering questions

Present the information in a natural, readable format."""),
    ("user", """Subgraph:
{subgraph}

Summary:"""),
])


GRAPHRAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant. Answer the question using information from
the knowledge graph.

The context includes:
- Entities and their descriptions
- Relationships between entities
- Multi-hop connections

Metadata:
- Query entities: {query_entities}
- Graph hops explored: {num_hops}
- Nodes in subgraph: {num_nodes}

Context:
{context}"""),
    ("user", "{input}"),
])


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
    prompts: dict[str, ChatPromptTemplate | PromptTemplate] = {
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
        # Contextual RAG
        "document_summary": DOCUMENT_SUMMARY_PROMPT,
        "contextual_chunk": CONTEXTUAL_CHUNK_PROMPT,
        "contextual_rag": CONTEXTUAL_RAG_ANSWER_PROMPT,
        # Fusion RAG
        "fusion_query": FUSION_QUERY_GENERATION_PROMPT,
        "fusion_rag": FUSION_RAG_ANSWER_PROMPT,
        # SQL RAG
        "sql_schema": SQL_SCHEMA_SUMMARY_PROMPT,
        "text_to_sql": TEXT_TO_SQL_PROMPT,
        "sql_results": SQL_RESULTS_INTERPRETATION_PROMPT,
        "sql_error": SQL_ERROR_RECOVERY_PROMPT,
        # GraphRAG
        "entity_extraction": ENTITY_EXTRACTION_PROMPT,
        "relationship_extraction": RELATIONSHIP_EXTRACTION_PROMPT,
        "entity_disambiguation": ENTITY_DISAMBIGUATION_PROMPT,
        "graph_summarization": GRAPH_SUMMARIZATION_PROMPT,
        "graphrag": GRAPHRAG_ANSWER_PROMPT,
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
