# Advanced RAG Architectures

This directory contains tutorial implementations of advanced RAG patterns (notebooks 04–18), plus a RAGAS evaluation walkthrough. These are learning notebooks, not hardened services.

## Prerequisites

Before exploring advanced architectures, complete the fundamentals:

1. ✓ `fundamentals/01_setup_and_basics.ipynb`
2. ✓ `fundamentals/02_embeddings_comparison.ipynb`
3. ✓ `fundamentals/03_simple_rag.ipynb`

These provide the baseline components (vector stores, embeddings, retrievers) used by all advanced architectures.

---

## Architecture Overview

| Notebook | Architecture | Complexity | Use Case | Key Feature |
|----------|--------------|------------|----------|-------------|
| **04** | RAG with Memory | ⭐⭐ | Conversational AI, Support Bots | Maintains chat history for follow-up questions |
| **05** | Branched RAG | ⭐⭐⭐ | Multi-domain search, Analysis | Parallel sub-query generation |
| **06** | HyDe | ⭐⭐⭐ | Ambiguous queries, Specialized domains | Hypothetical document generation |
| **07** | Adaptive RAG | ⭐⭐⭐⭐ | Mixed workloads, Search tools | Query complexity routing |
| **08** | Corrective RAG (CRAG) | ⭐⭐⭐⭐ | High-stakes domains (legal, medical) | Relevance grading + web fallback |
| **09** | Self-RAG | ⭐⭐⭐⭐⭐ | Exploratory research, Dynamic Q&A | Self-critique and refinement |
| **10** | Agentic RAG | ⭐⭐⭐⭐⭐ | Multi-step reasoning, BI dashboards | Autonomous agents with tools |
| **11** | Comparison | - | Qualitative comparison | Side-by-side trade-off tables (not a measured harness) |
| **12** ✨ | Contextual RAG | ⭐⭐⭐ | Technical docs, Code documentation | Context-augmented chunking (Anthropic) |
| **13** ✨ | Fusion RAG | ⭐⭐⭐ | Research, Best ranking quality | Reciprocal Rank Fusion algorithm |
| **14** ✨ | SQL RAG | ⭐⭐⭐⭐ | Analytics, BI, Structured data | Natural Language to SQL with safety |
| **15** ✨ | GraphRAG | ⭐⭐⭐⭐⭐ | Knowledge graphs, Relationships | Entity extraction + multi-hop reasoning |
| **16** ✨ | RAGAS Evaluation | - | Quality assessment | Comprehensive RAG metrics framework |

---

## Detailed Descriptions

### 04_rag_with_memory.ipynb

#### RAG with Conversational Memory

Extends Simple RAG with conversation history to handle follow-up questions and anaphoric references.

**When to Use:**

- Chatbots and conversational interfaces
- Customer support systems
- Interactive Q&A sessions

**Key Components:**

- `ConversationBufferMemory` or `ConversationBufferWindowMemory`
- `RunnableWithMessageHistory` for LCEL integration
- Modified prompts with `MessagesPlaceholder`

**Example Query Flow:**

```text
User: "What is RAG?"
Bot: "RAG is Retrieval-Augmented Generation..."
User: "What are its main components?" ← References "RAG" from context
Bot: "The main components of RAG are..." ← Understands reference
```

Duration: ~10 minutes

---

### 05_branched_rag.ipynb

#### Multi-Query Parallel Retrieval

Generates multiple sub-queries from a single user question and retrieves documents in parallel for better coverage.

**When to Use:**

- Multi-intent queries
- Cross-domain research
- Comprehensive topic exploration

**Key Components:**

- `MultiQueryRetriever` (LangChain built-in)
- Query generation prompts
- Document deduplication

**Example:**

```text
Query: "Compare OpenAI and HuggingFace embeddings for cost and performance"

Generated sub-queries:
1. "OpenAI embeddings pricing and cost"
2. "HuggingFace embeddings performance benchmarks"
3. "Comparison of embedding providers"

→ Retrieves diverse documents covering all aspects
```

Duration: ~8 minutes

---

### 06_hyde.ipynb

#### Hypothetical Document Embeddings

Generates a hypothetical "perfect answer" document, embeds it, and uses it for retrieval instead of the raw query.

**When to Use:**

- Ambiguous or vague queries
- Domain-specific jargon
- Queries with abbreviations or shorthand

**Key Components:**

- HyDe prompt for document generation
- Two-step process: generate → embed → search
- Semantic similarity improvement

**Example:**

```text
Query: "How does MMR work?"

Hypothetical Doc (generated):
"MMR (Maximal Marginal Relevance) is a retrieval strategy that balances
relevance with diversity. It works by first fetching a larger set of
candidate documents, then iteratively selecting documents that are both
relevant to the query and dissimilar to already selected documents..."

→ Embedding this detailed description finds better matches
```

Duration: ~10 minutes

---

### 07_adaptive_rag.ipynb

#### Query Complexity-Based Routing

Analyzes query complexity and routes to the optimal retrieval strategy (simple, MMR, or HyDe).

**When to Use:**

- Mixed workload systems
- Cost optimization (use simple retrieval when possible)
- Performance/quality balance

**Key Components:**

- LLM-based complexity classifier
- Router logic (SIMPLE → similarity, MEDIUM → MMR, COMPLEX → HyDe)
- Performance monitoring

**Example:**

```text
"What is FAISS?" → SIMPLE → Fast similarity search
"Compare vector databases" → MEDIUM → MMR for diversity
"How to architect production RAG with privacy constraints?" → COMPLEX → HyDe
```

Duration: ~12 minutes

---

### 08_corrective_rag.ipynb

#### CRAG - Relevance Grading with Web Fallback

Grades retrieved documents for relevance and triggers web search if quality is low.

**When to Use:**

- High-accuracy requirements (legal, medical)
- Out-of-domain queries
- Fact-checking applications

**Key Components:**

- Relevance grader (LLM-based)
- DuckDuckGo web search tool
- Quality threshold logic

**Example:**

```text
Query: "What is the latest LangChain version released in 2025?"

Vector DB retrieval → Low relevance (outdated docs)
→ Trigger web search → Find current information
→ Combine sources → High-quality answer
```

Duration: ~15 minutes

---

### 09_self_rag.ipynb

#### Self-Reflective RAG with Auto-Critique

LLM decides autonomously when to retrieve, evaluates its own responses, and retries if quality is low.

**When to Use:**

- Exploratory research
- High-quality requirements
- Systems requiring self-correction

**Key Components:**

- Retrieval need classifier
- Response self-critique
- Iterative refinement loop
- Citation validation

**Example:**

```text
Query: "What is 5 + 7?"

Retrieval need: NO (general knowledge)
→ Direct answer: "12"
→ Self-critique: SCORE 5 → Approved

Query: "What are MMR parameters in LangChain?"

Retrieval need: YES (specific info needed)
→ Retrieve docs → Generate answer
→ Self-critique: SCORE 3 → Retry with more context
→ Improved answer → SCORE 5 → Approved
```

Duration: ~20 minutes

---

### 10_agentic_rag.ipynb

#### Autonomous Agent with Tools

Combines RAG with ReAct agents that can reason, plan, and use multiple tools (retriever, calculator, web search).

**When to Use:**

- Multi-step reasoning tasks
- BI dashboards and analytics
- Complex decision-making workflows

**Key Components:**

- ReAct agent (Reasoning + Acting)
- Tool suite (retriever, calculator, web search)
- Agent memory for conversation
- LangGraph orchestration

**Example:**

```text
Query: "If I have 10,000 documents and process 1M tokens/day,
        should I use OpenAI or HuggingFace embeddings?"

Agent reasoning:
1. Thought: Need to calculate embedding costs
   Action: Calculator → Cost estimation
2. Thought: Need embedding comparison info
   Action: Knowledge Base → Retrieve comparison
3. Thought: Analyze privacy/cost trade-offs
   Final Answer: "HuggingFace is better for your use case because..."
```

Duration: ~25 minutes

---

### 11_comparison.ipynb

#### Comprehensive Benchmark

Side-by-side comparison of all 12 architectures across various query types and metrics.

#### Metrics Evaluated

- Response time (latency)
- Token usage (cost)
- Success rate per query type
- Qualitative response quality

#### Query Types Tested

- Simple factual
- Follow-up questions
- Multi-concept queries
- Ambiguous queries
- Out-of-domain queries
- Complex reasoning

Duration: ~30 minutes (runs all architectures)

---

### 12_contextual_rag.ipynb ✨

#### Context-Augmented Chunking (Anthropic Technique)

Enhances document chunks by prepending them with document-level context, improving retrieval precision with minimal query overhead.

**When to Use:**

- Technical documentation
- Code documentation
- Legal/policy documents
- Any domain where chunks need broader context

**Key Components:**

- Document summarization with LLM
- Chunk-specific contextualization
- Context-augmented embeddings
- Extra document/chunk context at index time (quality not measured here)

**Example:**

```text
Original chunk: "The function returns a list of tokens."

Contextualized chunk:
"Document: LangChain Text Splitting API
Section: RecursiveCharacterTextSplitter methods
The function returns a list of tokens."

→ Embedding this contextualized version improves semantic matching
```

Duration: ~12 minutes

---

### 13_fusion_rag.ipynb ✨

#### RAG-Fusion with Reciprocal Rank Fusion

Generates multiple query perspectives and combines results using the RRF algorithm for superior ranking quality.

**When to Use:**

- Research and literature review
- Complex multi-aspect queries
- When ranking quality is critical
- Exploratory information gathering

**Key Components:**

- Multi-query generation (3-5 perspectives)
- Parallel retrieval for each query
- Reciprocal Rank Fusion (RRF) algorithm
- De-duplication with score aggregation

**Example:**

```text
Query: "How do I optimize RAG performance?"

Generated queries:
1. "RAG performance optimization techniques"
2. "Reduce latency in retrieval augmented generation"
3. "Improve RAG accuracy and speed"
4. "RAG caching and indexing strategies"

RRF Score Calculation:
For each document: score = sum(1 / (k + rank_i)) across all queries
→ Documents appearing in multiple result sets get higher scores
```

Duration: ~15 minutes

---

### 14_sql_rag.ipynb ✨

#### Natural Language to SQL

Converts natural language questions into SQL queries, executes them safely, and interprets results.

**When to Use:**

- Business intelligence and analytics
- Data exploration tools
- Reporting dashboards
- Any structured database queries

**Key Components:**

- Schema retrieval with semantic search
- Text-to-SQL generation with validation
- Safe SQL execution (read-only, SELECT only)
- SQL error recovery
- Result interpretation with LLM
- Chinook sample database (music store)

**Example:**

Query: "Show me the top 5 customers by total purchase amount"

Pipeline:

1. Retrieve relevant schema (Customer, Invoice, InvoiceLine tables)
2. Generate SQL:

   ```sql
   SELECT c.FirstName, c.LastName, SUM(i.Total) as TotalSpent
   FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId
   GROUP BY c.CustomerId ORDER BY TotalSpent DESC LIMIT 5
   ```

3. Execute safely (read-only connection)
4. Interpret results: "The top customer is Frank Harris who spent $144..."

Duration: ~18 minutes

---

### 15_graphrag.ipynb ✨

#### Graph-Based Knowledge Retrieval (Microsoft Research)

Extracts entities and relationships from documents, constructs a knowledge graph, and performs multi-hop reasoning.

**When to Use:**

- Knowledge graphs and ontologies
- Relationship-centric queries
- Multi-hop reasoning ("friend of a friend")
- Network analysis and community detection
- Exploratory knowledge discovery

**Key Components:**

- Entity extraction with LLM
- Relationship extraction and typing
- NetworkX graph construction
- Graph traversal algorithms
- Community detection (Louvain algorithm)
- Graph visualization

**Example:**

```text
Documents: "Alice works at OpenAI. Bob works at Anthropic. Alice and Bob are friends."

Graph Construction:
Nodes: [Alice, Bob, OpenAI, Anthropic]
Edges: [Alice --WORKS_AT--> OpenAI,
        Bob --WORKS_AT--> Anthropic,
        Alice --FRIEND--> Bob]

Query: "Who are Alice's colleagues' friends?"
Multi-hop: Alice → OpenAI → [employees] → [their friends]
```

Duration: ~25 minutes

---

### 16_evaluation_ragas.ipynb ✨

#### RAGAS Evaluation Framework

Comprehensive quality assessment for RAG systems using 6 evaluation metrics.

**When to Use:**

- Benchmarking multiple architectures
- Comparing chains on a labeled set before you ship anything
- A/B testing RAG changes
- Cost vs quality exploration (you still have to measure)

#### Key Metrics

1. **Faithfulness**: Answer grounded in context?
2. **Answer Relevancy**: Response addresses the question?
3. **Context Precision**: Relevant chunks ranked high?
4. **Context Recall**: All needed info retrieved?
5. **Answer Similarity**: Semantic match with ground truth?
6. **Answer Correctness**: Factual accuracy score

#### Example Evaluation

```text
Test Dataset:
- Question: "What is RAG?"
- Ground Truth: "RAG is Retrieval-Augmented Generation..."
- Context: [retrieved chunks]
- Answer: [generated response]

Scores:
- Faithfulness: 0.95 (well-grounded)
- Relevancy: 0.92 (on-topic)
- Precision: 0.88 (good retrieval)
- Recall: 0.85 (mostly complete)
- Similarity: 0.90 (semantically close)
- Correctness: 0.93 (factually accurate)
```

Duration: ~20 minutes

---

## Comparison matrix (qualitative)

Teaching estimates only — not timed or scored in this repo. Use notebook 16 to measure on your data.

| Architecture | Typical extra LLM work | Complexity | Best for exploring |
|--------------|------------------------|------------|--------------------|
| Simple RAG | 1 generation call | ⭐ | General Q&A |
| Memory RAG | 1 generation + history | ⭐⭐ | Follow-up questions |
| Branched RAG | Several query + 1 gen | ⭐⭐⭐ | Multi-intent questions |
| HyDe | Hypothetical doc + gen | ⭐⭐⭐ | Vague queries |
| Contextual RAG ✨ | Index-time context calls | ⭐⭐⭐ | Long technical docs |
| Fusion RAG ✨ | Several queries + RRF | ⭐⭐⭐ | Merging ranked lists |
| Adaptive RAG | Classify then route | ⭐⭐⭐⭐ | Mixed query types |
| SQL RAG ✨ | Schema + SQL + interpret | ⭐⭐⭐⭐ | Structured tables |
| CRAG | Grade + optional web | ⭐⭐⭐⭐ | Weak retrieval fallback |
| Self-RAG | Critique / retry loop | ⭐⭐⭐⭐⭐ | Self-check patterns |
| GraphRAG ✨ | Extract + traverse | ⭐⭐⭐⭐⭐ | Relationship questions |
| Agentic RAG | Multi-tool loop | ⭐⭐⭐⭐⭐ | Multi-step tool use |

---

## Shared Dependencies

All notebooks reuse components from `fundamentals`:

```python
# Shared utilities
from shared import (
    load_vector_store,          # Load pre-built vector stores
    RAG_PROMPT_TEMPLATE,        # Base prompts
    MEMORY_RAG_PROMPT,          # Memory-specific prompts
    HYDE_PROMPT,                # HyDe prompts
    # ... etc
)

# Shared artifacts
vectorstore_openai = load_vector_store("data/vector_stores/openai_embeddings", embeddings)
```

This avoids redundant embedding computation and ensures consistent baselines.

---

## Installation Notes

Some architectures require additional dependencies:

```bash
# For CRAG (web search)
pip install duckduckgo-search>=4.0.0

# For Agentic RAG (agent orchestration)
pip install langgraph>=0.0.20

# For GraphRAG (graph algorithms) ✨
pip install networkx>=3.2 matplotlib>=3.8.0

# For SQL RAG (database operations) ✨
pip install sqlalchemy>=2.0.25 pandas>=2.2.0

# For RAGAS Evaluation ✨
pip install ragas>=0.1.7 datasets>=2.16.0

# For advanced NLP (entity extraction) ✨
pip install spacy>=3.7.0
# Download spaCy model:
python -m spacy download en_core_web_sm

# Optional: Premium web search for CRAG
pip install tavily-python>=0.3.0
```

These are already included in `requirements.txt`.

---

## Progression Recommendations

**Beginner Path** (Start here):

1. 04_rag_with_memory.ipynb ← Easiest extension
2. 05_branched_rag.ipynb
3. 06_hyde.ipynb

**Intermediate Path**:

1. 12_contextual_rag.ipynb ✨ ← Context-augmented chunks
2. 13_fusion_rag.ipynb ✨ ← Best ranking quality
3. 07_adaptive_rag.ipynb
4. 08_corrective_rag.ipynb

**Advanced Path**:

1. 14_sql_rag.ipynb ✨ ← Natural language to SQL
2. 09_self_rag.ipynb
3. 10_agentic_rag.ipynb

**Expert Path** ✨:

1. 15_graphrag.ipynb ✨ ← Graph-based reasoning

**Analysis & Evaluation**:

1. 11_comparison.ipynb ← Benchmark all 12 architectures
2. 16_evaluation_ragas.ipynb ✨ ← Comprehensive quality metrics

---

## Before you take a pattern further

The notebooks show the pattern, not a deployable service. If you reuse an idea:

1. **Cost**: Trace token usage (`tiktoken` or provider logs)
2. **Latency**: Time retrieve vs generate separately
3. **Errors**: Add fallbacks the notebooks only sketch
4. **Caching**: Reuse embeddings and frequent queries
5. **Limits**: Provider rate limits are easy to hit in loops (Self-RAG, agents)
6. **Tracing**: Optional LangSmith keys are documented in `.env.example`

See each notebook's optimizations section for pattern-specific notes.

---

## 📖 Documentation

For comprehensive guides, see:

- 🚀 **[Getting Started](../../docs/GETTING_STARTED.md)** - Quick start (5 min)
- 🏗️ **[Architecture](../../docs/ARCHITECTURE.md)** - Design decisions
- ⚡ **[Performance](../../docs/PERFORMANCE.md)** - What to measure
- 🚀 **[Deployment](../../docs/DEPLOYMENT.md)** - Starter templates
- 📝 **[Examples](../../docs/EXAMPLES.md)** - Usage patterns
- 🐛 **[Troubleshooting](../../docs/TROUBLESHOOTING.md)** - Detailed troubleshooting
- ❓ **[FAQ](../../docs/FAQ.md)** - Common questions

---

## Resources

### Core RAG

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)

### Advanced Architectures

- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### New Architectures ✨

- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval) - Context-augmented chunking
- [RAG-Fusion Paper](https://arxiv.org/abs/2402.03367) - Reciprocal Rank Fusion
- [GraphRAG (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) - Graph-based RAG
- [RAGAS Framework](https://docs.ragas.io/) - RAG evaluation metrics
- [Text-to-SQL Survey](https://arxiv.org/abs/2208.13629) - Natural language to SQL

---

## Troubleshooting

**Issue**: "Vector store not found"

- Solution: Run `fundamentals/02_embeddings_comparison.ipynb` first

**Issue**: "Module 'shared' not found"

- Solution: Ensure you're in the project root or adjust `sys.path`

**Issue**: "Rate limit exceeded"

- Solution: Add delays between API calls or use batch processing

**Issue**: "DuckDuckGo search fails"

- Solution: Check internet connection or use Tavily as alternative

See main `README.md` for full troubleshooting guide.
