# Advanced RAG Architectures

This directory contains implementations of 8 advanced RAG architectures, each optimized for different use cases and requirements.

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
| **11** | Comparison | - | Benchmarking | Side-by-side performance analysis |

---

## Detailed Descriptions

### 04_rag_with_memory.ipynb

**RAG with Conversational Memory**

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

```
User: "What is RAG?"
Bot: "RAG is Retrieval-Augmented Generation..."
User: "What are its main components?" ← References "RAG" from context
Bot: "The main components of RAG are..." ← Understands reference
```

**Duration:** ~10 minutes

---

### 05_branched_rag.ipynb

**Multi-Query Parallel Retrieval**

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

```
Query: "Compare OpenAI and HuggingFace embeddings for cost and performance"

Generated sub-queries:
1. "OpenAI embeddings pricing and cost"
2. "HuggingFace embeddings performance benchmarks"
3. "Comparison of embedding providers"

→ Retrieves diverse documents covering all aspects
```

**Duration:** ~8 minutes

---

### 06_hyde.ipynb

**Hypothetical Document Embeddings**

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

```
Query: "How does MMR work?"

Hypothetical Doc (generated):
"MMR (Maximal Marginal Relevance) is a retrieval strategy that balances
relevance with diversity. It works by first fetching a larger set of
candidate documents, then iteratively selecting documents that are both
relevant to the query and dissimilar to already selected documents..."

→ Embedding this detailed description finds better matches
```

**Duration:** ~10 minutes

---

### 07_adaptive_rag.ipynb

**Query Complexity-Based Routing**

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

```
"What is FAISS?" → SIMPLE → Fast similarity search
"Compare vector databases" → MEDIUM → MMR for diversity
"How to architect production RAG with privacy constraints?" → COMPLEX → HyDe
```

**Duration:** ~12 minutes

---

### 08_corrective_rag.ipynb

**CRAG - Relevance Grading with Web Fallback**

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

```
Query: "What is the latest LangChain version released in 2025?"

Vector DB retrieval → Low relevance (outdated docs)
→ Trigger web search → Find current information
→ Combine sources → High-quality answer
```

**Duration:** ~15 minutes

---

### 09_self_rag.ipynb

**Self-Reflective RAG with Auto-Critique**

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

```
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

**Duration:** ~20 minutes

---

### 10_agentic_rag.ipynb

**Autonomous Agent with Tools**

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

```
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

**Duration:** ~25 minutes

---

### 11_comparison.ipynb

**Comprehensive Benchmark**

Side-by-side comparison of all 8 architectures across various query types and metrics.

**Metrics Evaluated:**

- Response time (latency)
- Token usage (cost)
- Success rate per query type
- Qualitative response quality

**Query Types Tested:**

- Simple factual
- Follow-up questions
- Multi-concept queries
- Ambiguous queries
- Out-of-domain queries
- Complex reasoning

**Duration:** ~30 minutes (runs all architectures)

---

## Comparison Matrix

| Architecture | Latency | Cost | Accuracy | Complexity | Best For |
|--------------|---------|------|----------|------------|----------|
| Simple RAG | Fast (2s) | Low | Good | ⭐ | General purpose |
| Memory RAG | Fast (2-3s) | Low-Med | Good | ⭐⭐ | Conversations |
| Branched RAG | Medium (5-8s) | Medium | Very Good | ⭐⭐⭐ | Multi-intent |
| HyDe | Medium (4-6s) | Medium | Very Good | ⭐⭐⭐ | Ambiguous queries |
| Adaptive RAG | Variable | Optimized | Very Good | ⭐⭐⭐⭐ | Mixed workloads |
| CRAG | Slow (10-15s) | High | Excellent | ⭐⭐⭐⭐ | High-accuracy |
| Self-RAG | Slow (10-20s) | High | Excellent | ⭐⭐⭐⭐⭐ | Quality-critical |
| Agentic RAG | Very Slow (20-40s) | Very High | Excellent | ⭐⭐⭐⭐⭐ | Complex reasoning |

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
4. 07_adaptive_rag.ipynb
5. 08_corrective_rag.ipynb

**Advanced Path**:
6. 09_self_rag.ipynb
7. 10_agentic_rag.ipynb

**Final Step**:
8. 11_comparison.ipynb ← Benchmark everything

---

## Production Considerations

Before deploying any advanced architecture:

1. **Cost Analysis**: Track token usage with `tiktoken`
2. **Latency Monitoring**: Profile each component
3. **Error Handling**: Implement robust fallbacks
4. **Caching**: Cache embeddings and frequent queries
5. **Rate Limiting**: Prevent API overuse
6. **Logging**: Use LangSmith for tracing

See each notebook's "Production Optimizations" section for specific guidance.

---

## 📖 Documentation

For comprehensive guides, see:

- 🚀 **[Getting Started](../../docs/GETTING_STARTED.md)** - Quick start (5 min)
- 🏗️ **[Architecture](../../docs/ARCHITECTURE.md)** - Design decisions
- ⚡ **[Performance](../../docs/PERFORMANCE.md)** - Benchmarks & optimization
- 🚀 **[Deployment](../../docs/DEPLOYMENT.md)** - Production setup
- 📝 **[Examples](../../docs/EXAMPLES.md)** - Usage patterns
- 🐛 **[Troubleshooting](../../docs/TROUBLESHOOTING.md)** - Detailed troubleshooting
- ❓ **[FAQ](../../docs/FAQ.md)** - Common questions

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

## Troubleshooting

**Issue**: "Vector store not found"

- **Solution**: Run `fundamentals/02_embeddings_comparison.ipynb` first

**Issue**: "Module 'shared' not found"

- **Solution**: Ensure you're in the project root or adjust `sys.path`

**Issue**: "Rate limit exceeded"

- **Solution**: Add delays between API calls or use batch processing

**Issue**: "DuckDuckGo search fails"

- **Solution**: Check internet connection or use Tavily as alternative

See main `README.md` for full troubleshooting guide.
