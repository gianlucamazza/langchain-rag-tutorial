# Advanced RAG Architectures

This directory contains notebooks 04-20: **14 advanced RAG architectures**, each suited to different use cases and requirements, plus a comparison benchmark, a RAGAS evaluation framework and an embedding fine-tuning guide.

## Prerequisites

Before exploring advanced architectures, complete the fundamentals:

1. `fundamentals/01_setup_and_basics.ipynb`
2. `fundamentals/02_embeddings_comparison.ipynb`
3. `fundamentals/03_simple_rag.ipynb`

These provide the baseline components (vector stores, embeddings, retrievers) used by all advanced architectures. Alternatively, build the vector stores with `make vector-stores` from the project root.

---

## Architecture Overview

| Notebook | Architecture | Complexity | Use Case | Key Feature |
|----------|--------------|------------|----------|-------------|
| **04** | RAG with Memory | 2/5 | Conversational AI, Support Bots | Maintains chat history for follow-up questions |
| **05** | Branched RAG | 3/5 | Multi-domain search, Analysis | Parallel sub-query generation |
| **06** | HyDe | 3/5 | Ambiguous queries, Specialized domains | Hypothetical document generation |
| **07** | Adaptive RAG | 4/5 | Mixed workloads, Search tools | Query complexity routing |
| **08** | Corrective RAG (CRAG) | 4/5 | High-stakes domains (legal, medical) | Relevance grading + web fallback |
| **09** | Self-RAG | 5/5 | Exploratory research, Dynamic Q&A | Self-critique and refinement |
| **10** | Agentic RAG | 5/5 | Multi-step reasoning, BI dashboards | Autonomous agents with tools |
| **11** | Comparison | - | Benchmarking | Side-by-side performance analysis |
| **12** | Contextual RAG | 3/5 | Technical docs, Code documentation | Context-augmented chunking (Anthropic) |
| **13** | Fusion RAG | 3/5 | Research, Best ranking quality | Reciprocal Rank Fusion algorithm |
| **14** | SQL RAG | 4/5 | Analytics, BI, Structured data | Natural Language to SQL with safety |
| **15** | GraphRAG | 5/5 | Knowledge graphs, Relationships | Entity extraction + multi-hop reasoning |
| **16** | RAGAS Evaluation | - | Quality assessment | Comprehensive RAG metrics framework |
| **17** | Multimodal RAG | 4/5 | Images + text, scanned documents | Vision model + OCR (Tesseract, Poppler) |
| **18** | Fine-tuning Embeddings | 4/5 | Domain-specific retrieval | Custom sentence-transformers models |
| **19** | Hybrid Search + Reranking | 3/5 | Identifiers and jargon in queries | BM25 + dense fusion (RRF) + cross-encoder reranker |
| **20** | Parent-Document and Multi-Vector | 3/5 | Chunk-size dilemma, vocabulary mismatch | Search small chunks or summaries/questions, return full context |

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

- `InMemoryChatMessageHistory` (`langchain_core`) for per-session history
- `trim_messages` to bound the history window
- `RunnableWithMessageHistory` for LCEL integration
- Modified prompts with `MessagesPlaceholder`

**Example Query Flow:**

```
User: "What is RAG?"
Bot: "RAG is Retrieval-Augmented Generation..."
User: "What are its main components?" <- References "RAG" from context
Bot: "The main components of RAG are..." <- Understands reference
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

-> Retrieves diverse documents covering all aspects
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
- Two-step process: generate -> embed -> search
- Semantic similarity improvement

**Example:**

```
Query: "How does MMR work?"

Hypothetical Doc (generated):
"MMR (Maximal Marginal Relevance) is a retrieval strategy that balances
relevance with diversity. It works by first fetching a larger set of
candidate documents, then iteratively selecting documents that are both
relevant to the query and dissimilar to already selected documents..."

-> Embedding this detailed description finds better matches
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
- Router logic (SIMPLE -> similarity, MEDIUM -> MMR, COMPLEX -> HyDe)
- Performance monitoring

**Example:**

```
"What is FAISS?" -> SIMPLE -> Fast similarity search
"Compare vector databases" -> MEDIUM -> MMR for diversity
"How to architect production RAG with privacy constraints?" -> COMPLEX -> HyDe
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
- Tavily web search tool (`langchain_tavily.TavilySearch`, requires `TAVILY_API_KEY`)
- Quality threshold logic

**Example:**

```
Query: "What is the latest LangChain version released in 2025?"

Vector DB retrieval -> Low relevance (outdated docs)
-> Trigger web search -> Find current information
-> Combine sources -> High-quality answer
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
-> Direct answer: "12"
-> Self-critique: SCORE 5 -> Approved

Query: "What are MMR parameters in LangChain?"

Retrieval need: YES (specific info needed)
-> Retrieve docs -> Generate answer
-> Self-critique: SCORE 3 -> Retry with more context
-> Improved answer -> SCORE 5 -> Approved
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
- Tool suite (retriever, `numexpr` calculator, Tavily web search)
- Agent memory for conversation
- LangGraph orchestration

**Example:**

```
Query: "If I have 10,000 documents and process 1M tokens/day,
        should I use OpenAI or HuggingFace embeddings?"

Agent reasoning:
1. Thought: Need to calculate embedding costs
   Action: Calculator -> Cost estimation
2. Thought: Need embedding comparison info
   Action: Knowledge Base -> Retrieve comparison
3. Thought: Analyze privacy/cost trade-offs
   Final Answer: "HuggingFace is better for your use case because..."
```

**Duration:** ~25 minutes

---

### 11_comparison.ipynb

**Comprehensive Benchmark**

Side-by-side comparison of all 12 architectures across various query types and metrics.

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

### 12_contextual_rag.ipynb

**Context-Augmented Chunking (Anthropic Technique)**

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
- ~15-30% better retrieval quality
- Full contextual retrieval (section 11): contextual embeddings + contextual BM25 + reranking, with an ablation of 4 configurations

**Example:**

```
Original chunk: "The function returns a list of tokens."

Contextualized chunk:
"Document: LangChain Text Splitting API
Section: RecursiveCharacterTextSplitter methods
The function returns a list of tokens."

-> Embedding this contextualized version improves semantic matching
```

**Duration:** ~12 minutes

---

### 13_fusion_rag.ipynb

**RAG-Fusion with Reciprocal Rank Fusion**

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

```
Query: "How do I optimize RAG performance?"

Generated queries:
1. "RAG performance optimization techniques"
2. "Reduce latency in retrieval augmented generation"
3. "Improve RAG accuracy and speed"
4. "RAG caching and indexing strategies"

RRF Score Calculation:
For each document: score = sum(1 / (k + rank_i)) across all queries
-> Documents appearing in multiple result sets get higher scores
```

**Duration:** ~15 minutes

---

### 14_sql_rag.ipynb

**Natural Language to SQL**

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

```
Query: "Show me the top 5 customers by total purchase amount"

Pipeline:
1. Retrieve relevant schema (Customer, Invoice, InvoiceLine tables)
2. Generate SQL:
   SELECT c.FirstName, c.LastName, SUM(i.Total) as TotalSpent
   FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId
   GROUP BY c.CustomerId ORDER BY TotalSpent DESC LIMIT 5
3. Execute safely (read-only connection)
4. Interpret results: "The top customer is Frank Harris who spent $144..."
```

**Duration:** ~18 minutes

---

### 15_graphrag.ipynb

**Graph-Based Knowledge Retrieval (Microsoft Research)**

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

```
Documents: "Alice works at OpenAI. Bob works at Anthropic. Alice and Bob are friends."

Graph Construction:
Nodes: [Alice, Bob, OpenAI, Anthropic]
Edges: [Alice --WORKS_AT--> OpenAI,
        Bob --WORKS_AT--> Anthropic,
        Alice --FRIEND--> Bob]

Query: "Who are Alice's colleagues' friends?"
Multi-hop: Alice -> OpenAI -> [employees] -> [their friends]
```

**Duration:** ~25 minutes

---

### 16_evaluation_ragas.ipynb

**RAGAS Evaluation Framework**

Comprehensive quality assessment for RAG systems using 6 evaluation metrics.

**When to Use:**

- Benchmarking multiple architectures
- Quality assurance before production
- A/B testing RAG improvements
- Cost-quality trade-off analysis

**Key Metrics:**

1. **Faithfulness**: Answer grounded in context?
2. **Answer Relevancy**: Response addresses the question?
3. **Context Precision**: Relevant chunks ranked high?
4. **Context Recall**: All needed info retrieved?
5. **Answer Similarity**: Semantic match with ground truth?
6. **Answer Correctness**: Factual accuracy score

**Example Evaluation:**

```
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

**Duration:** ~20 minutes

---

### 19_hybrid_search_reranking.ipynb

**Hybrid Search + Cross-Encoder Reranking**

Combines keyword retrieval (BM25) with dense retrieval (FAISS) through weighted Reciprocal Rank Fusion, then reranks the fused candidates with a local cross-encoder before they reach the LLM.

**When to Use:**

- Queries that mix natural language with identifiers (API names, error codes, SKUs)
- Corpora with domain jargon the embedding model has not seen
- When precision of the top few chunks matters more than a few hundred ms of latency

**Key Components:**

- `BM25Retriever` + FAISS combined with `EnsembleRetriever` (weighted RRF)
- `ContextualCompressionRetriever` + `CrossEncoderReranker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, set with `DEFAULT_RERANKER_MODEL`)
- Dense / BM25 / Hybrid / Hybrid+Rerank comparison with Hit@4 and MRR@4 on exact-identifier vs paraphrased queries
- `bm25_weight` sweep and inspection of reranker scores
- Reusable builders in `shared/retrievers.py`

**Example:**

```
Query: "What does max_retries control?"

Dense:  semantically close chunks, exact token often missed
BM25:   chunks containing "max_retries"
Hybrid: RRF merges both rankings (~20 candidates)
Rerank: cross-encoder scores each (query, chunk) pair -> top 4
```

No extra API key is needed; the first run downloads the reranker model (~90 MB).

**Duration:** ~15 minutes

---

### 20_parent_multivector_retrieval.ipynb

**Parent-Document and Multi-Vector Retrieval**

Separates what is searched from what is returned to the LLM: a vector store holds the searchable vectors, a docstore holds the text the LLM reads.

**When to Use:**

- Answers need more context than a small, precisely matching chunk provides
- Users ask questions in words the text does not use
- Dense chunks (code, tables) that embed poorly as raw text

**Key Components:**

- `ParentDocumentRetriever`: 400-char child chunks are embedded, 2000-char parents are returned
- `MultiVectorRetriever`: indexes an LLM summary and 3 hypothetical questions per chunk (`with_structured_output`), returns the original chunk
- Generated representations cached in `data/cache/`
- Comparison with the baseline on Hit@4, MRR@4 and context size

**Example:**

```
Chunk: "RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) ..."

Indexed representations:
- Summary: "Configuring chunk size and overlap for text splitting"
- Question: "How do I control the size of text chunks?"
- Question: "What does chunk_overlap do?"

-> A question-shaped query matches a question-shaped vector; the LLM gets the original chunk
```

**Duration:** ~15 minutes

---

## Comparison Matrix

| Architecture | Latency | Cost | Accuracy | Complexity | Best For |
|--------------|---------|------|----------|------------|----------|
| Simple RAG | Fast (2s) | Low | Good | 1/5 | General purpose |
| Memory RAG | Fast (2-3s) | Low-Med | Good | 2/5 | Conversations |
| Branched RAG | Medium (5-8s) | Medium | Very Good | 3/5 | Multi-intent |
| HyDe | Medium (4-6s) | Medium | Very Good | 3/5 | Ambiguous queries |
| Contextual RAG | Fast (2-3s) | Low | Very Good | 3/5 | Technical docs |
| Fusion RAG | Medium (5-8s) | Medium | Excellent | 3/5 | Research |
| Hybrid + Rerank | Fast (2-4s) | Low | Very Good | 3/5 | Identifiers, jargon |
| Parent / Multi-Vector | Fast (2-3s) | Low*** | Very Good | 3/5 | Chunk-size trade-offs |
| Adaptive RAG | Variable | Optimized | Very Good | 4/5 | Mixed workloads |
| SQL RAG | Fast (2-5s) | Low-Med | High if SQL valid* | 4/5 | Analytics |
| CRAG | Slow (10-15s) | High | Excellent | 4/5 | High-accuracy |
| Self-RAG | Slow (10-20s) | High | Excellent | 5/5 | Quality-critical |
| GraphRAG | Medium (3-8s) | High | Excellent** | 5/5 | Knowledge graphs |
| Agentic RAG | Very Slow (20-40s) | Very High | Excellent | 5/5 | Complex reasoning |

*High if the generated SQL is valid and the schema is in scope; model/SQL failure is possible (no guarantee) | **Strong for relationship queries | ***Multi-vector adds LLM calls at indexing time

---

## Shared Dependencies

All notebooks reuse components from `fundamentals`:

```python
from langchain_openai import OpenAIEmbeddings

from shared import RAG_PROMPT_TEMPLATE, HYDE_PROMPT, require_vector_store
from shared.config import OPENAI_EMBEDDING_MODEL, OPENAI_VECTOR_STORE_PATH
from shared.prompts import MEMORY_RAG_PROMPT

# Shared artifacts (data/vector_stores/openai__<OPENAI_EMBEDDING_MODEL>)
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vectorstore_openai = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
```

This avoids redundant embedding computation and ensures consistent baselines.

---

## Installation Notes

All dependencies are in the project `requirements.txt`. Architecture-specific requirements:

- **CRAG (08) and Agentic RAG (10)**: `langchain-tavily` for web search, plus `TAVILY_API_KEY`
  in `.env`; notebook 10 also uses `numexpr` for its calculator tool and `langgraph>=1.0`
- **SQL RAG (14)**: standard-library `sqlite3` and `pandas`
- **GraphRAG (15)**: `networkx`, `python-louvain`, `matplotlib`
- **RAGAS Evaluation (16)**: `ragas`, `datasets` (requires `langchain-community<0.4.2`, see
  [INSTALLATION.md](../../docs/INSTALLATION.md#why-langchain-community-is-pinned-below-042))
- **Multimodal RAG (17)**: `pillow`, `pytesseract`, `pdf2image`, plus the Tesseract and Poppler
  system packages
- **Fine-tuning (18)**: `sentence-transformers`, `accelerate`
- **Hybrid Search + Reranking (19)** and section 11 of **Contextual RAG (12)**: `rank-bm25`,
  `langchain-classic`, `sentence-transformers`; the reranker model (`cross-encoder/ms-marco-MiniLM-L-6-v2`,
  ~90 MB) is downloaded on first run
- **Parent-Document and Multi-Vector (20)**: `langchain-classic`

---

## Progression Recommendations

**Beginner Path** (Start here):

1. 04_rag_with_memory.ipynb <- Easiest extension
2. 05_branched_rag.ipynb
3. 06_hyde.ipynb

**Intermediate Path**:

4. 12_contextual_rag.ipynb  <- Context-augmented chunks
5. 13_fusion_rag.ipynb  <- Best ranking quality
6. 19_hybrid_search_reranking.ipynb  <- BM25 + dense + reranker
7. 20_parent_multivector_retrieval.ipynb  <- Decouple search from context
8. 07_adaptive_rag.ipynb
9. 08_corrective_rag.ipynb

**Advanced Path**:

10. 14_sql_rag.ipynb  <- Natural language to SQL
11. 09_self_rag.ipynb
12. 10_agentic_rag.ipynb

**Expert Path**:

13. 15_graphrag.ipynb (graph-based reasoning)
14. 17_multimodal_rag.ipynb (images + text)
15. 18_finetuning_embeddings.ipynb (custom embeddings)

**Analysis & Evaluation**:

16. 11_comparison.ipynb (benchmark of the architectures)
17. 16_evaluation_ragas.ipynb (quality metrics)

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

## Documentation

- [Getting Started](../../docs/GETTING_STARTED.md) - Quick start
- [Architecture](../../docs/ARCHITECTURE.md) - Design decisions
- [Performance](../../docs/PERFORMANCE.md) - Benchmarks and optimization
- [Deployment](../../docs/DEPLOYMENT.md) - Starter templates
- [Examples](../../docs/EXAMPLES.md) - Usage patterns
- [Troubleshooting](../../docs/TROUBLESHOOTING.md) - Detailed troubleshooting
- [FAQ](../../docs/FAQ.md) - Common questions

---

## Resources

**Core RAG:**
- [LangChain Documentation](https://docs.langchain.com/)
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)

**Advanced Architectures:**
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)

**Newer Architectures:**
- [Contextual Retrieval (Anthropic)](https://www.anthropic.com/news/contextual-retrieval) - Context-augmented chunking
- [RAG-Fusion Paper](https://arxiv.org/abs/2402.03367) - Reciprocal Rank Fusion
- [Reciprocal Rank Fusion (Cormack et al.)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Hybrid search fusion
- [GraphRAG (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) - Graph-based RAG
- [RAGAS Framework](https://docs.ragas.io/) - RAG evaluation metrics
- [Text-to-SQL Survey](https://arxiv.org/abs/2208.13629) - Natural language to SQL

---

## Troubleshooting

**Issue**: "Vector store not found"

- **Solution**: Run `fundamentals/02_embeddings_comparison.ipynb` or `make vector-stores` first

**Issue**: "Module 'shared' not found"

- **Solution**: Ensure you're in the project root or adjust `sys.path`

**Issue**: "Rate limit exceeded"

- **Solution**: Add delays between API calls or use batch processing

**Issue**: "Tavily search fails"

- **Solution**: Check that `TAVILY_API_KEY` is set in `.env` and `langchain-tavily` is installed

See [TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md) for the full guide.
