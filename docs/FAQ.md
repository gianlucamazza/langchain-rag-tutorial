# Frequently Asked Questions (FAQ)

Common questions about LangChain RAG Tutorial.

## General Questions

### What is RAG?

**RAG (Retrieval-Augmented Generation)** combines:

- **Retrieval**: Finding relevant documents from a knowledge base
- **Generation**: Using an LLM to generate answers based on retrieved context

**Benefits:**

- Up-to-date information (vs static training data)
- Source attribution (cite documents)
- Reduced hallucinations (grounded in facts)

### Which architecture should I choose?

| If you need... | Use this | Notebook |
|---|---|---|
| Fast, simple Q&A | Simple RAG | [03_simple_rag.ipynb](../notebooks/fundamentals/03_simple_rag.ipynb) |
| Chatbot with memory | Memory RAG | [04_rag_with_memory.ipynb](../notebooks/advanced_architectures/04_rag_with_memory.ipynb) |
| Comprehensive research | Branched RAG or Fusion RAG | [05_branched_rag.ipynb](../notebooks/advanced_architectures/05_branched_rag.ipynb) |
| Ambiguous queries | HyDE | [06_hyde.ipynb](../notebooks/advanced_architectures/06_hyde.ipynb) |
| Mixed workload / cost optimization | Adaptive RAG | [07_adaptive_rag.ipynb](../notebooks/advanced_architectures/07_adaptive_rag.ipynb) |
| High accuracy with web fallback | Corrective RAG (CRAG) | [08_corrective_rag.ipynb](../notebooks/advanced_architectures/08_corrective_rag.ipynb) |
| Self-correcting system | Self-RAG | [09_self_rag.ipynb](../notebooks/advanced_architectures/09_self_rag.ipynb) |
| Complex multi-step reasoning | Agentic RAG | [10_agentic_rag.ipynb](../notebooks/advanced_architectures/10_agentic_rag.ipynb) |
| Technical documentation | Contextual RAG | [12_contextual_rag.ipynb](../notebooks/advanced_architectures/12_contextual_rag.ipynb) |
| Best ranking quality | Fusion RAG | [13_fusion_rag.ipynb](../notebooks/advanced_architectures/13_fusion_rag.ipynb) |
| Queries with identifiers, codes or jargon | Hybrid search + reranking | [19_hybrid_search_reranking.ipynb](../notebooks/advanced_architectures/19_hybrid_search_reranking.ipynb) |
| Small chunks lack context, or questions don't match the text's wording | Parent-document / multi-vector retrieval | [20_parent_multivector_retrieval.ipynb](../notebooks/advanced_architectures/20_parent_multivector_retrieval.ipynb) |
| Analytics/BI queries | SQL RAG | [14_sql_rag.ipynb](../notebooks/advanced_architectures/14_sql_rag.ipynb) |
| Knowledge graphs, multi-hop | GraphRAG | [15_graphrag.ipynb](../notebooks/advanced_architectures/15_graphrag.ipynb) |
| Images + text | Multimodal RAG | [17_multimodal_rag.ipynb](../notebooks/advanced_architectures/17_multimodal_rag.ipynb) |
| Domain-specific retrieval | Fine-tuned embeddings | [18_finetuning_embeddings.ipynb](../notebooks/advanced_architectures/18_finetuning_embeddings.ipynb) |
| Quality evaluation | RAGAS | [16_evaluation_ragas.ipynb](../notebooks/advanced_architectures/16_evaluation_ragas.ipynb) |

**Rule of thumb:** start with Simple RAG, add Contextual RAG for quality, then move to a specialized architecture for specific needs.

**Tip:** if retrieval on domain-specific content is weak, try fine-tuning embeddings (notebook 18). Reported percentage gains in older notes are teaching estimates, not a measured result on this repo.

### Do I need an OpenAI account?

**Yes, for this tutorial:**

- OpenAI API key required for GPT-4o-mini
- Minimum $5 credit recommended
- Tutorial costs ~$0.10-$0.50 total

**Alternatives:**

- Use HuggingFace embeddings (local, free)
- Use open-source LLMs (Ollama, Llama)
- Modify notebooks to use different providers

### Can I use this in production?

**Not as-is.** This repo is a hands-on tutorial. The notebooks show patterns you would typically need before production (evaluation in notebook 16, example Prometheus/Grafana config, starter FastAPI/Streamlit/Lambda apps). They are not a complete production system.

You would still need your own auth, tenancy, evaluation on *your* data, monitoring, rate limiting, and load testing. See [DEPLOYMENT.md](DEPLOYMENT.md) for the starter templates and a checklist of gaps.

## Installation & Setup

### What Python version do I need?

**Supported:** Python 3.10, 3.11, 3.12 and 3.13. Python 3.9 is not supported: it is end-of-life and LangChain 1.x requires Python 3.10+.

Check version:

```bash
python --version
```

### How much disk space is needed?

**Total: ~1.5GB**

- Dependencies: 500MB
- Virtual environment: 900MB
- HuggingFace model: 90MB (optional)
- Vector stores: 1-5MB

### Can I run this on Google Colab?

**Yes!** No local installation needed.

```python
# In Colab notebook
!git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
%cd langchain-rag-tutorial
!pip install -q -r requirements.txt

# Add API key via Colab Secrets
from google.colab import userdata
import os
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

## Cost & Performance

### How much does it cost to run?

**One-time costs:**

- Embeddings: ~$0.02 (10,000 documents)

**Per-query costs:**

- Simple RAG: $0.00036 per query
- Agentic RAG: $0.00360 per query

**Full tutorial:** $0.10-$0.50 total

See [PERFORMANCE.md](PERFORMANCE.md) for detailed breakdown.

### How long does it take to complete?

| Component | Time |
|---|---|
| Setup | 10 min |
| Fundamentals (01-03) | 30-40 min |
| One advanced architecture | 12-30 min |
| All advanced notebooks (04-20) | 4-5 hours |
| With evaluation (RAGAS) | +20 min |
| **Total** | **5-7 hours** |

**First run is slower** (model downloads, vector store creation, Chinook DB).

### Why is the first run so slow?

**Expected behavior:**

1. **HuggingFace model download**: ~90MB (one-time)
2. **Document embedding**: 8-10 min (one-time)
3. **Vector store creation**: Creates FAISS indexes

**Subsequent runs:** 10x faster (uses cached data)

### How can I speed things up?

**Quick wins:**

1. Run notebook 02 or `make vector-stores` first (creates vector stores)
2. Use cached vector stores (load, don't recreate)
3. Reduce k=2 (retrieve fewer documents)
4. Use HuggingFace embeddings (local, no API calls)

See [PERFORMANCE.md](PERFORMANCE.md) for optimization strategies.

## Technical Questions

### What's the difference between OpenAI and HuggingFace embeddings?

| Feature | OpenAI | HuggingFace |
|---|---|---|
| **Dimensions** | 1536 | 384 |
| **Quality** | Excellent | Good |
| **Cost** | $0.02/1M tokens | FREE |
| **Latency** | 100-200ms (API) | 500-1000ms (CPU) |
| **Offline** | No | Yes |

**Recommendation:** OpenAI when you want the notebook defaults; HuggingFace for local/offline demos.

### What is FAISS?

**FAISS** (Facebook AI Similarity Search):

- Vector similarity search library
- Fast nearest neighbor search
- Supports billions of vectors
- Used by Facebook, Google, etc.

**In this tutorial:**

- Stores document embeddings
- Enables fast retrieval (milliseconds)
- Persists to disk (no re-embedding)

### Do I need a GPU?

**No!** This tutorial runs on CPU.

**GPU helps with:**

- Faster local embeddings (HuggingFace)
- Larger local embedding or fine-tuning jobs

**Not needed for:**

- Tutorial completion
- OpenAI API calls (server-side)

### Can I use my own documents?

**Yes.** Load them with a LangChain document loader and split them with the shared helpers:

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load your documents
loader = TextLoader("path/to/your/docs.txt")
# or
loader = PyPDFLoader("path/to/your/docs.pdf")

docs = loader.load()
chunks = split_documents(docs)
```

Then create vector store as usual.

## Architecture-Specific

### When should I use Contextual RAG?

**Use Contextual RAG when:**

- Working with technical documentation
- Code documentation needs context
- Legal/policy documents where context matters
- Chunks need document-level understanding

**How it works:**

```
Original: "The function returns a list of tokens."
Contextualized: "Document: LangChain API | Section: Text Splitting | The function returns a list of tokens."
Result: better semantic matching (quality gains are workload-dependent and not measured in this repo)
```

**Benefits:** often better retrieval on technical docs, at the cost of an extra LLM call per chunk at index time.

### When should I use Fusion RAG?

**Use Fusion RAG when:**

- Ranking quality is critical
- Research and literature review
- Complex multi-aspect queries
- Need multi-query result aggregation

**How it works:**

1. Generate 3-5 query perspectives
2. Retrieve documents for each query
3. Combine results using Reciprocal Rank Fusion (RRF)
4. Documents appearing in multiple result sets rank higher

**Trade-off:** ~3x slower than Simple RAG, but best ranking quality.

### When should I use SQL RAG?

**Use SQL RAG when:**

- Querying structured databases (SQL)
- Business intelligence and analytics
- Data exploration tools
- Natural language to SQL conversion

**How it works:**

1. User asks: "Show top customers by revenue"
2. Retrieve relevant database schema
3. Generate SQL with validation
4. Execute safely (read-only)
5. Interpret results with LLM

**Benefits:** High accuracy *if* the generated SQL is valid and the schema is in scope. Model or SQL failures are possible; there is no guarantee. Includes the Chinook sample database.

### When should I use GraphRAG?

**Use GraphRAG when:**

- Building knowledge graphs
- Relationship-centric queries ("Who works with whom?")
- Multi-hop reasoning ("Friend of a friend")
- Network analysis and community detection

**How it works:**

1. Extract entities from documents
2. Extract relationships between entities
3. Build NetworkX graph
4. Query graph with traversal algorithms
5. Visualize with matplotlib

**Trade-off:** More complex setup, but excellent for relationship queries.

### When should I use HyDe?

**Use HyDe when:**

- Queries are ambiguous or vague
- Technical jargon needs translation
- Semantic matching is more important than keyword matching

**Example:**

```
Query: "How do I make my RAG faster?"
HyDe generates: "To optimize RAG performance, use vector store caching, reduce k parameter, and implement batch processing..."
Then embeds hypothetical answer for better retrieval.
```

### What's the difference between CRAG and Self-RAG?

| Feature | CRAG | Self-RAG |
|---|---|---|
| **Focus** | Document quality | Retrieval necessity |
| **Grading** | Relevance grader | Retrieval need classifier |
| **Fallback** | Web search | Re-retrieve |
| **Iterations** | 1 | 1-3 |
| **Use Case** | Out-of-domain queries | Self-correcting system |

### Why is Agentic RAG so slow?

**Agentic RAG uses ReAct pattern:**

1. **Think**: Analyze query
2. **Act**: Select tool (retriever, calculator, web)
3. **Observe**: Examine result
4. **Repeat**: Until answer found

**Each iteration = 1-2 LLM calls**
**Total**: 5-10 LLM calls, 20-40s latency

**Trade-off**: Slow but autonomous multi-step reasoning.

## Troubleshooting

Installation, API key, vector store and Jupyter problems are covered in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Contributing

### How can I contribute?

**Ways to contribute:**

- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests
- Share use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Can I add a new architecture?

**Yes!** Follow this process:

1. Create notebook: `notebooks/advanced_architectures/21_your_architecture.ipynb`
2. Add prompts to `shared/prompts.py`
3. Update comparison in `11_comparison.ipynb`
4. Document in `notebooks/advanced_architectures/README.md`
5. Update CHANGELOG.md with your addition
6. Submit pull request

## Licensing

### Can I use this commercially?

**Yes!** MIT License allows:

- Commercial use
- Modification
- Distribution
- Private use

**Requirements:**

- Keep the copyright and license notice
- Attribute original authors

### Can I fork and modify?

**Absolutely!** MIT License encourages forks.

**Please:**

- Link back to the original repository
- Share improvements (optional but appreciated)

## Getting Help

Still have questions?

1. **Check docs**: [docs/](.)
2. **Search issues**: [GitHub Issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
3. **Ask community**: [Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)
4. **Report bug**: [New Issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues/new)

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start
- [INSTALLATION.md](INSTALLATION.md) - Detailed setup
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [PERFORMANCE.md](PERFORMANCE.md) - Benchmarks
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
