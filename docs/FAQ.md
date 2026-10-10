# Frequently Asked Questions (FAQ)

Common questions about LangChain RAG Tutorial.

## General Questions

### What is RAG?

**RAG (Retrieval-Augmented Generation)** combines:

- **Retrieval**: Finding relevant documents from a knowledge base
- **Generation**: Using an LLM to generate answers based on retrieved context

**Benefits:**

- ✅ Up-to-date information (vs static training data)
- ✅ Source attribution (cite documents)
- ✅ Reduced hallucinations (grounded in facts)

### Which architecture should I choose?

| If you need... | Use this |
|---|---|
| Fast, simple Q&A | Simple RAG |
| Chatbot with memory | Memory RAG |
| Comprehensive research | Fusion RAG ✨ |
| Handle ambiguous queries | HyDe |
| Technical documentation | Contextual RAG ✨ |
| Rank fusion across query variants | Fusion RAG ✨ |
| Mixed workload optimization | Adaptive RAG |
| High accuracy + web fallback | Corrective RAG (CRAG) |
| Self-correcting system | Self-RAG |
| Complex multi-step reasoning | Agentic RAG |
| Analytics/BI queries | SQL RAG ✨ |
| Knowledge graphs | GraphRAG ✨ |
| Quality evaluation | RAGAS ✨ |

**Rule of thumb:** Start with Simple RAG → Add Contextual for quality → Use specialized for specific needs.

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

**Required:** Python 3.9+  
**Recommended:** Python 3.10 or 3.11

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

**Per-query costs (illustrative list-price arithmetic, not measured):**

- Simple RAG: on the order of fractions of a cent per query with GPT-4o-mini
- Agentic RAG: several times that (more LLM calls)

**Full tutorial:** often well under a dollar if you use the cheap chat model, but it depends on retries and how many cells you re-run.

See [PERFORMANCE.md](PERFORMANCE.md) for assumed token counts and how to measure.

### How long does it take to complete?

| Component | Time |
|---|---|
| Setup | 10 min |
| Fundamentals (01-03) | 30-40 min |
| One advanced architecture | 12-30 min |
| Advanced notebooks (pick several) | 3-4 hours |
| With evaluation (RAGAS) | +20 min |
| **Total** | **4-6 hours** |

**First run is slower** (model downloads, vector store creation, Chinook DB).

### Why is the first run so slow?

**Expected behavior:**

1. **HuggingFace model download**: ~90MB (one-time)
2. **Document embedding**: 8-10 min (one-time)
3. **Vector store creation**: Creates FAISS indexes

**Subsequent runs:** 10x faster (uses cached data)

### How can I speed things up?

**Quick wins:**

1. ✅ Run notebook 02 first (creates vector stores)
2. ✅ Use cached vector stores (load, don't recreate)
3. ✅ Reduce k=2 (retrieve fewer documents)
4. ✅ Use HuggingFace embeddings (local, no API calls)

See [PERFORMANCE.md](PERFORMANCE.md) for optimization strategies.

## Technical Questions

### What's the difference between OpenAI and HuggingFace embeddings?

| Feature | OpenAI | HuggingFace |
|---|---|---|
| **Dimensions** | 1536 | 384 |
| **Quality** | Excellent | Good |
| **Cost** | $0.02/1M tokens | FREE |
| **Latency** | 100-200ms (API) | 500-1000ms (CPU) |
| **Offline** | ❌ No | ✅ Yes |

**In this tutorial:** OpenAI is the default for quality-sensitive notebook cells; HuggingFace is the local/no-cost option in notebook 02.

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
- Training or serving large local embedding models

**Not needed for:**

- Tutorial completion
- OpenAI API calls (server-side)

### Can I use my own documents?

**Yes!** Modify `shared/loaders.py`:

```python
from langchain_community.document_loaders import TextLoader, PDFLoader

# Load your documents
loader = TextLoader("path/to/your/docs.txt")
# or
loader = PDFLoader("path/to/your/docs.pdf")

docs = loader.load()
chunks = split_documents(docs)
```

Then create vector store as usual.

## Architecture-Specific

### When should I use Contextual RAG? ✨

**Use Contextual RAG when:**

- Working with technical documentation
- Code documentation needs context
- Legal/policy documents where context matters
- Chunks need document-level understanding

**How it works:**

```
Original: "The function returns a list of tokens."
Contextualized: "Document: LangChain API | Section: Text Splitting | The function returns a list of tokens."
→ The extra context can make isolated chunks easier to retrieve
```

**Trade-off:** extra indexing-time LLM calls; query-time cost stays similar. This repo does not include a measured quality delta.

### When should I use Fusion RAG? ✨

**Use Fusion RAG when:**

- Ranking quality is critical
- Research and literature review
- Complex multi-aspect queries
- Need to merge ranked lists from several query variants

**How it works:**

1. Generate 3-5 query perspectives
2. Retrieve documents for each query
3. Combine results using Reciprocal Rank Fusion (RRF)
4. Documents appearing in multiple result sets rank higher

**Trade-off:** more LLM calls than Simple RAG (several query variants + one generation). Quality is not measured in this repo.

### When should I use SQL RAG? ✨

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

**What the notebook shows:** generated SQL against the Chinook sample database, with a read-only SELECT guard. Accuracy depends on the schema and the generated SQL — there is no measured score here.

### When should I use GraphRAG? ✨

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
**Total**: 5-10 LLM calls → 20-40s latency

**Trade-off**: Slow but autonomous multi-step reasoning.

## Troubleshooting

### "ModuleNotFoundError: No module named 'langchain'"

**Solution:**

```bash
# Activate venv
source venv/bin/activate

# Verify
which python  # Should point to venv/bin/python

# Reinstall
pip install langchain
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more.

### "API key not found"

**Solutions:**

1. Create `.env` file in project root
2. Add: `OPENAI_API_KEY=sk-proj-...`
3. Restart Jupyter kernel
4. Verify: `cat .env`

### "Vector store not found"

**Solution:** Run notebook 02 first

```bash
jupyter notebook notebooks/fundamentals/02_embeddings_comparison.ipynb
```

This creates vector stores in `data/vector_stores/`.

### Notebook kernel keeps dying

**Solutions:**

1. Reduce RAM usage (lower k, smaller chunks)
2. Restart kernel between notebooks
3. Check system resources (htop/Activity Monitor)
4. Increase Jupyter memory limit

## Contributing

### How can I contribute?

**Ways to contribute:**

- 🐛 Report bugs
- ✨ Suggest features
- 📝 Improve documentation
- 💻 Submit pull requests
- 🎓 Share use cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Can I add a new architecture?

**Yes!** Follow this process:

1. Create notebook: `notebooks/advanced_architectures/17_your_architecture.ipynb`
2. Add prompts to `shared/prompts.py`
3. Update comparison in `11_comparison.ipynb`
4. Document in `notebooks/advanced_architectures/README.md`
5. Update CHANGELOG.md with your addition
6. Submit pull request

## Licensing

### Can I use this commercially?

**Yes!** MIT License allows:

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

**Requirements:**

- Include LICENSE file
- Attribute original authors

### Can I fork and modify?

**Absolutely!** MIT License encourages forks.

**Please:**

- Star the original repo
- Link back to original
- Share improvements (optional but appreciated)

## Getting Help

Still have questions?

1. 📖 **Check docs**: [docs/](.)
2. 🔍 **Search issues**: [GitHub Issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
3. 💬 **Ask community**: [Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)
4. 🐛 **Report bug**: [New Issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues/new)

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start
- [INSTALLATION.md](INSTALLATION.md) - Detailed setup
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [PERFORMANCE.md](PERFORMANCE.md) - What to measure (illustrative notes)
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
