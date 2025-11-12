# RAG Fundamentals

This directory contains foundational notebooks for understanding and building basic RAG systems with LangChain.

## Notebooks

### 01_setup_and_basics.ipynb

**Topics Covered:**

- Environment setup and API key configuration
- Document loading from web sources
- Text splitting strategies and comparison
- Metadata management

**Prerequisites:** None (start here!)

**Outputs:**

- Loaded documents from LangChain documentation
- Document chunks with configurable strategies

**Duration:** ~10 minutes

---

### 02_embeddings_comparison.ipynb

**Topics Covered:**

- OpenAI embeddings (text-embedding-3-small)
- HuggingFace embeddings (sentence-transformers)
- FAISS vector store creation
- Vector store persistence (save/load)
- Performance comparison

**Prerequisites:**

- 01_setup_and_basics.ipynb (for documents and chunks)
- OpenAI API key configured

**Outputs:**

- Vector stores saved to `data/vector_stores/`
- Performance metrics for both embedding types

**Duration:** ~5-8 minutes

---

### 03_simple_rag.ipynb

**Topics Covered:**

- Retrieval strategies (Similarity vs MMR)
- RAG chain construction with LCEL
- Complete end-to-end RAG workflow
- Query evaluation and comparison
- Best practices and optimization tips

**Prerequisites:**

- 01_setup_and_basics.ipynb
- 02_embeddings_comparison.ipynb
- Vector stores created

**Outputs:**

- Working RAG chains for similarity and MMR retrieval
- Baseline performance metrics for comparison with advanced architectures

**Duration:** ~10-15 minutes

---

## Learning Path

Follow the notebooks in order:

```
01_setup_and_basics.ipynb
         ↓
02_embeddings_comparison.ipynb
         ↓
03_simple_rag.ipynb
```

After completing these fundamentals, proceed to the **advanced_architectures** directory to explore more sophisticated RAG patterns.

## Key Concepts

- **Document Loading**: WebBaseLoader for web content
- **Text Splitting**: RecursiveCharacterTextSplitter with chunk_size and overlap
- **Embeddings**: Converting text to numerical vectors
  - OpenAI: High quality, API-based, paid
  - HuggingFace: Local, free, privacy-friendly
- **Vector Stores**: FAISS for efficient similarity search
- **Retrievers**: Similarity search vs MMR (Maximal Marginal Relevance)
- **RAG Chains**: Combining retrieval + LLM generation with LCEL

## Shared Utilities

All notebooks use the `shared` module for common functions:

```python
from shared import (
    load_langchain_docs,      # Document loading
    split_documents,          # Text splitting
    load_vector_store,        # Load saved vector stores
    save_vector_store,        # Save vector stores
    RAG_PROMPT_TEMPLATE,      # Standard RAG prompt
)
```

This ensures code reusability and consistency across notebooks.

## Outputs Created

After running these notebooks, you'll have:

1. **Vector Stores** (saved in `data/vector_stores/`):
   - `openai_embeddings/` - FAISS index with OpenAI embeddings
   - `huggingface_embeddings/` - FAISS index with HuggingFace embeddings

2. **Performance Baselines**:
   - Embedding generation times
   - Vector store creation times
   - Query response times
   - Retrieval quality metrics

These artifacts are reused in advanced architecture notebooks to avoid redundant computation.

## 📖 Documentation

For detailed guidance, see:

- 🚀 **[Getting Started](../../docs/GETTING_STARTED.md)** - Quick start guide
- 🛠️ **[Installation](../../docs/INSTALLATION.md)** - Detailed setup
- 📚 **[API Reference](../../docs/API_REFERENCE.md)** - Shared module docs
- 🐛 **[Troubleshooting](../../docs/TROUBLESHOOTING.md)** - Common issues
- ❓ **[FAQ](../../docs/FAQ.md)** - Frequently asked questions

## Next Steps

Once you've completed the fundamentals:

- Explore **Memory RAG** for conversational interactions
- Try **Branched RAG** for multi-intent queries
- Implement **HyDe** for ambiguous queries
- Build **Adaptive RAG** for intelligent query routing
- Apply **CRAG** for high-accuracy requirements
- Experiment with **Self-RAG** for self-reflective systems
- Create **Agentic RAG** for autonomous reasoning

See `../advanced_architectures/README.md` for details.
