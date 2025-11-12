# Getting Started

Get up and running with LangChain RAG Tutorial in **5 minutes**.

## Prerequisites

Before you begin, ensure you have:

- Python 3.9+ installed
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- ~2GB RAM available
- Internet connection for dependencies

## Quick Start

### 1. Clone and Navigate

```bash
cd llm_rag
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

- LangChain and extensions
- OpenAI SDK
- FAISS for vector storage
- Jupyter notebook support
- HuggingFace transformers (local embeddings)
- NetworkX (graph algorithms) ✨
- SQLAlchemy (SQL RAG) ✨
- RAGAS (evaluation framework) ✨
- Spacy (entity extraction) ✨

### 4. Configure API Key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=sk-proj-your-key-here" > .env
```

Or copy from template:

```bash
cp .env.example .env
# Edit .env with your actual API key
```

### 5. Launch Jupyter

```bash
jupyter notebook
```

Then navigate to `notebooks/00_index.ipynb` to start!

## Learning Path

Follow this recommended sequence:

### **Step 1: Navigation Hub** (2 minutes)

- 📍 Start here: [notebooks/00_index.ipynb](../notebooks/00_index.ipynb)
- Overview of all notebooks
- Environment validation
- Architecture comparison

### **Step 2: Fundamentals** (30-40 minutes)

Complete these in order:

1. [01_setup_and_basics.ipynb](../notebooks/fundamentals/01_setup_and_basics.ipynb) - Document loading & splitting
2. [02_embeddings_comparison.ipynb](../notebooks/fundamentals/02_embeddings_comparison.ipynb) - OpenAI vs HuggingFace
3. [03_simple_rag.ipynb](../notebooks/fundamentals/03_simple_rag.ipynb) - Your first RAG chain

### **Step 3: Advanced Architectures** (Pick based on use case)

Explore **12 advanced patterns**:

**Foundation (⭐⭐-⭐⭐⭐):**
- **Chatbots?** → [04_rag_with_memory.ipynb](../notebooks/advanced_architectures/04_rag_with_memory.ipynb)
- **Research tool?** → [05_branched_rag.ipynb](../notebooks/advanced_architectures/05_branched_rag.ipynb)
- **Ambiguous queries?** → [06_hyde.ipynb](../notebooks/advanced_architectures/06_hyde.ipynb)

**Enhanced (⭐⭐⭐⭐):**
- **Mixed workload?** → [07_adaptive_rag.ipynb](../notebooks/advanced_architectures/07_adaptive_rag.ipynb)
- **High accuracy?** → [08_corrective_rag.ipynb](../notebooks/advanced_architectures/08_corrective_rag.ipynb)

**Self-Improving (⭐⭐⭐⭐⭐):**
- **Self-correcting?** → [09_self_rag.ipynb](../notebooks/advanced_architectures/09_self_rag.ipynb)
- **Complex reasoning?** → [10_agentic_rag.ipynb](../notebooks/advanced_architectures/10_agentic_rag.ipynb)

**Cutting-Edge ✨ (⭐⭐⭐-⭐⭐⭐⭐⭐):**
- **Technical docs?** → [12_contextual_rag.ipynb](../notebooks/advanced_architectures/12_contextual_rag.ipynb) - Context-augmented chunks
- **Best ranking?** → [13_fusion_rag.ipynb](../notebooks/advanced_architectures/13_fusion_rag.ipynb) - Reciprocal Rank Fusion
- **Analytics/BI?** → [14_sql_rag.ipynb](../notebooks/advanced_architectures/14_sql_rag.ipynb) - Natural Language to SQL
- **Knowledge graphs?** → [15_graphrag.ipynb](../notebooks/advanced_architectures/15_graphrag.ipynb) - Graph-based reasoning

**Analysis:**
- **Benchmark all 12?** → [11_comparison.ipynb](../notebooks/advanced_architectures/11_comparison.ipynb)
- **Quality metrics?** → [16_evaluation_ragas.ipynb](../notebooks/advanced_architectures/16_evaluation_ragas.ipynb) ✨

## First Run Checklist

Before running notebooks, verify:

- [ ] Virtual environment activated (`which python` should point to `venv/`)
- [ ] Dependencies installed (`pip list | grep langchain`)
- [ ] API key configured (`.env` file exists)
- [ ] Jupyter running (`jupyter notebook`)
- [ ] Started with `00_index.ipynb`

## Quick Troubleshooting

**Import errors?**

```bash
pip install --upgrade -r requirements.txt
```

**API key errors?**

- Verify `.env` file exists in project root
- Check key format: `OPENAI_API_KEY=sk-proj-...`
- Ensure no quotes around the key in `.env`

**Jupyter kernel issues?**

```bash
python -m ipykernel install --user --name=venv
```

For more troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Next Steps

- 📖 Detailed setup instructions: [INSTALLATION.md](INSTALLATION.md)
- 🏗️ Understand the architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- 📚 Explore shared module API: [API_REFERENCE.md](API_REFERENCE.md)
- ❓ Common questions: [FAQ.md](FAQ.md)

## Need Help?

- 🐛 Found a bug? [Open an issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- 💬 Questions? Check [FAQ.md](FAQ.md)
- 🤝 Want to contribute? See [CONTRIBUTING.md](CONTRIBUTING.md)
