# Getting Started

Get the tutorial running in about five minutes. For platform-specific details, optional setups and
the full configuration reference, see [INSTALLATION.md](INSTALLATION.md).

## Prerequisites

- Python 3.10-3.13
- OpenAI API key ([create one](https://platform.openai.com/api-keys))
- About 2 GB of free RAM and disk space
- Internet connection for dependencies and API calls

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
cd langchain-rag-tutorial
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs LangChain 1.x (with `langchain-openai`, `langchain-community`,
`langchain-huggingface`, `langchain-text-splitters`, `langchain-tavily` and `langgraph`), FAISS,
sentence-transformers, NetworkX, RAGAS, Jupyter and the deployment template dependencies. See
[INSTALLATION.md](INSTALLATION.md#dependencies) for the full list.

### 4. Configure API Keys

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (and TAVILY_API_KEY for notebooks 08 and 10)
```

### 5. Launch Jupyter

```bash
jupyter notebook notebooks/00_index.ipynb
```

## Learning Path

### Step 1: Navigation Hub (2 minutes)

[00_index.ipynb](../notebooks/00_index.ipynb) gives an overview of all notebooks and validates the
environment.

### Step 2: Fundamentals (30-40 minutes)

Complete these in order:

1. [01_setup_and_basics.ipynb](../notebooks/fundamentals/01_setup_and_basics.ipynb) - Document
   loading and splitting
2. [02_embeddings_comparison.ipynb](../notebooks/fundamentals/02_embeddings_comparison.ipynb) -
   OpenAI vs HuggingFace embeddings; creates the vector stores used by later notebooks
3. [03_simple_rag.ipynb](../notebooks/fundamentals/03_simple_rag.ipynb) - Your first RAG chain

Alternatively, build the vector stores up front with `make vector-stores`
(see [INSTALLATION.md](INSTALLATION.md#pre-building-vector-stores)).

### Step 3: Advanced Architectures (pick by use case)

| Complexity | Question                    | Notebook                                                                                                           |
| ---------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 2/5        | Chatbot with memory?        | [04_rag_with_memory.ipynb](../notebooks/advanced_architectures/04_rag_with_memory.ipynb)                           |
| 3/5        | Research coverage?          | [05_branched_rag.ipynb](../notebooks/advanced_architectures/05_branched_rag.ipynb)                                 |
| 3/5        | Ambiguous queries?          | [06_hyde.ipynb](../notebooks/advanced_architectures/06_hyde.ipynb)                                                 |
| 4/5        | Mixed workload?             | [07_adaptive_rag.ipynb](../notebooks/advanced_architectures/07_adaptive_rag.ipynb)                                 |
| 4/5        | High accuracy?              | [08_corrective_rag.ipynb](../notebooks/advanced_architectures/08_corrective_rag.ipynb)                             |
| 5/5        | Self-correcting?            | [09_self_rag.ipynb](../notebooks/advanced_architectures/09_self_rag.ipynb)                                         |
| 5/5        | Complex reasoning?          | [10_agentic_rag.ipynb](../notebooks/advanced_architectures/10_agentic_rag.ipynb)                                   |
| 3/5        | Technical docs?             | [12_contextual_rag.ipynb](../notebooks/advanced_architectures/12_contextual_rag.ipynb)                             |
| 3/5        | Best ranking?               | [13_fusion_rag.ipynb](../notebooks/advanced_architectures/13_fusion_rag.ipynb)                                     |
| 4/5        | Analytics/BI?               | [14_sql_rag.ipynb](../notebooks/advanced_architectures/14_sql_rag.ipynb)                                           |
| 5/5        | Knowledge graphs?           | [15_graphrag.ipynb](../notebooks/advanced_architectures/15_graphrag.ipynb)                                         |
| 4/5        | Images + text?              | [17_multimodal_rag.ipynb](../notebooks/advanced_architectures/17_multimodal_rag.ipynb)                             |
| 4/5        | Domain-specific embeddings? | [18_finetuning_embeddings.ipynb](../notebooks/advanced_architectures/18_finetuning_embeddings.ipynb)               |
| 3/5        | Identifiers and jargon?     | [19_hybrid_search_reranking.ipynb](../notebooks/advanced_architectures/19_hybrid_search_reranking.ipynb)           |
| 3/5        | Chunk-size trade-offs?      | [20_parent_multivector_retrieval.ipynb](../notebooks/advanced_architectures/20_parent_multivector_retrieval.ipynb) |

For a qualitative overview, [11_comparison.ipynb](../notebooks/advanced_architectures/11_comparison.ipynb)
prints trade-off tables (not a measured harness).
[16_evaluation_ragas.ipynb](../notebooks/advanced_architectures/16_evaluation_ragas.ipynb) measures
quality with RAGAS when you have a chain and an API key.

Suggested tracks:

- **Fast track** (1-2 hours): Simple RAG, then Contextual RAG, then your use case
- **Complete tutorial** (5-7 hours): all notebooks in order
- **With multimodal and evaluation**: add 2 hours
- **Starter templates**: add 1-2 hours (see [DEPLOYMENT.md](DEPLOYMENT.md))

## First Run Checklist

- [ ] Virtual environment activated (`which python` points to `venv/`)
- [ ] Dependencies installed (`pip list | grep langchain`)
- [ ] `.env` exists in the project root with `OPENAI_API_KEY` set
- [ ] Started with `00_index.ipynb`

If something fails, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Next Steps

- [ARCHITECTURE.md](ARCHITECTURE.md) - How the architectures work
- [API_REFERENCE.md](API_REFERENCE.md) - The `shared` module
- [FAQ.md](FAQ.md) - Common questions
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contributing
