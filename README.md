# LangChain RAG Tutorial

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/langchain-%3E%3D0.1.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)

Hands-on tutorial for building **Retrieval-Augmented Generation (RAG)** systems with LangChain. It walks through patterns you would typically need before production — evaluation, monitoring, and starter deployment templates — only to the extent the notebooks and templates in this repo actually cover them.

This is a learning repo, not a finished product you can drop into production.

## Status

Pinned dependencies target the LangChain 0.1+ / early 1.x APIs used when these notebooks were last updated (changelog: 2025-11-13). Several notebooks import current LangChain surfaces (`langchain_openai`, `langchain_core`, `create_agent`) and also older helpers (for example `ConversationBufferMemory` in sample text). Full notebook execution was **not** re-verified in this docs-only pass and needs API keys plus live model calls. Treat the notebooks as educational walkthroughs; expect import or API drift if you install today's latest LangChain without pinning.

Shared-module unit tests (`pytest tests/ -m "not integration"`) do not require API keys.

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
cd langchain-rag-tutorial

# Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure API key
echo "OPENAI_API_KEY=sk-proj-your-key-here" > .env

# Start learning
jupyter notebook notebooks/00_index.ipynb
```

**📖 Full guide:** [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)

## 📚 What the notebooks cover

### Fundamentals (notebooks 01–03)

- **Document loading & splitting** — load web docs and compare chunking strategies
- **Embeddings comparison** — OpenAI vs HuggingFace embeddings and FAISS persistence
- **Simple RAG** — similarity vs MMR retrieval and an LCEL RAG chain

[📘 Fundamentals →](notebooks/fundamentals/)

### Advanced patterns (notebooks 04–18)

Tutorial implementations of these patterns (not hardened services):

| Notebook | Pattern | What it demonstrates |
| -------- | ------- | -------------------- |
| **04** | Memory RAG | Conversation history with `RunnableWithMessageHistory` |
| **05** | Branched RAG | Multi-query / parallel retrieval |
| **06** | HyDe | Hypothetical document embeddings |
| **07** | Adaptive RAG | Route simple / medium / complex queries |
| **08** | Corrective RAG | Relevance grading + web-search fallback |
| **09** | Self-RAG | Self-critique / refinement loop |
| **10** | Agentic RAG | Tool-using agent loop |
| **11** | Comparison | Qualitative trade-off tables (not a measured harness) |
| **12** | Contextual RAG | Context-augmented chunks |
| **13** | Fusion RAG | Reciprocal Rank Fusion |
| **14** | SQL RAG | Natural language → SQL on the Chinook sample DB |
| **15** | GraphRAG | Entity/relationship extraction + NetworkX |
| **16** | RAGAS evaluation | Faithfulness, relevancy, context precision/recall, correctness |
| **17** | Multimodal RAG | GPT-4 Vision + Tesseract OCR on images + text |
| **18** | Embedding fine-tuning | Contrastive fine-tune of a sentence-transformers model |

[🔬 Advanced notebooks →](notebooks/advanced_architectures/)

## 📖 Documentation

- 🚀 **[Getting Started](docs/GETTING_STARTED.md)** — setup and learning path
- 🛠️ **[Installation](docs/INSTALLATION.md)** — detailed setup
- 📚 **[API Reference](docs/API_REFERENCE.md)** — `shared/` helpers
- 🏗️ **[Architecture](docs/ARCHITECTURE.md)** — how the tutorial is structured
- 🐛 **[Troubleshooting](docs/TROUBLESHOOTING.md)** — common issues
- ⚡ **[Performance](docs/PERFORMANCE.md)** — what to measure; illustrative cost/latency notes
- ❓ **[FAQ](docs/FAQ.md)** — frequently asked questions
- 🚀 **[Deployment](docs/DEPLOYMENT.md)** — starter templates and checklists
- 📝 **[Examples](docs/EXAMPLES.md)** — usage snippets
- 🤝 **[Contributing](docs/CONTRIBUTING.md)** — contribution guidelines
- 📜 **[Changelog](docs/CHANGELOG.md)** — version history

## 🏗️ Project Structure

```text
langchain-rag-tutorial/
├── docs/                          # Modular documentation
├── notebooks/
│   ├── 00_index.ipynb            # Navigation hub
│   ├── fundamentals/             # Core RAG (01-03)
│   └── advanced_architectures/   # Patterns 04-18
├── templates/                     # Starter apps (FastAPI, Streamlit, Lambda)
├── tests/                         # pytest for shared/ (plus optional integration)
├── shared/                        # Config, loaders, prompts, utilities
├── data/                          # Vector stores, Chinook DB (gitignored)
├── monitoring/                    # Example Prometheus/Grafana config
├── Dockerfile
├── docker-compose.yml             # App + optional Redis/Prometheus/Grafana
├── Makefile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## What this repo includes

**Notebooks**

- Fundamentals through GraphRAG, multimodal RAG, embedding fine-tuning, and RAGAS
- Reusable helpers in `shared/` (config, loaders, prompts, FAISS save/load)
- Vector-store persistence so later notebooks can reuse embeddings from notebook 02

**Starter infrastructure (educational)**

- Docker Compose stack (app, optional Redis, Prometheus, Grafana)
- FastAPI, Streamlit, and AWS Lambda starter templates
- pytest for `shared/` plus GitHub Actions workflows already in the repo
- Makefile and pre-commit config

**Stack used in the notebooks**

- LangChain (see [Status](#status) and `requirements.txt`)
- OpenAI chat + embeddings (GPT-4o-mini / GPT-4o vision in notebook 17)
- FAISS, sentence-transformers, NetworkX, SQLAlchemy, RAGAS
- Tesseract OCR + Pillow (notebook 17)
- FastAPI / Streamlit / Docker in the templates

[🔍 Architecture notes →](docs/ARCHITECTURE.md)

## 💡 Choosing a notebook

| If you want to try… | Open |
| ------------------- | ---- |
| Fast, simple Q&A | [03_simple_rag.ipynb](notebooks/fundamentals/03_simple_rag.ipynb) |
| Chat with follow-up questions | [04_rag_with_memory.ipynb](notebooks/advanced_architectures/04_rag_with_memory.ipynb) |
| Multi-query retrieval | [05_branched_rag.ipynb](notebooks/advanced_architectures/05_branched_rag.ipynb) |
| Ambiguous queries (HyDe) | [06_hyde.ipynb](notebooks/advanced_architectures/06_hyde.ipynb) |
| Query routing | [07_adaptive_rag.ipynb](notebooks/advanced_architectures/07_adaptive_rag.ipynb) |
| Relevance grading + web fallback | [08_corrective_rag.ipynb](notebooks/advanced_architectures/08_corrective_rag.ipynb) |
| Self-critique loop | [09_self_rag.ipynb](notebooks/advanced_architectures/09_self_rag.ipynb) |
| Tool-using agent | [10_agentic_rag.ipynb](notebooks/advanced_architectures/10_agentic_rag.ipynb) |
| Context-augmented chunks | [12_contextual_rag.ipynb](notebooks/advanced_architectures/12_contextual_rag.ipynb) |
| Reciprocal Rank Fusion | [13_fusion_rag.ipynb](notebooks/advanced_architectures/13_fusion_rag.ipynb) |
| Natural language → SQL | [14_sql_rag.ipynb](notebooks/advanced_architectures/14_sql_rag.ipynb) |
| Knowledge-graph style retrieval | [15_graphrag.ipynb](notebooks/advanced_architectures/15_graphrag.ipynb) |
| Images + text | [17_multimodal_rag.ipynb](notebooks/advanced_architectures/17_multimodal_rag.ipynb) |
| Fine-tune a local embedding model | [18_finetuning_embeddings.ipynb](notebooks/advanced_architectures/18_finetuning_embeddings.ipynb) |
| RAGAS metrics | [16_evaluation_ragas.ipynb](notebooks/advanced_architectures/16_evaluation_ragas.ipynb) |

Start with Simple RAG, then pick a pattern that matches the question you want to explore. Notebook 11 is a qualitative comparison, not a reproducible benchmark.

[❓ FAQ: which architecture? →](docs/FAQ.md#which-architecture-should-i-choose)

## 🚀 Starter deployment templates

These are teaching templates (auth, tenancy, SLOs, and load testing are out of scope).

### Docker Compose

```bash
docker-compose up -d

# Or build a single image
docker build -t langchain-rag:latest .
docker run -p 8000:8000 --env-file .env langchain-rag:latest
```

Compose can also start Redis plus example Prometheus/Grafana configs under `monitoring/`.

### FastAPI starter

```bash
cd templates/fastapi
pip install -r requirements.txt
uvicorn app:app --reload
# http://localhost:8000/docs
```

### Streamlit starter

```bash
cd templates/streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### AWS Lambda starter

```bash
cd templates/lambda
zip -r function.zip .
# See templates/lambda/README.md for IAM, layers, and API Gateway steps
```

**📖 Notes and checklists:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## 🚦 Prerequisites

- **Python** 3.9+ (3.10+ recommended)
- **OpenAI API key** ([create one](https://platform.openai.com/api-keys))
- **~2GB RAM** (more if you run notebook 18 fine-tuning)
- **~2GB disk** for dependencies and local models
- **System packages** for notebook 17: Tesseract OCR, Poppler (see [INSTALLATION.md](docs/INSTALLATION.md))

[📖 Detailed requirements →](docs/INSTALLATION.md#prerequisites)

## 🎓 Learning path

1. **Setup:** [GETTING_STARTED.md](docs/GETTING_STARTED.md)
2. **Hub:** [00_index.ipynb](notebooks/00_index.ipynb)
3. **Fundamentals:** [notebooks 01–03](notebooks/fundamentals/)
4. Then either:
   - Simple RAG → one specialized notebook for your use case
   - Work through 04–16 in order
   - Skim [11_comparison.ipynb](notebooks/advanced_architectures/11_comparison.ipynb) then jump
   - Run [16_evaluation_ragas.ipynb](notebooks/advanced_architectures/16_evaluation_ragas.ipynb) after you have a chain

Times in the docs are rough teaching estimates, not measured wall-clock benchmarks.

## 🤝 Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## 🛠️ Development

```bash
make install      # Install dependencies
make test         # pytest with coverage
make lint         # black, flake8, isort, mypy (as configured)
make format       # Auto-format
make docker-build
make docker-run
make clean
```

```bash
pytest tests/ -v --cov=shared --cov-report=html
pytest tests/test_utils.py -v
pytest -m "not slow and not integration"  # no API key
```

```bash
pre-commit install
pre-commit run --all-files
```

**📖 More:** [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)

## 📜 License

MIT License — see [LICENSE](LICENSE).

## 🔗 Resources

- 📖 **Docs:** [docs/](docs/)
- 🐛 **Issues:** [GitHub Issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)
- 🌐 **LangChain:** [python.langchain.com](https://python.langchain.com/)

## 💬 Getting help

- [FAQ](docs/FAQ.md)
- [Existing issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- [New issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues/new)
- [Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)

---

Latest tagged notes: v1.2.1 — import and Python 3.9 fixes (see [Changelog](docs/CHANGELOG.md)).
