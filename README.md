# LangChain RAG Tutorial

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/langchain-%3E%3D0.1.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)

A comprehensive, production-ready tutorial for building **Retrieval-Augmented Generation (RAG)** systems using LangChain.

**🎯 Features:** **12 advanced RAG architectures** | Multimodal RAG (images + text) | RAGAS evaluation | SQL & Graph support | Docker & production templates | Complete testing & CI/CD

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

## 📚 What You'll Learn

### Fundamentals (30-40 min)

Master the core concepts of RAG:

- **Document Loading & Splitting** - Process and chunk text efficiently
- **Embeddings Comparison** - OpenAI vs HuggingFace benchmarks
- **Simple RAG** - Build your first end-to-end RAG system

[📘 Start with Fundamentals →](notebooks/fundamentals/)

### Advanced Architectures (4-5 hours)

Explore **14 production-ready patterns**:

| Architecture              | Complexity | Use Case            | Key Feature                      |
| ------------------------- | ---------- | ------------------- | -------------------------------- |
| **Memory RAG**            | ⭐⭐         | Chatbots            | Conversation history             |
| **Branched RAG**          | ⭐⭐⭐        | Research            | Multi-query parallel retrieval   |
| **HyDe**                  | ⭐⭐⭐        | Ambiguous queries   | Hypothetical documents           |
| **Adaptive RAG**          | ⭐⭐⭐⭐       | Mixed workloads     | Intelligent query routing        |
| **Corrective RAG**        | ⭐⭐⭐⭐       | High accuracy       | Quality check + web fallback     |
| **Self-RAG**              | ⭐⭐⭐⭐⭐      | Self-correcting     | Autonomous refinement            |
| **Agentic RAG**           | ⭐⭐⭐⭐⭐      | Complex reasoning   | Multi-tool agent loops           |
| **Contextual RAG** ✨      | ⭐⭐⭐        | Technical docs      | Context-augmented chunks         |
| **Fusion RAG** ✨          | ⭐⭐⭐        | Best ranking        | Reciprocal Rank Fusion           |
| **SQL RAG** ✨             | ⭐⭐⭐⭐       | Analytics/BI        | Natural Language to SQL          |
| **GraphRAG** ✨            | ⭐⭐⭐⭐⭐      | Knowledge graphs    | Entity relationships + multi-hop |
| **Multimodal RAG** 🆕     | ⭐⭐⭐⭐       | Images + text       | GPT-4 Vision + OCR               |
| **Fine-tuning Guide** 🆕  | ⭐⭐⭐⭐       | Domain embeddings   | Custom embedding models          |
| **RAGAS Evaluation**      | -          | Quality metrics     | Comprehensive RAG assessment     |

[🔬 Explore Advanced Patterns →](notebooks/advanced_architectures/)

## 📖 Documentation

Comprehensive docs organized by topic:

- 🚀 **[Getting Started](docs/GETTING_STARTED.md)** - 5-minute quick start
- 🛠️ **[Installation](docs/INSTALLATION.md)** - Detailed setup guide
- 📚 **[API Reference](docs/API_REFERENCE.md)** - Shared module documentation
- 🏗️ **[Architecture](docs/ARCHITECTURE.md)** - Design decisions
- 🐛 **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues & solutions
- ⚡ **[Performance](docs/PERFORMANCE.md)** - Benchmarks & optimization
- ❓ **[FAQ](docs/FAQ.md)** - Frequently asked questions
- 🚀 **[Deployment](docs/DEPLOYMENT.md)** - Production deployment
- 📝 **[Examples](docs/EXAMPLES.md)** - Usage patterns
- 🤝 **[Contributing](docs/CONTRIBUTING.md)** - Contribution guidelines
- 📜 **[Changelog](docs/CHANGELOG.md)** - Version history

## 🏗️ Project Structure

```bash
llm_rag/
├── docs/                          # 📖 Modular documentation
│   ├── GETTING_STARTED.md        # Quick start guide
│   ├── INSTALLATION.md           # Setup instructions
│   ├── API_REFERENCE.md          # Shared module API
│   └── ... (8 more specialized docs)
├── notebooks/
│   ├── 00_index.ipynb            # 🎯 START HERE - Navigation hub
│   ├── fundamentals/             # Core RAG concepts (01-03)
│   │   ├── 01_setup_and_basics.ipynb
│   │   ├── 02_embeddings_comparison.ipynb
│   │   └── 03_simple_rag.ipynb
│   └── advanced_architectures/   # Advanced patterns (04-18)
│       ├── 04_rag_with_memory.ipynb
│       ├── 05_branched_rag.ipynb
│       ├── 06_hyde.ipynb
│       ├── 07_adaptive_rag.ipynb
│       ├── 08_corrective_rag.ipynb
│       ├── 09_self_rag.ipynb
│       ├── 10_agentic_rag.ipynb
│       ├── 11_comparison.ipynb           # All 12 architectures
│       ├── 12_contextual_rag.ipynb ✨     # v1.1.0
│       ├── 13_fusion_rag.ipynb ✨         # v1.1.0
│       ├── 14_sql_rag.ipynb ✨            # v1.1.0
│       ├── 15_graphrag.ipynb ✨           # v1.1.0
│       ├── 16_evaluation_ragas.ipynb ✨   # v1.1.0 - Quality metrics
│       ├── 17_multimodal_rag.ipynb 🆕    # v1.2.0 - Images + Text
│       └── 18_finetuning_embeddings.ipynb 🆕  # v1.2.0 - Custom embeddings
├── templates/                     # 🚀 Production deployment templates (NEW v1.2.0)
│   ├── fastapi/                  # REST API template
│   ├── streamlit/                # Web UI template
│   └── lambda/                   # AWS Lambda serverless
├── tests/                        # 🧪 Test suite (NEW v1.2.0)
│   ├── conftest.py              # pytest fixtures
│   ├── test_utils.py            # Utility tests
│   └── test_config.py           # Config tests
├── shared/                        # 🔧 Reusable utilities (1500+ lines)
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Utility functions
│   ├── loaders.py                # Document loading
│   └── prompts.py                # Prompt templates (30+ prompts)
├── data/                         # 💾 Vector stores, Chinook DB (gitignored)
├── Dockerfile                    # 🐳 Docker support (NEW v1.2.0)
├── docker-compose.yml            # 🐳 Full stack orchestration (NEW v1.2.0)
├── Makefile                      # 🛠️ Development commands (NEW v1.2.0)
├── pytest.ini                    # 🧪 Test configuration (NEW v1.2.0)
├── .pre-commit-config.yaml       # 🔍 Pre-commit hooks (NEW v1.2.0)
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies (NEW v1.2.0)
├── .env.example                  # 🔑 API key template
└── README.md                     # This file
```

## ✨ Key Features

**Core Capabilities:**

- ✅ **12 RAG Architectures** - From simple to graph-based
- ✅ **Multimodal RAG** 🆕 - GPT-4 Vision + OCR for images + text
- ✅ **Fine-tuning Guide** 🆕 - Train custom domain-specific embeddings
- ✅ **RAGAS Evaluation** - Comprehensive quality metrics
- ✅ **SQL & Graph Support** - Structured data + relationships
- ✅ **Modular Design** - Reusable shared utilities (DRY)
- ✅ **Vector Store Persistence** - No re-embedding needed
- ✅ **Comprehensive Benchmarks** - Performance & cost analysis

**Production-Ready Infrastructure:** 🆕

- ✅ **Docker Support** - Multi-stage builds with Redis, Prometheus, Grafana
- ✅ **3 Deployment Templates** - FastAPI, Streamlit, AWS Lambda
- ✅ **Testing Suite** - pytest with 70%+ coverage target
- ✅ **CI/CD Pipelines** - GitHub Actions (testing, linting, coverage)
- ✅ **Development Tools** - Makefile, pre-commit hooks, linting
- ✅ **Security Best Practices** - Non-root Docker, API key management

**Technical Stack:**

- **LangChain** v1.0+ - Framework & LCEL
- **OpenAI** GPT-4o-mini + GPT-4 Vision - Fast LLM + multimodal
- **FAISS** - Vector similarity search
- **NetworkX** - Graph algorithms
- **SQLAlchemy** - Database abstraction
- **RAGAS** - RAG evaluation framework
- **FastAPI** + **Streamlit** - Production deployment
- **Docker** + **Docker Compose** - Containerization
- **pytest** + **GitHub Actions** - Testing & CI/CD
- **Python** 3.9+ - Modern type hints

[🔍 See Architecture Details →](docs/ARCHITECTURE.md)

## 💡 Architecture Selection Guide

**Choose based on your needs:**

| Your Need                     | Architecture      | Docs                                                                                            |
| ----------------------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| 🚀 **Fast & simple**           | Simple RAG        | [03_simple_rag.ipynb](notebooks/fundamentals/03_simple_rag.ipynb)                               |
| 💬 **Chatbot with memory**     | Memory RAG        | [04_rag_with_memory.ipynb](notebooks/advanced_architectures/04_rag_with_memory.ipynb)           |
| 📚 **Research tool**           | Fusion RAG        | [13_fusion_rag.ipynb](notebooks/advanced_architectures/13_fusion_rag.ipynb) ✨                   |
| 🔍 **Ambiguous queries**       | Contextual RAG    | [12_contextual_rag.ipynb](notebooks/advanced_architectures/12_contextual_rag.ipynb) ✨           |
| ⚖️ **Cost optimization**       | Adaptive RAG      | [07_adaptive_rag.ipynb](notebooks/advanced_architectures/07_adaptive_rag.ipynb)                 |
| 🎯 **High accuracy**           | Fusion / CRAG     | [13_fusion_rag.ipynb](notebooks/advanced_architectures/13_fusion_rag.ipynb) ✨                   |
| 🔄 **Self-correcting**         | Self-RAG          | [09_self_rag.ipynb](notebooks/advanced_architectures/09_self_rag.ipynb)                         |
| 🤖 **Complex reasoning**       | Agentic RAG       | [10_agentic_rag.ipynb](notebooks/advanced_architectures/10_agentic_rag.ipynb)                   |
| 📊 **Analytics/BI**            | SQL RAG           | [14_sql_rag.ipynb](notebooks/advanced_architectures/14_sql_rag.ipynb) ✨                         |
| 🕸️ **Knowledge graphs**        | GraphRAG          | [15_graphrag.ipynb](notebooks/advanced_architectures/15_graphrag.ipynb) ✨                       |
| 🖼️ **Images + text** 🆕        | Multimodal RAG    | [17_multimodal_rag.ipynb](notebooks/advanced_architectures/17_multimodal_rag.ipynb) 🆕          |
| 🎯 **Custom embeddings** 🆕    | Fine-tuning Guide | [18_finetuning_embeddings.ipynb](notebooks/advanced_architectures/18_finetuning_embeddings.ipynb) 🆕 |
| 📈 **Quality evaluation**      | RAGAS             | [16_evaluation_ragas.ipynb](notebooks/advanced_architectures/16_evaluation_ragas.ipynb) ✨       |

**Rule of thumb:** Start with Simple RAG → Add Contextual for quality → Use specialized for specific needs.

[❓ Need help choosing? See FAQ →](docs/FAQ.md#which-architecture-should-i-choose)

## 🚀 Production Deployment 🆕

Ready to deploy? Choose from **3 production-ready templates**:

### 🐳 Docker (Recommended)

```bash
# Quick start with Docker Compose
docker-compose up -d

# Or build custom image
docker build -t langchain-rag:latest .
docker run -p 8000:8000 --env-file .env langchain-rag:latest
```

**Features:** Multi-stage builds, Redis caching, Prometheus + Grafana monitoring, non-root security

### 🚂 FastAPI REST API

Complete REST API with automatic documentation:

```bash
cd templates/fastapi
pip install -r requirements.txt
uvicorn app:app --reload
# Visit http://localhost:8000/docs for Swagger UI
```

**Features:** Pydantic validation, CORS, health checks, error handling, async support

### 🎨 Streamlit Web UI

Interactive web application:

```bash
cd templates/streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Features:** Real-time queries, source document display, architecture selection, metrics visualization

### ⚡ AWS Lambda (Serverless)

Deploy to AWS Lambda:

```bash
cd templates/lambda
zip -r function.zip .
aws lambda create-function --function-name rag-api --zip-file fileb://function.zip
```

**Features:** S3 vector store integration, cold start optimization, API Gateway ready

**📖 Full guide:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## 📊 Performance at a Glance

| Architecture     | Latency  | Cost/Query | Accuracy    | Best For        |
| ---------------- | -------- | ---------- | ----------- | --------------- |
| Simple RAG       | ~2s      | $0.002     | Good        | General Q&A     |
| Contextual RAG ✨ | ~2-3s    | $0.002     | Very Good   | Technical docs  |
| Fusion RAG ✨     | ~5-8s    | $0.006     | Excellent   | Research        |
| SQL RAG ✨        | ~2-5s    | $0.004     | Perfect*    | Analytics       |
| GraphRAG ✨       | ~3-8s    | $0.010+    | Excellent** | Relationships   |
| Adaptive RAG     | Variable | $0.003     | Very Good   | Mixed workloads |
| Agentic RAG      | ~30s     | $0.012     | Excellent   | Complex tasks   |

*For structured data | **For multi-hop queries

**Full benchmarks:** [11_comparison.ipynb](notebooks/advanced_architectures/11_comparison.ipynb) | [RAGAS Evaluation](notebooks/advanced_architectures/16_evaluation_ragas.ipynb)

## 🚦 Prerequisites

- **Python** 3.9+ (3.10+ recommended)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **~2GB RAM** (4GB+ recommended)
- **~1.5GB disk space** (dependencies + models)

[📖 Detailed requirements →](docs/INSTALLATION.md#prerequisites)

## 🎓 Learning Path

**Recommended sequence:**

1. **Setup** (10 min): [GETTING_STARTED.md](docs/GETTING_STARTED.md)
2. **Navigation Hub** (5 min): [00_index.ipynb](notebooks/00_index.ipynb)
3. **Fundamentals** (30-40 min): [Notebooks 01-03](notebooks/fundamentals/)
4. **Choose Your Path**:
   - 🏃 **Fast track**: Simple RAG → Contextual RAG → Your use case
   - 🔬 **Deep dive**: Complete all 12 architectures
   - 📊 **Comparison**: Jump to [11_comparison.ipynb](notebooks/advanced_architectures/11_comparison.ipynb)
   - 📈 **Evaluation**: Try [16_evaluation_ragas.ipynb](notebooks/advanced_architectures/16_evaluation_ragas.ipynb)

**Total time:**

- Fast track: ~1-2 hours
- Complete tutorial: ~5-7 hours
- With multimodal + evaluation: ~7-9 hours
- Production deployment: +1-2 hours

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Ways to contribute:**

- 🐛 Report bugs
- ✨ Suggest features
- 📝 Improve documentation
- 💻 Submit pull requests

## 🛠️ Development 🆕

**Quick commands with Makefile:**

```bash
make install      # Install all dependencies
make test         # Run test suite with coverage
make lint         # Run code quality checks
make format       # Auto-format code (black, isort)
make docker-build # Build Docker image
make docker-run   # Run full stack with Docker Compose
make clean        # Clean cache and build files
```

**Testing:**

```bash
# Run all tests with coverage
pytest tests/ -v --cov=shared --cov-report=html

# Run specific test file
pytest tests/test_utils.py -v

# Run with markers
pytest -m "not slow"  # Skip slow tests
```

**Pre-commit hooks:**

```bash
# Install hooks (runs on every commit)
pre-commit install

# Run manually
pre-commit run --all-files
```

**CI/CD:** Automated testing and linting on every push via GitHub Actions (Python 3.9, 3.10, 3.11)

**📖 More details:** [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

**TL;DR:** Free to use commercially, modify, and distribute. Just include the license.

## 🔗 Resources

- 📖 **Documentation**: [docs/](docs/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)
- 🌐 **LangChain Docs**: [python.langchain.com](https://python.langchain.com/)

## 💬 Getting Help

- 📖 Check [FAQ](docs/FAQ.md) first
- 🔍 Search [existing issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- 🐛 [Open a new issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues/new)
- 💬 Ask in [Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)

---

**⭐ If this helps you, please star the repo!**

**Latest:** v1.2.0 - Multimodal RAG, Docker, Production Templates | Made with ❤️ using Claude Code | [View Changelog](docs/CHANGELOG.md)
