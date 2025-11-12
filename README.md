# LangChain RAG Tutorial

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/langchain-%3E%3D0.1.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)

A comprehensive, production-ready tutorial for building **Retrieval-Augmented Generation (RAG)** systems using LangChain.

**🎯 Features:** 8 advanced RAG architectures | Modular design | Complete documentation | Best practices

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

### Advanced Architectures (2-3 hours)
Explore 8 production-ready patterns:

| Architecture | Complexity | Use Case | Key Feature |
|--------------|------------|----------|-------------|
| **Memory RAG** | ⭐⭐ | Chatbots | Conversation history |
| **Branched RAG** | ⭐⭐⭐ | Research | Multi-query parallel retrieval |
| **HyDe** | ⭐⭐⭐ | Ambiguous queries | Hypothetical documents |
| **Adaptive RAG** | ⭐⭐⭐⭐ | Mixed workloads | Intelligent query routing |
| **Corrective RAG** | ⭐⭐⭐⭐ | High accuracy | Quality check + web fallback |
| **Self-RAG** | ⭐⭐⭐⭐⭐ | Self-correcting | Autonomous refinement |
| **Agentic RAG** | ⭐⭐⭐⭐⭐ | Complex reasoning | Multi-tool agent loops |
| **Comparison** | - | Benchmarking | Full performance analysis |

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

```
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
│   └── advanced_architectures/   # Advanced patterns (04-11)
│       ├── 04_rag_with_memory.ipynb
│       ├── 05_branched_rag.ipynb
│       ├── 06_hyde.ipynb
│       ├── 07_adaptive_rag.ipynb
│       ├── 08_corrective_rag.ipynb
│       ├── 09_self_rag.ipynb
│       ├── 10_agentic_rag.ipynb
│       └── 11_comparison.ipynb
├── shared/                        # 🔧 Reusable utilities (983 lines)
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Utility functions
│   ├── loaders.py                # Document loading
│   └── prompts.py                # Prompt templates (13 prompts)
├── data/                         # 💾 Vector stores & cache (gitignored)
├── .env.example                  # 🔑 API key template
└── README.md                     # This file
```

## ✨ Key Features

**Core Capabilities:**
- ✅ **8 RAG Architectures** - From simple to agentic
- ✅ **Modular Design** - Reusable shared utilities (DRY)
- ✅ **Vector Store Persistence** - No re-embedding needed
- ✅ **Comprehensive Benchmarks** - Performance & cost analysis
- ✅ **Production-Ready** - Error handling, monitoring, security

**Technical Stack:**
- **LangChain** v0.1.0+ - Framework & LCEL
- **OpenAI** GPT-4o-mini - Fast, cost-effective LLM
- **FAISS** - Facebook AI similarity search
- **HuggingFace** - Free local embeddings
- **Python** 3.9+ - Modern type hints

[🔍 See Architecture Details →](docs/ARCHITECTURE.md)

## 💡 Architecture Selection Guide

**Choose based on your needs:**

| Your Need | Architecture | Docs |
|-----------|--------------|------|
| 🚀 **Fast & simple** | Simple RAG | [03_simple_rag.ipynb](notebooks/fundamentals/03_simple_rag.ipynb) |
| 💬 **Chatbot with memory** | Memory RAG | [04_rag_with_memory.ipynb](notebooks/advanced_architectures/04_rag_with_memory.ipynb) |
| 📚 **Research tool** | Branched RAG | [05_branched_rag.ipynb](notebooks/advanced_architectures/05_branched_rag.ipynb) |
| 🔍 **Ambiguous queries** | HyDe | [06_hyde.ipynb](notebooks/advanced_architectures/06_hyde.ipynb) |
| ⚖️ **Cost optimization** | Adaptive RAG | [07_adaptive_rag.ipynb](notebooks/advanced_architectures/07_adaptive_rag.ipynb) |
| 🎯 **High accuracy** | Corrective RAG | [08_corrective_rag.ipynb](notebooks/advanced_architectures/08_corrective_rag.ipynb) |
| 🔄 **Self-correcting** | Self-RAG | [09_self_rag.ipynb](notebooks/advanced_architectures/09_self_rag.ipynb) |
| 🤖 **Complex reasoning** | Agentic RAG | [10_agentic_rag.ipynb](notebooks/advanced_architectures/10_agentic_rag.ipynb) |

**Rule of thumb:** Start with Simple RAG, upgrade only when needed.

[❓ Need help choosing? See FAQ →](docs/FAQ.md#which-architecture-should-i-choose)

## 📊 Performance at a Glance

| Architecture | Latency | Cost/Query | Accuracy | Best For |
|--------------|---------|------------|----------|----------|
| Simple RAG | ~2s | $0.00036 | Good | General Q&A |
| Adaptive RAG | Variable | $0.00090 | Very Good | Mixed workloads |
| Agentic RAG | ~30s | $0.00360 | Excellent | Complex tasks |

**Full benchmarks:** [docs/PERFORMANCE.md](docs/PERFORMANCE.md)

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
   - 🏃 **Fast track**: Pick one advanced architecture
   - 🔬 **Deep dive**: Complete all 8 architectures
   - 📊 **Comparison**: Jump to [11_comparison.ipynb](notebooks/advanced_architectures/11_comparison.ipynb)

**Total time:** 3-4 hours for complete tutorial

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 🐛 Report bugs
- ✨ Suggest features
- 📝 Improve documentation
- 💻 Submit pull requests

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

Made with ❤️ using Claude Code | [View Changelog](docs/CHANGELOG.md)
