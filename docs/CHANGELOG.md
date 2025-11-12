# Changelog

All notable changes to LangChain RAG Tutorial will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-11-12

### Added

- **Modular documentation structure** (`docs/` directory)
  - GETTING_STARTED.md - Quick start guide
  - INSTALLATION.md - Detailed setup instructions
  - API_REFERENCE.md - Shared module documentation
  - CONTRIBUTING.md - Contribution guidelines
  - ARCHITECTURE.md - Design decisions
  - TROUBLESHOOTING.md - Common issues and solutions
  - PERFORMANCE.md - Benchmarks and optimization
  - FAQ.md - Frequently asked questions
  - CHANGELOG.md - Version history (this file)
  - EXAMPLES.md - Usage patterns
  - DEPLOYMENT.md - Production deployment guide

- **Shared utilities module** (`shared/`)
  - config.py - Centralized configuration
  - utils.py - Reusable utility functions
  - loaders.py - Document loading and splitting
  - prompts.py - 13 prompt templates for all architectures
  - Vector store persistence (save/load)
  - Cost estimation utilities

- **12 comprehensive notebooks**
  - 00_index.ipynb - Navigation hub with environment validation
  - **Fundamentals (01-03)**:
    - 01_setup_and_basics.ipynb - Document loading and splitting
    - 02_embeddings_comparison.ipynb - OpenAI vs HuggingFace
    - 03_simple_rag.ipynb - Basic RAG implementation
  - **Advanced Architectures (04-11)**:
    - 04_rag_with_memory.ipynb - Conversational RAG (⭐⭐)
    - 05_branched_rag.ipynb - Multi-query retrieval (⭐⭐⭐)
    - 06_hyde.ipynb - Hypothetical documents (⭐⭐⭐)
    - 07_adaptive_rag.ipynb - Query routing (⭐⭐⭐⭐)
    - 08_corrective_rag.ipynb - CRAG with web search (⭐⭐⭐⭐)
    - 09_self_rag.ipynb - Self-reflective RAG (⭐⭐⭐⭐⭐)
    - 10_agentic_rag.ipynb - Autonomous agents (⭐⭐⭐⭐⭐)
    - 11_comparison.ipynb - Full benchmark of all architectures

- **Features**:
  - OpenAI GPT-4o-mini integration
  - FAISS vector store with persistence
  - HuggingFace local embeddings (no API costs)
  - LangChain LCEL chains
  - Metadata filtering
  - MMR retrieval strategy
  - Web search fallback (DuckDuckGo)
  - ReAct agent pattern
  - Comprehensive comparison metrics

### Fixed

- Security: Removed .env from git tracking
- Security: Added .env.example template
- Documentation: Eliminated 13% content duplication
- Code: Centralized configuration management

### Changed

- **README.md refactored**: 615 lines → 250 lines (landing page only)
- **Documentation modularized**: Single-source-of-truth approach
- **Project structure**: Organized into fundamentals/ and advanced_architectures/
- **Dependencies updated**: Added duckduckgo-search, langgraph

### Removed

- Legacy monolithic notebook (langchain_rag_tutorial.ipynb)
- Duplicate content across READMEs
- Empty data/ subdirectories

## [0.1.0] - 2024-11-11

### Initial Release (Pre-Modularization)

- Single monolithic notebook: langchain_rag_tutorial.ipynb
- Basic RAG implementation
- OpenAI and HuggingFace embeddings comparison
- FAISS vector store
- Simple retrieval strategies

---

## Upcoming

### Planned for v1.1.0

- [ ] Graph RAG notebook
- [ ] Multimodal RAG (images)
- [ ] Fine-tuning embeddings guide
- [ ] Advanced evaluation metrics (RAGAS)
- [ ] Docker support
- [ ] CI/CD pipeline for notebook testing

### Planned for v1.2.0

- [ ] Production deployment templates (FastAPI, Streamlit)
- [ ] Monitoring and observability (LangSmith integration)
- [ ] Cost optimization strategies
- [ ] Batch processing patterns
- [ ] Async/concurrent implementations

### Under Consideration

- Ollama local LLM integration
- Azure OpenAI support
- AWS Bedrock support
- Google Vertex AI support
- Custom embeddings training
- Hybrid search (keyword + semantic)
- Re-ranking strategies

---

## Version History

- **v1.0.0** (2024-11-12): Modular structure, 8 advanced architectures, comprehensive docs
- **v0.1.0** (2024-11-11): Initial release, monolithic notebook

---

## How to Upgrade

### From v0.1.0 to v1.0.0

**Breaking Changes:**

- Monolithic notebook replaced by modular structure
- Imports now use `shared` module

**Migration Steps:**

1. **Backup old work:**

   ```bash
   cp langchain_rag_tutorial.ipynb langchain_rag_tutorial_backup.ipynb
   ```

2. **Pull latest:**

   ```bash
   git pull origin main
   ```

3. **Update dependencies:**

   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **Start fresh:**
   - Begin with `notebooks/00_index.ipynb`
   - Complete fundamentals (01-03)
   - Explore advanced architectures (04-11)

**No data migration needed:** Vector stores are backward compatible.

---

## Contributing

Found a bug? Have a feature request? See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
