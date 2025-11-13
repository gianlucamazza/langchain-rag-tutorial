# Changelog

All notable changes to LangChain RAG Tutorial will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2025-11-13

### Fixed

**Critical Fixes (Notebook Import Errors):**
- Fixed incorrect config imports in notebooks 12-18: `LLM_MODEL` → `DEFAULT_MODEL`, `LLM_TEMPERATURE` → `DEFAULT_TEMPERATURE`, `EMBEDDINGS_MODEL` → `OPENAI_EMBEDDING_MODEL`, `VECTOR_STORE_PATH` → `VECTOR_STORE_DIR`
- Fixed deprecated LangChain imports: `langchain.schema.Document` → `langchain_core.documents.Document` (notebooks 17-18)
- Fixed `load_and_split()` tuple unpacking in notebooks 12, 13, 15, 16 (was causing AttributeError)
- Removed unused import `RAG_PROMPT_TEMPLATE` from notebook 09 (code cleanup)
- All notebooks now execute without ImportError, ModuleNotFoundError, or AttributeError

**Dependencies:**
- Added missing dependencies to requirements.txt:
  - `langchain-core>=0.1.0` (required for updated imports)
  - `sentence-transformers>=2.2.0` (fine-tuning embeddings, notebook 18)
  - `python-louvain>=0.16` (community detection in GraphRAG, notebook 15)
  - `fastapi>=0.109.0`, `streamlit>=1.31.0`, `uvicorn>=0.27.0`, `boto3>=1.34.0` (production templates)
- Reorganized requirements.txt with clear categories and inline documentation

**Configuration:**
- Added `DEFAULT_VISION_MODEL` constant to `shared/config.py` for multimodal RAG (defaults to "gpt-4o")
- Improved configuration consistency across all notebooks

### Changed

- **requirements.txt**: Complete reorganization with categories (Core LangChain, Vector Stores, Data Analysis, Graph & Knowledge, Multimodal, Production Deployment, etc.)
- **requirements.txt**: Updated header with v1.2.1 version and installation instructions

### Improved

- All 18 notebooks now use consistent, correct config constant names
- Better dependency management with clear categorization and comments
- Improved code portability and maintainability

---

## [1.2.0] - 2025-11-13

### Added

- **2 New Advanced Notebooks** (17-18):
  - **17_multimodal_rag.ipynb** - Multimodal RAG with Images + Text (⭐⭐⭐⭐)
    - GPT-4 Vision API integration for image understanding
    - OCR support with Tesseract for text extraction
    - PDF image extraction capabilities
    - Combined text + image retrieval system
    - Production optimization strategies (caching, batch processing)
    - Cost-benefit analysis for vision vs OCR approaches
  - **18_finetuning_embeddings.ipynb** - Domain-Specific Fine-tuning Guide (⭐⭐⭐⭐)
    - Complete guide to fine-tuning sentence-transformers
    - Contrastive learning with MultipleNegativesRankingLoss
    - Dataset preparation and auto-generation strategies
    - Baseline vs fine-tuned performance comparison
    - Production best practices and hyperparameter tuning
    - Cost-benefit analysis and ROI calculations

- **Docker Support** (Production Infrastructure):
  - Multi-stage Dockerfile with security best practices
  - docker-compose.yml with full stack (app + Redis + monitoring)
  - .dockerignore for optimized builds
  - Non-root user configuration
  - Health checks and monitoring setup
  - Prometheus + Grafana integration (optional profiles)

- **Production Templates** (templates/ directory):
  - **FastAPI Template** (templates/fastapi/):
    - Complete REST API with automatic documentation
    - Request validation with Pydantic models
    - Error handling and logging
    - CORS configuration
    - Health check endpoints
    - Rate limiting ready
    - Async support
  - **Streamlit Template** (templates/streamlit/):
    - Interactive web UI for RAG queries
    - Real-time query processing
    - Source document display
    - Architecture selection
    - Performance metrics visualization
    - Sample queries and responsive design
  - **AWS Lambda Template** (templates/lambda/):
    - Serverless deployment ready
    - S3 vector store integration
    - Cold start optimization
    - API Gateway integration
    - Cost-optimized configuration

- **Testing Infrastructure** (tests/ directory):
  - pytest configuration (pytest.ini)
  - Comprehensive test suite:
    - test_config.py - Configuration tests
    - test_utils.py - Utility function tests
    - conftest.py - Shared fixtures
  - Test coverage reporting (HTML + XML)
  - CI/CD integration ready

- **CI/CD Pipelines** (.github/workflows/):
  - Automated testing workflow (Python 3.9, 3.10, 3.11)
  - Code linting workflow (black, flake8, isort, mypy)
  - Coverage reporting to Codecov
  - Matrix testing across Python versions

- **Development Tools**:
  - **Makefile** with common commands:
    - `make install` - Install dependencies
    - `make test` - Run test suite
    - `make lint` - Run linters
    - `make format` - Format code
    - `make docker-build` - Build Docker image
    - `make docker-run` - Run containers
    - `make clean` - Clean cache files
  - **pre-commit hooks** (.pre-commit-config.yaml):
    - Automatic code formatting (black, isort)
    - Linting checks (flake8)
    - Security checks (detect-private-key)
    - YAML/JSON validation
  - **requirements-dev.txt** - Development dependencies:
    - pytest + pytest-cov
    - black, flake8, mypy, isort
    - pre-commit
    - mkdocs + mkdocs-material

- **Documentation**:
  - **SECURITY.md** - Security policy and vulnerability reporting
  - Comprehensive README files for each template
  - Docker deployment documentation
  - CI/CD setup guides

- **New Dependencies**:
  - `pillow>=10.0.0` - Image processing
  - `pytesseract>=0.3.10` - OCR text extraction
  - `pdf2image>=1.16.0` - PDF image extraction
  - `fastapi>=0.109.0` - REST API framework
  - `streamlit>=1.31.0` - Web UI framework
  - `uvicorn>=0.27.0` - ASGI server
  - `boto3>=1.34.0` - AWS SDK (Lambda deployment)

### Changed

- **requirements.txt**: Updated with all multimodal dependencies
- **Project Structure**: Now includes templates/, tests/, .github/ directories
- **README.md**: Updated to mention Docker support and production templates
- **Total Notebooks**: 16 → 18 (+2)
- **Project Completeness**: Development → Production-ready

### Improved

- **Deployment Options**: Docs only → 3 production templates (FastAPI, Streamlit, Lambda)
- **Docker Support**: None → Full container support with monitoring
- **Testing**: 0% coverage → Infrastructure for 70%+ coverage
- **CI/CD**: None → Full GitHub Actions pipeline
- **Developer Experience**: Added Makefile, pre-commit hooks, comprehensive dev tools

### Fixed

- requirements.txt: Consolidated all dependencies with proper version constraints
- Documentation: Added missing SECURITY.md referenced in other docs

---

## [1.1.0] - 2025-11-12

### Added

- **4 New Advanced RAG Architectures** (notebooks 12-15):
  - **12_contextual_rag.ipynb** - Context-Augmented Retrieval (⭐⭐⭐)
    - Anthropic's technique for chunk augmentation
    - Document-level summarization with chunk-specific context
    - Improves precision with minimal query overhead
    - ~15-30% better retrieval quality with same query cost
  - **13_fusion_rag.ipynb** - Reciprocal Rank Fusion (⭐⭐⭐)
    - RAG-Fusion implementation with RRF algorithm
    - Multiple query generation + sophisticated ranking
    - Best-in-class ranking quality
    - Outperforms simple multi-query deduplication
  - **14_sql_rag.ipynb** - Natural Language to SQL (⭐⭐⭐⭐)
    - Complete text-to-SQL pipeline
    - Chinook database integration (music store sample DB)
    - Schema retrieval with semantic search
    - Safe SQL execution with validation
    - Error recovery and query correction
    - Perfect for analytics and structured data queries
  - **15_graphrag.ipynb** - Graph-Based Knowledge Retrieval (⭐⭐⭐⭐⭐)
    - Microsoft Research's GraphRAG approach
    - Entity extraction + relationship mapping
    - NetworkX graph construction and traversal
    - Multi-hop reasoning capability
    - Community detection (Louvain algorithm)
    - Graph visualization with matplotlib
    - Best for relationship queries and knowledge exploration

- **RAGAS Evaluation Framework** (notebook 16):
  - **16_evaluation_ragas.ipynb** - Comprehensive RAG Assessment
    - 6 evaluation metrics (faithfulness, relevancy, precision, recall, similarity, correctness)
    - Test dataset creation and ground truth management
    - Per-architecture quality comparison
    - Cost-quality trade-off analysis
    - Production readiness scoring
    - Visualization and reporting

- **Enhanced Shared Module**:
  - **prompts.py**: 18 new prompt templates (13 → 30+ total)
    - Contextual RAG: DOCUMENT_SUMMARY, CONTEXTUAL_CHUNK, CONTEXTUAL_RAG_ANSWER
    - Fusion RAG: FUSION_QUERY_GENERATION, FUSION_RAG_ANSWER
    - SQL RAG: SQL_SCHEMA_SUMMARY, TEXT_TO_SQL, SQL_RESULTS_INTERPRETATION, SQL_ERROR_RECOVERY
    - GraphRAG: ENTITY_EXTRACTION, RELATIONSHIP_EXTRACTION, ENTITY_DISAMBIGUATION, GRAPH_SUMMARIZATION, GRAPHRAG_ANSWER
  - **utils.py**: Expanded from 983 to 1500+ lines
  - **config.py**: New configuration options for SQL and Graph databases

- **New Dependencies**:
  - `networkx>=3.2` - Graph algorithms and analysis
  - `matplotlib>=3.8.0` - Graph visualization
  - `plotly>=5.18.0` - Interactive visualizations
  - `sqlalchemy>=2.0.25` - Database ORM and SQL toolkit
  - `pandas>=2.2.0` - Data manipulation and SQL results
  - `spacy>=3.7.0` - NLP and entity extraction
  - `ragas>=0.1.7` - RAG evaluation framework
  - `datasets>=2.16.0` - Evaluation dataset management

- **Data Assets**:
  - Chinook SQLite database (984KB) for SQL RAG demonstrations
  - Sample evaluation datasets for RAGAS metrics

### Changed

- **Comparison notebook** (11_comparison.ipynb):
  - Extended from 8 to 12 architectures
  - Updated all comparison tables and matrices
  - Added hybrid architecture recommendations
  - New decision framework for 12 architectures
  - Enhanced trade-offs analysis

- **README.md**:
  - Updated feature count: "8 architectures" → "12 architectures"
  - Added RAGAS evaluation mention
  - Updated technical stack (NetworkX, SQLAlchemy, RAGAS)
  - New architecture selection guide with 12 entries
  - Performance table now includes new architectures

- **Project Statistics**:
  - Total notebooks: 12 → 16
  - Advanced architectures: 8 → 13 (including comparison + evaluation)
  - Prompt templates: 13 → 30+
  - Shared module: 983 → 1500+ lines
  - Dependencies: 12 → 20+ packages

### Fixed

- requirements.txt: Added all new dependencies with version constraints
- Prompt templates: Improved JSON parsing error handling
- Vector store: Better caching and persistence patterns

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

### Planned for v1.2.0

- [ ] Multimodal RAG (images + text)
- [ ] Fine-tuning embeddings guide
- [ ] Docker support
- [ ] CI/CD pipeline for notebook testing

### Planned for v1.3.0

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
