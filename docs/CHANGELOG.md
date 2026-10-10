# Changelog

All notable changes to LangChain RAG Tutorial will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Ported LangChain 1.x migration** from [Venere-Labs/langchain-rag-tutorial#1](https://github.com/Venere-Labs/langchain-rag-tutorial/pull/1) into this repository (2026-10-10). The v1.3.0 and v1.4.0 entries below keep the fork's original 2026-09-21 dates. Docs keep the tutorial/honesty framing from the 2026-10-10 claims pass (no “production-ready” / enterprise claims).
- **Claims honesty (docs/metadata):** Dropped unbacked “production-ready” / superlative framing from README, notebook markdown, docs, and package comments. The repo is a hands-on tutorial; starter templates and evaluation/monitoring examples are listed only where they actually exist. Benchmark-style tables without a reproducible script are labeled as teaching estimates.

## [1.4.0] - 2026-09-21

Hybrid search, reranking, and parent-document / multi-vector retrieval.

### Added

- **Notebook 19, Hybrid Search + Reranking** (complexity 3/5): BM25 and dense (FAISS) retrieval
  fused with weighted Reciprocal Rank Fusion (`EnsembleRetriever`), then reranked by a local
  cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) through `ContextualCompressionRetriever` +
  `CrossEncoderReranker`. Compares Dense, BM25, Hybrid and Hybrid + Rerank with Hit@4 and MRR@4
  on a small labeled query set (exact-identifier vs paraphrased queries), sweeps `bm25_weight`
  and inspects reranker scores. Needs no extra API key; the first run downloads the reranker
  model (~90 MB).
- **Notebook 20, Parent-Document and Multi-Vector Retrieval** (complexity 3/5):
  `ParentDocumentRetriever` (400-character children, 2000-character parents) and
  `MultiVectorRetriever` indexing an LLM summary and 3 hypothetical questions per chunk
  (`with_structured_output`), cached in `data/cache/`. Compared with the baseline on Hit@4,
  MRR@4 and context size.
- **Notebook 12, section 11 "Full Contextual Retrieval: Contextual BM25 + Reranking"**:
  Anthropic's full recipe (contextual embeddings + contextual BM25 + reranking) with an ablation
  of 4 configurations.
- **`shared/retrievers.py`**: `bm25_tokenize()` (lowercases and splits on punctuation, so
  identifiers such as `max_retries=2,` match; LangChain's BM25 default splits on whitespace only),
  `build_bm25_retriever()`, `build_hybrid_retriever()` (raises
  `ValueError` if `bm25_weight` is not in [0, 1]; returns up to `2 * k` unique documents) and
  `build_reranking_retriever()`, all re-exported from `shared`.
- **`DEFAULT_RERANKER_MODEL`** in `shared/config.py` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`,
  env-overridable), also reported by `get_project_info()` as `reranker_model`.
- **Prompts** `CHUNK_SUMMARY_PROMPT` (variable `chunk`) and `HYPOTHETICAL_QUESTIONS_PROMPT`
  (variables `chunk`, `num_questions`), exported from `shared` and registered in
  `get_prompt_by_name()` as `"chunk_summary"` and `"hypothetical_questions"`.
- **Dependencies**: `langchain-classic>=1.0.0` (in LangChain 1.x, `EnsembleRetriever`, the
  parent-document and multi-vector retrievers and the rerankers live there) and
  `rank-bm25>=0.2.2`.
- `tests/test_retrievers.py`, offline (fake embeddings and a fake cross-encoder).

### Changed

- `sentence-transformers` is now also used for cross-encoder reranking (notebooks 12 and 19),
  not only for embedding fine-tuning (notebook 18).
- Documentation, the index notebook and the architecture selection guides cover notebooks 19 and
  20: 21 notebooks (index plus 01-20) and 15 RAG architectures.

---

## [1.3.0] - 2026-09-21

Migration to LangChain 1.x and modernized tooling.

### Changed

- **Python support is now 3.10-3.13.** The Docker image and the CI lint job use Python 3.12; the
  CI test matrix covers 3.10, 3.11, 3.12 and 3.13.
- **LangChain 1.x**: `langchain>=1.0`, `langchain-core>=1.0`, `langchain-openai>=1.0`,
  `langgraph>=1.0`; `langchain-text-splitters>=1.0` is now an explicit dependency (no longer pulled
  in by `langchain` 1.x).
- **Web search** (notebooks 08 and 10) uses `TavilySearch` from the new `langchain-tavily>=0.2`
  package instead of the deprecated `langchain_community` `TavilySearchResults`. Requires
  `TAVILY_API_KEY`.
- **`langchain-community` pinned to `>=0.4.0,<0.4.2`**: 0.4.2 removed
  `langchain_community.chat_models.vertexai`, which ragas 0.4.x imports at module load
  ([vibrantlabsai/ragas#2753](https://github.com/vibrantlabsai/ragas/issues/2753)). The upper
  bound will be removed once ragas ships a fix. `langchain-community` is being sunset upstream
  ([langchain-ai/langchain-community#674](https://github.com/langchain-ai/langchain-community/issues/674))
  but remains the official home of FAISS and `WebBaseLoader`.
- **Notebook 04** uses `langchain_core` `InMemoryChatMessageHistory` instead of
  `langchain_community` `ChatMessageHistory`.
- **Templates** (FastAPI, Streamlit, Lambda) read the chat model from `DEFAULT_MODEL`
  (default `gpt-4o-mini`) instead of hardcoding it.
- **Tooling**: black, isort and flake8 replaced by ruff (lint and format, configured in
  `ruff.toml`); mypy is kept. `make lint` runs `ruff check`, `ruff format --check` and mypy;
  `make format` runs `ruff format` and `ruff check --fix`. Pre-commit uses the `ruff-pre-commit`
  hooks.
- **Vector store paths are keyed by embedding model** (`shared/config.py`):
  `OPENAI_VECTOR_STORE_PATH` = `data/vector_stores/openai__<OPENAI_EMBEDDING_MODEL>` (default
  `openai__text-embedding-3-small`) and `HF_VECTOR_STORE_PATH` =
  `data/vector_stores/hf__<HF_EMBEDDING_MODEL with / replaced by __>` (default
  `hf__BAAI__bge-small-en-v1.5`), so changing the model never loads a stale index.
  `require_vector_store()` raises `FileNotFoundError` with build instructions when a store is
  missing.
- **Docker**: the image installs CPU-only `torch` (several GB smaller than the default CUDA
  build pulled in by `sentence-transformers`), `tesseract-ocr` and `poppler-utils` (notebook 17 works in the
  container), copies `shared/`, `scripts/`, `templates/` and `notebooks/`, drops the gcc/g++ build
  stage, and runs Jupyter as a non-root user without `--allow-root`.
- **Docker Compose** rewritten with two services: `notebooks` (Jupyter, port 8888) and `api`
  (FastAPI template via `uvicorn templates.fastapi.app:app`, port 8000, health check
  `GET /health`). Both use `env_file: .env` (run `cp .env.example .env` first) and mount `./data`.
  The obsolete `version:` key was removed. Makefile Docker targets use `docker compose`
  (`build`, `up -d`, `down`).
- `.python-version` is now 3.12.
- **`make vector-stores`** runs the new `scripts/build_vector_stores.py`, with
  `--provider {openai,huggingface,all}` (default `all`) and `--version TAG` (writes to
  `data/vector_stores/<TAG>/` instead of `data/vector_stores/`).
- **Documentation**: emoji removed; README reduced to an overview, quick start and links into
  `docs/`; duplicated troubleshooting, selection-guide and benchmark content consolidated into
  single sources.

### Added

- `scripts/build_vector_stores.py` (see `make vector-stores`).
- `tests/test_vector_stores.py` and `tests/test_lambda_handler.py`, using fake embeddings and LLMs
  (no network access).
- `ruff.toml` (notebooks are excluded from linting).
- `numexpr` for the calculator tool in notebook 10.

### Fixed

- OpenAI embeddings were created without a model in notebooks 02-10 and 17 and in the templates,
  silently using `text-embedding-ada-002` while configuration and docs said
  `text-embedding-3-small`. All now pass `OPENAI_EMBEDDING_MODEL`.
- Lambda template: `NameError` on `RAG_PROMPT` when `k != 4`; a custom `k` leaked into later
  requests through the global chain; documents were joined with literal backslash-n text instead of newlines;
  `FAISS.load_local` lacked `allow_dangerous_deserialization` (fails on current
  `langchain-community`); retrieval ran twice per request.
- Notebook 02: the comparison table labelled the OpenAI model incorrectly.
- `.env.example` set `HF_EMBEDDING_MODEL` to `all-MiniLM-L6-v2` while the code default and
  notebook 02 use `BAAI/bge-small-en-v1.5`; both now use BGE.
- Lambda template: `VECTOR_STORE_KEY` and the layer path default to `openai__<OPENAI_EMBEDDING_MODEL>`,
  matching the local store layout, instead of a hardcoded `openai_embeddings`.
- Default document URLs pointed to `python.langchain.com` pages that redirect (two to the same
  page); they now use docs.langchain.com pages (`deepagents/rag`, `deepagents/retrieval`,
  `langchain/models`, `langchain/short-term-memory`).
- Notebook 04: the memory tip referenced the removed `ConversationBufferWindowMemory`; it now uses
  `trim_messages`. RAGAS ground truth and the notebook 18 sample text no longer reference legacy
  memory classes.
- Docker health check now targets Jupyter's `/api` endpoint on port 8888, which is what the
  default command runs; it previously probed `:8000/health`, which nothing served.
- `make vector-stores` referenced a script that did not exist.
- Documentation no longer claims a `VECTOR_STORE_VERSION` Docker build argument, which never
  existed.

### Removed

- Python 3.9 support (end-of-life; LangChain 1.x requires 3.10+).
- Unused runtime dependencies: `duckduckgo-search`, `spacy`, `plotly`, `sqlalchemy` (the SQL
  notebook uses the standard-library `sqlite3`).
- Unused development dependencies: `mkdocs`, `mkdocs-material` (no MkDocs site exists), `black`,
  `flake8`, `isort`, `pytest-asyncio`, `memory-profiler`.
- Redis, Prometheus and Grafana services and the `monitoring` profile from `docker-compose.yml`
  (no code used Redis, and the monitoring configuration never existed).

### Upgrade notes

- Rebuild vector stores created with v1.2.x (`make vector-stores` or notebook 02). They were built
  with `text-embedding-ada-002` and live under the old `openai_embeddings/` and
  `huggingface_embeddings/` directories, which are no longer read.
- Create `.env` before `docker compose up` (`cp .env.example .env`).

---

## [1.2.1] - 2025-11-13

### Fixed

**Critical Fixes (Notebook Import Errors):**

- Fixed incorrect config imports in notebooks 12-18: `LLM_MODEL` -> `DEFAULT_MODEL`, `LLM_TEMPERATURE` -> `DEFAULT_TEMPERATURE`, `EMBEDDINGS_MODEL` -> `OPENAI_EMBEDDING_MODEL`, `VECTOR_STORE_PATH` -> `VECTOR_STORE_DIR`
- Fixed deprecated LangChain imports: `langchain.schema.Document` -> `langchain_core.documents.Document` (notebooks 17-18)
- Fixed `load_and_split()` tuple unpacking in notebooks 12, 13, 15, 16 (was causing AttributeError)
- Fixed `load_vector_store()` exception handling in `shared/utils.py` (now returns `None` instead of raising, fixing RuntimeError in notebooks 12-13)
- Fixed Python 3.9 compatibility: Updated type hints from lowercase `tuple`/`dict` to `Tuple`/`Dict` in:
  - `shared/loaders.py`: Lines 130, 132, 188 (functions `compare_splitting_strategies`, `load_and_split`)
  - `notebooks/14_sql_rag.ipynb`: Cell 12 (function `execute_sql_safely`)
  - `notebooks/14_sql_rag.ipynb`: Cell 2 (added `Tuple` import), Cell 4 (use `PROJECT_ROOT` for database path)
- Removed unused import `RAG_PROMPT_TEMPLATE` from notebook 09 (code cleanup)
- All notebooks now execute without ImportError, ModuleNotFoundError, AttributeError, RuntimeError, or TypeError (Python 3.9+)

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
- Enhanced `load_vector_store()` error handling with proper Optional[FAISS] return type
- Full Python 3.9+ compatibility with proper type hints (Tuple, Dict imports)
- Notebook 14: More robust database path construction using PROJECT_ROOT

---

## [1.2.0] - 2025-11-13

### Added

- **2 New Advanced Notebooks** (17-18):
  - **17_multimodal_rag.ipynb** - Multimodal RAG with Images + Text (complexity 4/5)
    - GPT-4 Vision API integration for image understanding
    - OCR support with Tesseract for text extraction
    - PDF image extraction capabilities
    - Combined text + image retrieval system
    - Production optimization strategies (caching, batch processing)
    - Cost-benefit analysis for vision vs OCR approaches
  - **18_finetuning_embeddings.ipynb** - Domain-Specific Fine-tuning Guide (complexity 4/5)
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
- **Total Notebooks**: 16 -> 18 (+2)
- **Project Completeness**: Development → tutorial + starter templates (later reframed; not a production product)

### Improved

- **Deployment Options**: Docs only -> 3 production templates (FastAPI, Streamlit, Lambda)
- **Docker Support**: None -> Full container support with monitoring
- **Testing**: 0% coverage -> Infrastructure for 70%+ coverage
- **CI/CD**: None -> Full GitHub Actions pipeline
- **Developer Experience**: Added Makefile, pre-commit hooks, comprehensive dev tools

### Fixed

- requirements.txt: Consolidated all dependencies with proper version constraints
- Documentation: Added missing SECURITY.md referenced in other docs

---

## [1.1.0] - 2025-11-12

### Added

- **4 New Advanced RAG Architectures** (notebooks 12-15):
  - **12_contextual_rag.ipynb** - Context-Augmented Retrieval (complexity 3/5)
    - Anthropic's technique for chunk augmentation
    - Document-level summarization with chunk-specific context
    - Improves precision with minimal query overhead
    - ~15-30% better retrieval quality with same query cost
  - **13_fusion_rag.ipynb** - Reciprocal Rank Fusion (complexity 3/5)
    - RAG-Fusion implementation with RRF algorithm
    - Multiple query generation + sophisticated ranking
    - Best-in-class ranking quality
    - Outperforms simple multi-query deduplication
  - **14_sql_rag.ipynb** - Natural Language to SQL (complexity 4/5)
    - Complete text-to-SQL pipeline
    - Chinook database integration (music store sample DB)
    - Schema retrieval with semantic search
    - Safe SQL execution with validation
    - Error recovery and query correction
    - Perfect for analytics and structured data queries
  - **15_graphrag.ipynb** - Graph-Based Knowledge Retrieval (complexity 5/5)
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
  - **prompts.py**: 18 new prompt templates (13 -> 30+ total)
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
  - Updated feature count: "8 architectures" -> "12 architectures"
  - Added RAGAS evaluation mention
  - Updated technical stack (NetworkX, SQLAlchemy, RAGAS)
  - New architecture selection guide with 12 entries
  - Performance table now includes new architectures

- **Project Statistics**:
  - Total notebooks: 12 -> 16
  - Advanced architectures: 8 -> 13 (including comparison + evaluation)
  - Prompt templates: 13 -> 30+
  - Shared module: 983 -> 1500+ lines
  - Dependencies: 12 -> 20+ packages

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
    - 04_rag_with_memory.ipynb - Conversational RAG (complexity 2/5)
    - 05_branched_rag.ipynb - Multi-query retrieval (complexity 3/5)
    - 06_hyde.ipynb - Hypothetical documents (complexity 3/5)
    - 07_adaptive_rag.ipynb - Query routing (complexity 4/5)
    - 08_corrective_rag.ipynb - CRAG with web search (complexity 4/5)
    - 09_self_rag.ipynb - Self-reflective RAG (complexity 5/5)
    - 10_agentic_rag.ipynb - Autonomous agents (complexity 5/5)
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

- **README.md refactored**: 615 lines -> 250 lines (landing page only)
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

## Under Consideration

- Ollama local LLM integration
- Azure OpenAI, AWS Bedrock and Google Vertex AI support
- Hybrid search (keyword + semantic)
- Re-ranking strategies

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

This project is licensed under the MIT License.
