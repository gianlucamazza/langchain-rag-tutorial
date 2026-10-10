# LangChain RAG Tutorial

![Python](https://img.shields.io/badge/python-3.10--3.13-blue.svg)
![LangChain](https://img.shields.io/badge/langchain-%3E%3D1.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange.svg)
![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)

Hands-on tutorial for building **Retrieval-Augmented Generation (RAG)** systems with LangChain 1.x.
It walks through patterns you would typically need before production — evaluation, monitoring, and
starter deployment templates — only to the extent the notebooks and templates in this repo actually
cover them.

This is a learning repo, not a finished product you can drop into production.

## Status

This tree ports the LangChain 1.x migration from the earlier fork (tip `a1ddb906`)
onto the claims-honesty `main`.

Pinned dependencies (see `requirements.txt`):

- Python **3.10–3.13**
- `langchain` / `langchain-core` / `langchain-openai` / `langgraph` >= 1.0
- `langchain-text-splitters` >= 1.0 (no longer pulled in by LangChain 1.x)
- `langchain-tavily` >= 0.2 (replaces deprecated `TavilySearchResults`)
- `langchain-community>=0.4.0,<0.4.2` because ragas 0.4.x still imports
  `langchain_community.chat_models.vertexai` ([vibrantlabsai/ragas#2753](https://github.com/vibrantlabsai/ragas/issues/2753))

Local checks on this port (Python 3.12.3, no API keys; 2026-10-10):

- Installed from these pins (CPU `torch` first, then `requirements.txt` + `requirements-dev.txt`). Resolved versions included `langchain==1.4.4`, `langchain-core==1.6.9`, `langchain-community==0.4.1`, `langchain-text-splitters==1.1.3`, `langchain-tavily==0.2.18`, `langgraph==1.2.14`, `ragas==0.4.3`.
- Unit tests: `pytest tests/ -m "not integration and not slow"` → **67 passed**, 16 deselected (integration). `shared/` coverage 60%.
- `jupyter nbconvert --to notebook --execute notebooks/00_index.ipynb` succeeded. The env-check cell printed `WARNING: OPENAI_API_KEY not found` / `OpenAI API Key: NOT LOADED` and continued.
- Import-check: `shared`, `shared.config`, `shared.loaders`, `shared.prompts`, `shared.retrievers`, `shared.utils` all import (`shared.__version__ == "1.4.0"`).
- Lint: `ruff check` + `ruff format --check` on `shared/ tests/ scripts/ templates/` and `mypy shared/ scripts/` are clean after formatting the main-only test files this merge kept.

Not run (need API keys and/or live model calls): notebooks 01–20 beyond the index (including 12, 19, 20), `make vector-stores`, and `tests/test_integration.py`.

Treat the notebooks as educational walkthroughs.

Shared-module unit tests (`pytest tests/ -m "not integration"`) do not require API keys.

## Quick Start

```bash
git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
cd langchain-rag-tutorial

python3 -m venv venv                # Python 3.10-3.13
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # then set OPENAI_API_KEY
jupyter notebook notebooks/00_index.ipynb
```

Prerequisites: Python 3.10-3.13, an OpenAI API key, about 2 GB of RAM and disk space. Notebooks 08
and 10 also use a Tavily API key for web search; notebook 17 needs Tesseract and Poppler.
See [Getting Started](docs/GETTING_STARTED.md) for the guided path and
[Installation](docs/INSTALLATION.md) for details.

## What the notebooks cover

### Fundamentals (notebooks 01–03)

- **Document loading & splitting** — load web docs and compare chunking strategies
- **Embeddings comparison** — OpenAI vs HuggingFace embeddings and FAISS persistence
- **Simple RAG** — similarity vs MMR retrieval and an LCEL RAG chain

[Fundamentals →](notebooks/fundamentals/)

### Advanced patterns (notebooks 04–20)

Tutorial implementations of these patterns (not hardened services):

| Notebook | Pattern | What it demonstrates |
| -------- | ------- | -------------------- |
| **04** | Memory RAG | Conversation history with `RunnableWithMessageHistory` |
| **05** | Branched RAG | Multi-query / parallel retrieval |
| **06** | HyDE | Hypothetical document embeddings |
| **07** | Adaptive RAG | Route simple / medium / complex queries |
| **08** | Corrective RAG | Relevance grading + web-search fallback |
| **09** | Self-RAG | Self-critique / refinement loop |
| **10** | Agentic RAG | Tool-using agent loop |
| **11** | Comparison | Qualitative trade-off tables (not a measured harness) |
| **12** | Contextual RAG | Context-augmented chunks; optional BM25 + rerank ablation |
| **13** | Fusion RAG | Reciprocal Rank Fusion |
| **14** | SQL RAG | Natural language → SQL on the Chinook sample DB |
| **15** | GraphRAG | Entity/relationship extraction + NetworkX |
| **16** | RAGAS evaluation | Faithfulness, relevancy, context precision/recall, correctness |
| **17** | Multimodal RAG | GPT-4 Vision + Tesseract OCR on images + text |
| **18** | Embedding fine-tuning | Contrastive fine-tune of a sentence-transformers model |
| **19** | Hybrid search + reranking | BM25 + dense (RRF) and a local cross-encoder |
| **20** | Parent / multi-vector | Search small chunks, return larger parents or summaries |

[Advanced notebooks →](notebooks/advanced_architectures/)

## Documentation

| Guide                                      | Contents                                            |
| ------------------------------------------ | --------------------------------------------------- |
| [Getting Started](docs/GETTING_STARTED.md) | Setup and learning path                             |
| [Installation](docs/INSTALLATION.md)       | Detailed setup, dependencies, configuration         |
| [Architecture](docs/ARCHITECTURE.md)       | How the tutorial is structured                      |
| [API Reference](docs/API_REFERENCE.md)     | The `shared/` helpers                               |
| [Examples](docs/EXAMPLES.md)               | Usage snippets                                      |
| [Performance](docs/PERFORMANCE.md)         | What to measure; illustrative cost/latency notes    |
| [Deployment](docs/DEPLOYMENT.md)           | Starter templates and checklists                    |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues                                       |
| [FAQ](docs/FAQ.md)                         | Frequently asked questions                          |
| [Contributing](docs/CONTRIBUTING.md)       | Development workflow                                |
| [Changelog](docs/CHANGELOG.md)             | Version history                                     |
| [Security](SECURITY.md)                    | Security policy                                     |

## Project Structure

```text
langchain-rag-tutorial/
|-- notebooks/
|   |-- 00_index.ipynb              # Start here: navigation and environment check
|   |-- fundamentals/               # 01-03
|   `-- advanced_architectures/     # 04-20
|-- shared/                         # config, utils, loaders, prompts, retrievers
|-- templates/                      # fastapi/, streamlit/, lambda/ (starter apps)
|-- scripts/build_vector_stores.py  # Pre-builds FAISS vector stores
|-- tests/                          # pytest suite for shared/ (plus optional integration)
|-- docs/                           # Documentation
|-- data/                           # Vector stores, Chinook DB (gitignored)
|-- monitoring/                     # Example Prometheus/Grafana config (not wired into compose)
|-- Dockerfile, docker-compose.yml  # Jupyter (8888) + FastAPI template (8000)
|-- Makefile, ruff.toml, pytest.ini, .pre-commit-config.yaml
|-- requirements.txt, requirements-dev.txt
`-- .env.example
```

## What this repo includes

**Notebooks**

- Fundamentals through GraphRAG, multimodal RAG, embedding fine-tuning, hybrid/rerank, parent/multi-vector, and RAGAS
- Reusable helpers in `shared/` (config, loaders, prompts, retrievers, FAISS save/load)
- Vector-store persistence keyed by embedding model so later notebooks can reuse embeddings from notebook 02

**Starter infrastructure (educational)**

- Docker Compose stack (Jupyter + FastAPI template)
- FastAPI, Streamlit, and AWS Lambda starter templates
- pytest for `shared/` plus GitHub Actions workflows already in the repo
- Makefile, ruff, and pre-commit config
- Example Prometheus/Grafana files under `monitoring/` (not started by compose)

**Stack used in the notebooks**

- LangChain 1.x (see [Status](#status) and `requirements.txt`)
- OpenAI chat + embeddings (GPT-4o-mini / GPT-4o vision in notebook 17)
- FAISS, sentence-transformers, NetworkX, RAGAS, rank-bm25
- Tesseract OCR + Pillow (notebook 17)
- FastAPI / Streamlit / Docker in the templates

[Architecture notes →](docs/ARCHITECTURE.md)

## Choosing a notebook

| If you want to try… | Open |
| ------------------- | ---- |
| Fast, simple Q&A | [03_simple_rag.ipynb](notebooks/fundamentals/03_simple_rag.ipynb) |
| Chat with follow-up questions | [04_rag_with_memory.ipynb](notebooks/advanced_architectures/04_rag_with_memory.ipynb) |
| Multi-query retrieval | [05_branched_rag.ipynb](notebooks/advanced_architectures/05_branched_rag.ipynb) |
| Ambiguous queries (HyDE) | [06_hyde.ipynb](notebooks/advanced_architectures/06_hyde.ipynb) |
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
| Identifiers, codes, or jargon | [19_hybrid_search_reranking.ipynb](notebooks/advanced_architectures/19_hybrid_search_reranking.ipynb) |
| Search small, return large | [20_parent_multivector_retrieval.ipynb](notebooks/advanced_architectures/20_parent_multivector_retrieval.ipynb) |
| RAGAS metrics | [16_evaluation_ragas.ipynb](notebooks/advanced_architectures/16_evaluation_ragas.ipynb) |

Start with Simple RAG, then pick a pattern that matches the question you want to explore. Notebook 11 is a qualitative comparison, not a reproducible benchmark.

[FAQ: which architecture? →](docs/FAQ.md#which-architecture-should-i-choose)

## Starter deployment templates

These are teaching templates (auth, tenancy, SLOs, and load testing are out of scope).

### Docker Compose

```bash
cp .env.example .env
docker compose up -d

# Or build a single image
docker build -t langchain-rag:latest .
docker run -p 8888:8888 --env-file .env langchain-rag:latest
```

Compose starts Jupyter on port 8888 and the FastAPI template on port 8000. Example Prometheus/Grafana configs remain under `monitoring/` but are not wired into compose.

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

**Notes and checklists:** [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)

## Prerequisites

- **Python** 3.10–3.13
- **OpenAI API key** ([create one](https://platform.openai.com/api-keys))
- **~2GB RAM** (more if you run notebook 18 fine-tuning)
- **~2GB disk** for dependencies and local models
- **System packages** for notebook 17: Tesseract OCR, Poppler (see [INSTALLATION.md](docs/INSTALLATION.md))

[Detailed requirements →](docs/INSTALLATION.md#prerequisites)

## Learning path

1. **Setup:** [GETTING_STARTED.md](docs/GETTING_STARTED.md)
2. **Hub:** [00_index.ipynb](notebooks/00_index.ipynb)
3. **Fundamentals:** [notebooks 01–03](notebooks/fundamentals/)
4. Then either:
   - Simple RAG → one specialized notebook for your use case
   - Work through 04–20 in order
   - Skim [11_comparison.ipynb](notebooks/advanced_architectures/11_comparison.ipynb) then jump
   - Run [16_evaluation_ragas.ipynb](notebooks/advanced_architectures/16_evaluation_ragas.ipynb) after you have a chain

Times in the docs are rough teaching estimates, not measured wall-clock benchmarks.

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

## Development

```bash
make install      # Install dependencies
make test         # pytest with coverage
make lint         # ruff check + ruff format --check + mypy
make format       # ruff format + ruff check --fix
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

CI runs tests on Python 3.10, 3.11, 3.12 and 3.13, and linting on 3.12. See
[Contributing](docs/CONTRIBUTING.md) for the full workflow.

## License

MIT License — see [LICENSE](LICENSE).

## Resources

- **Docs:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- **Discussions:** [GitHub Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)
- **LangChain:** [docs.langchain.com](https://docs.langchain.com/)

## Getting help

- [FAQ](docs/FAQ.md)
- [Existing issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- [New issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues/new)
- [Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)

---

Latest notes: v1.4.0 (ported 2026-10-10 from the 2026-09-21 fork migration) — LangChain 1.x, hybrid
search / reranking, parent-document and multi-vector retrieval. See the [Changelog](docs/CHANGELOG.md).
