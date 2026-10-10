# Installation Guide

Detailed installation and configuration instructions. For the short version, see
[GETTING_STARTED.md](GETTING_STARTED.md).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Dependencies](#dependencies)
- [API Key Configuration](#api-key-configuration)
- [Configuration Reference](#configuration-reference)
- [Pre-building Vector Stores](#pre-building-vector-stores)
- [Validation](#validation)

## Prerequisites

### System Requirements

- **Python**: 3.10, 3.11, 3.12 or 3.13. Python 3.9 is no longer supported (it is end-of-life and
  LangChain 1.x requires 3.10+).
- **RAM**: 2 GB minimum; 4 GB+ for HuggingFace embeddings and fine-tuning (notebook 18)
- **Disk**: about 2 GB for the virtual environment, models, vector stores and cache
- **Internet**: required for installation and API calls
- **System packages** (notebook 17 only): Tesseract OCR and Poppler

  ```bash
  brew install tesseract poppler                   # macOS
  sudo apt-get install tesseract-ocr poppler-utils # Debian/Ubuntu
  ```

  On Windows, install the Tesseract and Poppler binaries and add them to `PATH`.

### Accounts

1. **OpenAI** (required): [sign up](https://platform.openai.com/signup),
   [add billing](https://platform.openai.com/account/billing) and
   [create an API key](https://platform.openai.com/api-keys). About $5 of credit is enough for the
   tutorial.
2. **Tavily** (notebooks 08 and 10): web search API key from [tavily.com](https://tavily.com/).
3. **LangSmith** (optional): tracing, from [smith.langchain.com](https://smith.langchain.com/).

## Installation Methods

### Method 1: Standard Installation (Recommended)

```bash
git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
cd langchain-rag-tutorial

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import langchain; print('LangChain', langchain.__version__)"
python -c "import langchain_core; print('langchain-core', langchain_core.__version__)"
python -c "import faiss; print('FAISS OK')"
```

### Method 2: Development Installation

For contributors (adds pytest, ruff, mypy and pre-commit):

```bash
make install-dev
# equivalent to:
pip install -r requirements-dev.txt
pre-commit install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

### Method 3: Conda

```bash
conda create -n langchain-rag python=3.12
conda activate langchain-rag
pip install -r requirements.txt
```

### Method 4: Docker

```bash
docker build -t langchain-rag:latest .
docker run -p 8888:8888 --env-file .env langchain-rag:latest
```

The image uses Python 3.12, includes Tesseract and Poppler, and starts Jupyter on port 8888 as a
non-root user. See [DEPLOYMENT.md](DEPLOYMENT.md#docker)
for Docker Compose.

### Method 5: Google Colab

```python
!git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
%cd langchain-rag-tutorial
!pip install -q -r requirements.txt

from google.colab import userdata
import os
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

Add `OPENAI_API_KEY` in the Colab Secrets panel (left sidebar) first.

## Dependencies

`requirements.txt` groups dependencies by purpose. The key points:

### LangChain 1.x

| Package                    | Constraint       | Purpose                                                                                            |
| -------------------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `langchain`                | `>=1.0`          | Agents (`create_agent`), messages                                                                  |
| `langchain-core`           | `>=1.0`          | Runnables, prompts, documents, message history                                                     |
| `langchain-classic`        | `>=1.0`          | `EnsembleRetriever`, parent-document and multi-vector retrievers, rerankers (notebooks 12, 19, 20) |
| `langchain-openai`         | `>=1.0`          | Chat models and embeddings                                                                         |
| `langchain-text-splitters` | `>=1.0`          | Text splitters (no longer pulled in by `langchain` 1.x)                                            |
| `langchain-huggingface`    | `>=1.0`          | Local embeddings                                                                                   |
| `langchain-tavily`         | `>=0.2`          | Web search tool (notebooks 08 and 10)                                                              |
| `langgraph`                | `>=1.0`          | Graph-based workflows                                                                              |
| `langchain-community`      | `>=0.4.0,<0.4.2` | FAISS vector store, `WebBaseLoader`                                                                |

### Why `langchain-community` is pinned below 0.4.2

`langchain-community` 0.4.2 removed `langchain_community.chat_models.vertexai`, which ragas 0.4.x
imports at module load, so `import ragas` fails (upstream bug
[vibrantlabsai/ragas#2753](https://github.com/vibrantlabsai/ragas/issues/2753)). The upper bound will
be removed once ragas ships a fix.

`langchain-community` is also being sunset upstream
([langchain-ai/langchain-community#674](https://github.com/langchain-ai/langchain-community/issues/674)),
but it remains the only official home of the FAISS integration and `WebBaseLoader` used here.

### Web search

The web search tool is `TavilySearch` from `langchain-tavily`, which replaces the deprecated
`TavilySearchResults` from `langchain_community`:

```bash
pip install langchain-tavily     # already in requirements.txt
```

```python
from langchain_tavily import TavilySearch
```

It requires `TAVILY_API_KEY` in `.env`.

### Other notable dependencies

- **FAISS** (`faiss-cpu`) - vector similarity search
- **sentence-transformers** + **accelerate** - embedding fine-tuning (notebook 18) and the
  cross-encoder reranker (notebooks 12 and 19)
- **rank-bm25** - BM25 keyword retrieval for hybrid search (notebooks 12 and 19)
- **NetworkX** + **python-louvain** - GraphRAG (notebook 15)
- **pandas** - SQL RAG results and RAGAS reports; the SQL notebook uses the standard-library
  `sqlite3` module
- **numexpr** - safe calculator tool for the agent (notebook 10)
- **RAGAS** + **datasets** - evaluation (notebook 16)
- **pillow**, **pytesseract**, **pdf2image** - multimodal RAG (notebook 17)
- **FastAPI**, **uvicorn**, **Streamlit**, **boto3** - deployment templates

## API Key Configuration

### Create the `.env` File

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
OPENAI_API_KEY=sk-proj-your-actual-key-here
TAVILY_API_KEY=tvly-your-key-here      # notebooks 08 and 10
```

Do not put quotes around values. `.env` is loaded by `shared/config.py` from the project root;
restart the Jupyter kernel after changing it.

### Security

`.env` is listed in `.gitignore`:

```bash
grep "^\.env$" .gitignore
```

- Keep keys in `.env` or environment variables; never hardcode them in notebooks.
- Never commit, share or screenshot `.env`.
- Rotate a key immediately if it is exposed.

See [SECURITY.md](../SECURITY.md) for the security policy.

### HuggingFace Embeddings (No Key Required)

Local embeddings run offline after a one-time model download (cached in `~/.cache/huggingface/`).
The model is set by `HF_EMBEDDING_MODEL` (default `BAAI/bge-small-en-v1.5`):

```python
from langchain_huggingface import HuggingFaceEmbeddings
from shared.config import HF_EMBEDDING_MODEL

hf_embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
```

The cross-encoder reranker used by notebooks 12 and 19 (`DEFAULT_RERANKER_MODEL`, default
`cross-encoder/ms-marco-MiniLM-L-6-v2`) is also local: it needs no API key, and the first run downloads about
90 MB into the same cache.

## Configuration Reference

All settings are read by `shared/config.py`; `.env.example` documents each one.

### Environment and Logging

```bash
ENVIRONMENT=dev          # dev, test, prod
DEBUG_MODE=false
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Models

```bash
DEFAULT_MODEL=gpt-4o-mini                   # chat model (notebooks and templates)
DEFAULT_TEMPERATURE=0
DEFAULT_VISION_MODEL=gpt-4o                 # multimodal RAG (notebook 17)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
HF_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
DEFAULT_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2   # local cross-encoder (notebooks 12 and 19)
```

`text-embedding-3-small` has 1536 dimensions; `text-embedding-3-large` has 3072 and is more
accurate but more expensive.

### Retrieval

```bash
DEFAULT_K=4              # documents retrieved (3-5 recommended)
DEFAULT_MMR_FETCH_K=20   # candidates before MMR filtering (4-5x DEFAULT_K)
DEFAULT_MMR_LAMBDA=0.5   # 1.0 = relevance only, 0.0 = diversity only
```

### Text Processing

```bash
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200   # typically 10-20% of chunk size
```

### Display and Miscellaneous

```bash
SECTION_WIDTH=80
PREVIEW_LENGTH=300
TOKENIZERS_PARALLELISM=false
USER_AGENT=LangChain-RAG-Tutorial/1.0
# LANGSMITH_TRACING=true    # requires LANGSMITH_API_KEY
```

### Example Profiles

| Setting                  | Development   | Production    | High accuracy            |
| ------------------------ | ------------- | ------------- | ------------------------ |
| `ENVIRONMENT`            | `dev`         | `prod`        | `prod`                   |
| `LOG_LEVEL`              | `INFO`        | `WARNING`     | `INFO`                   |
| `DEFAULT_MODEL`          | `gpt-4o-mini` | `gpt-4o-mini` | `gpt-4o`                 |
| `OPENAI_EMBEDDING_MODEL` | small         | small         | `text-embedding-3-large` |
| `DEFAULT_K`              | 4             | 3             | 5                        |
| `DEFAULT_MMR_LAMBDA`     | 0.5           | 0.5           | 0.7                      |

## Pre-building Vector Stores

Notebook 02 creates the FAISS vector stores used by later notebooks. To build them without running
the notebook:

```bash
make vector-stores
# equivalent to:
python scripts/build_vector_stores.py
```

Options:

- `--provider {openai,huggingface,all}` - which embeddings to build (default `all`)
- `--version TAG` - write stores to `data/vector_stores/<TAG>/` instead of `data/vector_stores/`

```bash
python scripts/build_vector_stores.py --provider huggingface
python scripts/build_vector_stores.py --version v1
```

## Validation

Check the API key:

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
print(f"API key valid: {len(client.models.list().data)} models available")
```

Then open `notebooks/00_index.ipynb`, which verifies imports, the API key, the `shared` module and
the data directories.

If anything fails, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
