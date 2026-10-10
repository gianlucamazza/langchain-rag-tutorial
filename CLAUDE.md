# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A LangChain 1.x RAG tutorial: 21 Jupyter notebooks (`00_index` + 01-20) teaching RAG architectures,
backed by a reusable `shared/` package and three deployment templates (FastAPI, Streamlit, AWS
Lambda). Supported Python: 3.10-3.13 (`.python-version` is 3.12; ruff targets py310, so no 3.11+
syntax). OpenAI is the default provider; Tavily (notebooks 08, 10) and Tesseract/Poppler
(notebook 17) are optional extras.

## Commands

```bash
make install-dev        # requirements-dev.txt + pre-commit hooks
make test               # pytest (coverage on shared/ is always on via pytest.ini)
make lint               # ruff check + ruff format --check + mypy shared/ scripts/
make format             # ruff format + ruff check --fix
make vector-stores      # build FAISS stores (needs OPENAI_API_KEY and network)
docker compose up -d    # Jupyter on :8888, FastAPI template on :8000

pytest tests/test_utils.py::test_name   # single test
pytest -m "not slow"                    # markers: slow, integration (--strict-markers)
python scripts/build_vector_stores.py --provider huggingface --version v1   # versioned store
```

Lint/format scope is `shared/ tests/ scripts/ templates/`. Notebooks are deliberately excluded
from ruff (teaching code kept readable over lint-clean). mypy only covers `shared/` and
`scripts/`. CI mirrors `make test` (3.10-3.13 matrix) and `make lint` (3.12).

## Architecture

**`shared/` is the single source of truth** for config, prompts, loaders and vector-store I/O;
notebooks and templates import from it rather than redefining things.

- `shared/config.py` runs side effects at import: loads `.env`, copies API keys into the env
  vars LangChain expects (e.g. `LANGSMITH_API_KEY` → `LANGCHAIN_API_KEY`, tracing flags), and
  creates `data/vector_stores/` and `data/cache/`. All defaults (model, chunk size, k, MMR params)
  are env-overridable.
- `shared/__init__.py` must set warning filters **before** importing langchain; keep the
  `# noqa: E402` imports after that block. Public API is re-exported via `__all__` — update it
  when adding exports, and bump `__version__` together with README/CHANGELOG.
- `shared/prompts.py` holds all prompt templates (E501 ignored on purpose — don't rewrap prompt
  text, it changes the prompt).

**Vector stores are keyed by embedding model.** `OPENAI_VECTOR_STORE_PATH` /
`HF_VECTOR_STORE_PATH` embed the model name in the directory, so changing the embedding model
never loads a stale index. Stores are built by `scripts/build_vector_stores.py` (or notebook 02)
from the URLs in `DEFAULT_LANGCHAIN_URLS`, and consumed by notebooks 03-10 and the templates via
`require_vector_store()`, which raises `FileNotFoundError` with build instructions (tests assert
on that message). `load_vector_store()` returns `None` instead of raising. FAISS loading uses
`allow_dangerous_deserialization=True` (local pickles only).

**Import path conventions:** there is no installable package. Notebooks do
`sys.path.append('../..')`, templates and `scripts/` prepend the project root to `sys.path`,
and `tests/conftest.py` does the same. The FastAPI app runs from the repo root as
`uvicorn templates.fastapi.app:app`.

**The Lambda template is intentionally self-contained**: it does not import `shared/`, duplicates
its prompt/`format_docs`, reads config from env vars, and downloads the FAISS store from S3
(`VECTOR_STORE_BUCKET`/`VECTOR_STORE_KEY`) on cold start into module-level globals. Keep it
deployable without the rest of the repo.

## Testing conventions

Tests never hit the network: use `DeterministicFakeEmbedding` and `FakeListChatModel` from
`langchain_core`, and inject them into module globals (see `tests/test_lambda_handler.py`, which
loads the handler via `importlib` from its file path). Shared fixtures (`sample_documents`, etc.)
live in `tests/conftest.py`.

## Data

`data/` is gitignored: `data/vector_stores/` (built locally), `data/cache/`, and
`data/chinook.db` (SQLite sample DB used by notebook 14).
