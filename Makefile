.PHONY: help install install-dev test lint format clean vector-stores docker-build docker-run docker-stop

help:
	@echo "LangChain RAG Tutorial - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install dev dependencies and pre-commit hooks"
	@echo ""
	@echo "Quality:"
	@echo "  make test           Run tests"
	@echo "  make lint           Run ruff and mypy"
	@echo "  make format         Format and auto-fix with ruff"
	@echo ""
	@echo "Data:"
	@echo "  make vector-stores  Build shared vector stores"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Start notebooks and API"
	@echo "  make docker-stop    Stop containers"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          Clean cache files"

PY_SRC = shared/ tests/ scripts/ templates/

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest

lint:
	ruff check $(PY_SRC)
	ruff format --check $(PY_SRC)
	mypy shared/ scripts/

format:
	ruff format $(PY_SRC)
	ruff check --fix $(PY_SRC)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov/

vector-stores:
	python scripts/build_vector_stores.py

docker-build:
	docker compose build

docker-run:
	docker compose up -d

docker-stop:
	docker compose down
