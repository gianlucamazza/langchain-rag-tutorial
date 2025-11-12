.PHONY: install test lint format clean docker-build docker-run help

help:
	@echo "LangChain RAG Tutorial - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies"
	@echo "  make install-dev    Install dev dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-run    Run Docker container"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Clean cache files"
	@echo "  make vector-stores Build vector stores"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=shared --cov-report=html

lint:
	flake8 shared/ tests/
	mypy shared/
	black --check shared/ tests/

format:
	black shared/ tests/
	isort shared/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/

docker-build:
	docker build -t langchain-rag:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

vector-stores:
	python scripts/build_vector_stores.py
