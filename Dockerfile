# Multi-stage Dockerfile for LangChain RAG Tutorial

# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
# CPU-only torch (for sentence-transformers): avoids several GB of CUDA wheels
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

LABEL maintainer="LangChain RAG Tutorial"
LABEL description="RAG tutorial notebooks and API template with LangChain"
LABEL version="1.4.0"

# Non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# curl: healthchecks; tesseract/poppler: OCR and PDF images (notebook 17)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

COPY --chown=appuser:appuser shared/ ./shared/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser notebooks/ ./notebooks/
COPY --chown=appuser:appuser .env.example ./.env.example

RUN mkdir -p /app/data/vector_stores /app/data/cache && \
    chown -R appuser:appuser /app/data

USER appuser

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8888

# Probes the default command (Jupyter); the api service overrides it in compose
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -fs http://localhost:8888/api || exit 1

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
