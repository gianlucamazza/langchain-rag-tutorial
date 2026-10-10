# Deployment Guide

Starter notes for the tutorial templates (FastAPI, Streamlit, Lambda, Docker Compose). These are teaching apps, not a production rollout playbook.

## Table of Contents

- [Pre-deployment checklist](#production-checklist)
- [Docker](#docker)
- [Deployment Templates](#deployment-templates)
- [Custom FastAPI Image](#custom-fastapi-image)
- [Configuration Management](#configuration-management)
- [Scaling Strategies](#scaling-strategies)
- [Monitoring](#monitoring)
- [Security](#security)
- [Cost Optimization](#cost-optimization)

## Production Checklist

**Code and data**

- [ ] Shared module tested (`make test`) and linted (`make lint`)
- [ ] Error handling and logging configured
- [ ] Vector stores pre-built and versioned (`make vector-stores`)

**Performance**

- [ ] Latency requirements met and cost per query calculated
      (see [PERFORMANCE.md](PERFORMANCE.md))
- [ ] Caching strategy defined
- [ ] Load testing completed

**Security**

- [ ] No hardcoded secrets; API keys in environment variables or a secrets manager
- [ ] Input validation, rate limiting and authentication in place
- [ ] HTTPS enabled

**Operations**

- [ ] Health checks and monitoring configured
- [ ] Backup, rollback and incident response plans documented

## Docker

### Image

The repository `Dockerfile` is a multi-stage build on `python:3.12-slim`:

- Python dependencies are installed in a builder stage and copied into the runtime stage.
- The runtime stage installs `tesseract-ocr` and `poppler-utils`, so the multimodal notebook (17)
  works in the container.
- `shared/`, `scripts/`, `templates/`, `notebooks/` and `.env.example` are copied into `/app`.
- Everything runs as a non-root user (`appuser`, UID 1000); Jupyter does not use `--allow-root`.
- The default command starts Jupyter on port **8888**; the health check probes Jupyter's `/api`
  endpoint on that port.

```bash
docker build -t langchain-rag:latest .
docker run -p 8888:8888 --env-file .env langchain-rag:latest
```

### Docker Compose

`docker-compose.yml` defines two services built from the same image. Both read `.env` through
`env_file`, so create it first:

```bash
cp .env.example .env     # set OPENAI_API_KEY
make vector-stores       # the API needs a pre-built OpenAI vector store
make docker-build        # docker compose build
make docker-run          # docker compose up -d
make docker-stop         # docker compose down
```

| Service     | Port | Description                                                                  |
| ----------- | ---- | ---------------------------------------------------------------------------- |
| `notebooks` | 8888 | Jupyter; mounts `./data`, `./notebooks` and `./shared`                       |
| `api`       | 8000 | FastAPI template (`uvicorn templates.fastapi.app:app`); mounts `./data`; health check `GET /health` |

No cache or monitoring services are included. Redis caching and Prometheus metrics appear below
only as optional patterns you implement yourself.

## Deployment Templates

Three templates live in `templates/`. All of them read the chat model from the `DEFAULT_MODEL`
environment variable (default `gpt-4o-mini`) and expect a pre-built OpenAI vector store
at `OPENAI_VECTOR_STORE_PATH` (`data/vector_stores/openai__<OPENAI_EMBEDDING_MODEL>`, default
`openai__text-embedding-3-small`), created by notebook 02 or `make vector-stores`. They build
OpenAI embeddings with `OPENAI_EMBEDDING_MODEL`, which must match the model the store was built with.

| Template                                      | Best for                 | Details                                   |
| --------------------------------------------- | ------------------------ | ----------------------------------------- |
| [FastAPI](../templates/fastapi/README.md)     | REST APIs, microservices | Pydantic validation, CORS, `/health`      |
| [Streamlit](../templates/streamlit/README.md) | Internal tools, demos    | Web UI with sources and metrics           |
| [AWS Lambda](../templates/lambda/README.md)   | Serverless, low traffic  | Vector store loaded from S3 on cold start |

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing to
   `templates/streamlit/streamlit_app.py`.
3. Add `OPENAI_API_KEY` (and optionally `DEFAULT_MODEL`) as secrets.
4. Deploy.

## Custom FastAPI Image

The Compose `api` service already runs the FastAPI template from the main image. For a smaller,
API-only image, a minimal recipe:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY data/vector_stores/ ./data/vector_stores/
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# app.py
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

from shared import RAG_PROMPT_TEMPLATE, format_docs, require_vector_store
from shared.config import DEFAULT_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_VECTOR_STORE_PATH

chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
    vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )
    yield


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    question: str


@app.post("/query")
async def query_rag(query: Query):
    try:
        return {"answer": await chain.ainvoke(query.question)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}
```

```bash
docker build -t rag-api .
docker run -p 8000:8000 --env-file .env rag-api
```

## Configuration Management

### Environment Variables

The application settings documented in [INSTALLATION.md](INSTALLATION.md#configuration-reference)
apply in production as well. A typical production `.env`:

```bash
OPENAI_API_KEY=sk-proj-...
ENVIRONMENT=prod
LOG_LEVEL=WARNING
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_K=3
```

### Secrets Management

**AWS Secrets Manager:**

```python
import json

import boto3


def get_secret(secret_name: str) -> dict:
    client = boto3.client("secretsmanager")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


OPENAI_API_KEY = get_secret("rag-api-secrets")["OPENAI_API_KEY"]
```

**Azure Key Vault:**

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

client = SecretClient(
    vault_url="https://your-vault.vault.azure.net/", credential=DefaultAzureCredential()
)
OPENAI_API_KEY = client.get_secret("OPENAI-API-KEY").value
```

## Scaling Strategies

### Horizontal Scaling

Run several API replicas behind a load balancer:

```yaml
services:
  rag-api:
    image: rag-api:latest
    deploy:
      replicas: 3
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-api
```

### Response Caching With Redis (Optional Pattern)

Not included in the repository or the Compose file; add a Redis service and the `redis` package
yourself if you need it.

```python
import hashlib
import json

import redis

redis_client = redis.Redis(host="localhost", port=6379, db=0)


def cached_rag_query(query: str, chain, ttl: int = 3600):
    cache_key = f"rag:{hashlib.sha256(query.encode()).hexdigest()}"
    if cached := redis_client.get(cache_key):
        return json.loads(cached)
    response = chain.invoke(query)
    redis_client.setex(cache_key, ttl, json.dumps(response))
    return response
```

### Versioned Vector Stores

Build vector stores ahead of time and keep one directory per version, so a deployment can pin a
specific index and roll back:

```bash
python scripts/build_vector_stores.py --version v1.0                      # -> data/vector_stores/v1.0/
python scripts/build_vector_stores.py --provider openai --version v1.1    # OpenAI only
```

Without `--version`, stores are written to `data/vector_stores/`. Copy or mount the chosen
directory into the container and point the application at it.

## Monitoring

### Logging

```python
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def monitored_rag_query(query: str, chain):
    start = time.perf_counter()
    try:
        response = chain.invoke(query)
        logger.info("Query succeeded in %.2fs", time.perf_counter() - start)
        return response
    except Exception:
        logger.exception("Query failed")
        raise
```

### Metrics With Prometheus (Optional Pattern)

Not included in the repository; requires the `prometheus_client` package and your own Prometheus
setup.

```python
from prometheus_client import Counter, Histogram, start_http_server

query_counter = Counter("rag_queries_total", "Total RAG queries")
query_latency = Histogram("rag_query_latency_seconds", "RAG query latency")
error_counter = Counter("rag_errors_total", "Total RAG errors")


@query_latency.time()
def monitored_query(query: str, chain):
    query_counter.inc()
    try:
        return chain.invoke(query)
    except Exception:
        error_counter.inc()
        raise


start_http_server(9100)  # metrics endpoint for Prometheus to scrape
```

For LangChain-level tracing, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.

### Health Checks

Report readiness only when dependencies are available:

```python
@app.get("/health")
async def health_check():
    checks = {
        "vectorstore": check_vectorstore(),
        "openai_api": check_openai_api(),
    }
    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    raise HTTPException(status_code=503, detail=checks)
```

## Security

See [SECURITY.md](../SECURITY.md) for the security policy.

### Input Validation

```python
from pydantic import BaseModel, field_validator


class Query(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long")
        return v
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.post("/query")
@limiter.limit("10/minute")
async def query_rag(request: Request, query: Query): ...
```

### API Authentication

```python
import os

from fastapi import Depends, Header, HTTPException

API_KEYS = set(filter(None, os.getenv("API_KEYS", "").split(",")))


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.post("/query", dependencies=[Depends(verify_api_key)])
async def query_rag(query: Query): ...
```

## Cost Optimization

| Strategy                      | Typical saving                |
| ----------------------------- | ----------------------------- |
| Response caching (1-24 h TTL) | 80-90% on repeated queries    |
| Adaptive RAG routing          | 40-60% average cost reduction |
| Batching requests             | 20-30% less API overhead      |
| Smaller `DEFAULT_K`           | 20-30% fewer tokens per query |

Track cost per query and alert on outliers:

```python
from shared.utils import estimate_tokens


def track_costs(query: str, response: str) -> float:
    # gpt-4o-mini pricing: $0.15 / 1M input tokens, $0.60 / 1M output tokens
    return (estimate_tokens(query) * 0.15 + estimate_tokens(response) * 0.60) / 1_000_000
```

See [PERFORMANCE.md](PERFORMANCE.md) for detailed cost figures.

## See Also

- [PERFORMANCE.md](PERFORMANCE.md) - Optimization strategies
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [EXAMPLES.md](EXAMPLES.md) - Integration patterns
