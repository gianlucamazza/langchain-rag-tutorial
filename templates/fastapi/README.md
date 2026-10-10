# FastAPI Template

Educational FastAPI wrapper around the tutorial RAG chain. Useful for local demos; not a hardened production service (no auth, rate limits, or SLO testing).

## Features

- REST API with automatic OpenAPI documentation
- Request validation with Pydantic
- Error handling and logging
- CORS configuration
- Health check endpoint
- Async request handling

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

The template imports the project's `shared` module, which loads `.env` from the project root:

```bash
cp ../../.env.example ../../.env
# Set OPENAI_API_KEY; optionally set DEFAULT_MODEL (default: gpt-4o-mini)
```

The template loads the OpenAI vector store from `OPENAI_VECTOR_STORE_PATH`
(`data/vector_stores/openai__<OPENAI_EMBEDDING_MODEL>`, default
`openai__text-embedding-3-small`) using `OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)`. Build it first with notebook 02 or `make vector-stores` from the project root.

### 3. Run the Server

```bash
python app.py
# or, with auto-reload
uvicorn app:app --reload
```

The API is served at http://localhost:8000.

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

### POST /query

Query the RAG system.

**Request:**

```json
{
  "query": "What is RAG?",
  "k": 4,
  "architecture": "simple"
}
```

**Response:**

```json
{
  "answer": "RAG is Retrieval-Augmented Generation...",
  "sources": ["doc1.txt", "doc2.txt"],
  "latency_ms": 1234.56,
  "architecture": "simple"
}
```

### GET /health

Health check. Returns the service status, the application version and whether the vector store is
loaded:

```json
{
  "status": "healthy",
  "version": "<app version>",
  "vector_store_loaded": true
}
```

### GET /architectures

Lists the available architectures.

## Configuration

| Variable         | Default       | Description                     |
| ---------------- | ------------- | ------------------------------- |
| `OPENAI_API_KEY` | (required)    | OpenAI API key                  |
| `DEFAULT_MODEL`  | `gpt-4o-mini` | Chat model used for answers     |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model; selects the vector store |

## Compose / further deployment

The Compose `api` service runs this template from the main image (`docker compose up -d api`,
port 8000, health check `GET /health`); see
[docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md#docker-compose). No template-specific Dockerfile is
included. See
[docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md#custom-fastapi-image) for a container recipe and
notes on auth, rate limiting, and monitoring if you take the template beyond a local demo.
