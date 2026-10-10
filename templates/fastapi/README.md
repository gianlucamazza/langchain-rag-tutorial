# FastAPI starter template

Educational FastAPI wrapper around the tutorial RAG chain. Useful for local demos; not a hardened production service (no auth, rate limits, or SLO testing).

## Features

- RESTful API with automatic documentation
- Basic error handling and logging
- CORS configuration
- Health check endpoint
- Request validation with Pydantic
- Async request handlers

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your OpenAI API key
```

### 3. Run Development Server

```bash
python app.py
```

The API will be available at: http://localhost:8000

## API Documentation

Interactive API docs: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc

## Endpoints

### POST /query
Query the RAG system

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
Health check

**Response:**
```json
{
  "status": "healthy",
  "version": "1.2.0",
  "vector_store_loaded": true
}
```

### GET /architectures
List available architectures

## Running the template

### Docker

```bash
docker build -t rag-api .
docker run -p 8000:8000 --env-file .env rag-api
```

### Cloud Deployment

See main documentation for AWS, GCP, Azure deployment guides.
