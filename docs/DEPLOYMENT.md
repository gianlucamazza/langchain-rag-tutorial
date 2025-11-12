# Deployment Guide

Production deployment strategies for LangChain RAG applications.

## Table of Contents

- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Deployment Options](#deployment-options)
- [Configuration Management](#configuration-management)
- [Scaling Strategies](#scaling-strategies)
- [Monitoring](#monitoring)
- [Security](#security)
- [Cost Optimization](#cost-optimization)

## Pre-Deployment Checklist

Before deploying to production:

### Code Quality

- [ ] All notebooks execute without errors
- [ ] Shared module functions tested
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] No hardcoded secrets

### Performance

- [ ] Vector stores pre-built and cached
- [ ] Latency requirements met
- [ ] Cost per query calculated
- [ ] Rate limiting implemented
- [ ] Caching strategy defined

### Security

- [ ] API keys in environment variables
- [ ] Secrets management configured
- [ ] Input validation implemented
- [ ] Output sanitization added
- [ ] HTTPS enabled

### Documentation

- [ ] API documentation complete
- [ ] Deployment runbook created
- [ ] Incident response plan ready
- [ ] Contact information updated

## Deployment Options

### Option 1: FastAPI + Docker

**Best for:** REST API, microservices

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY shared/ ./shared/
COPY data/vector_stores/ ./data/vector_stores/
COPY app.py .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from shared import *
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

app = FastAPI()

# Initialize (on startup)
@app.on_event("startup")
async def startup():
    global chain
    embeddings = OpenAIEmbeddings()
    vectorstore = load_vector_store("data/vector_stores/openai", embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

class Query(BaseModel):
    question: str

@app.post("/query")
async def query_rag(query: Query):
    try:
        response = chain.invoke(query.question)
        return {"answer": response, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Deploy:**

```bash
# Build
docker build -t rag-api .

# Run locally
docker run -p 8000:8000 --env-file .env rag-api

# Push to registry
docker tag rag-api your-registry/rag-api:latest
docker push your-registry/rag-api:latest
```

### Option 2: Streamlit Cloud

**Best for:** Internal tools, demos

```python
# streamlit_app.py
import streamlit as st
from shared import *
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

st.title("RAG Chatbot")

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = load_vector_store("data/vector_stores/openai", embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | RAG_PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

chain = load_chain()

# Chat interface
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Thinking..."):
        response = chain.invoke(query)
    st.success(response)
```

**Deploy to Streamlit Cloud:**

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Add secrets in dashboard (OPENAI_API_KEY)
4. Deploy

### Option 3: AWS Lambda + API Gateway

**Best for:** Serverless, low traffic

```python
# lambda_handler.py
import json
import os
from shared import *

# Initialize outside handler (cold start optimization)
embeddings = OpenAIEmbeddings()
vectorstore = load_vector_store("/tmp/vector_stores/openai", embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | RAG_PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        query = body['question']
        
        response = chain.invoke(query)
        
        return {
            'statusCode': 200,
            'body': json.dump({'answer': response})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**Deploy:**

```bash
# Package
pip install -r requirements.txt -t package/
cp -r shared package/
cp lambda_handler.py package/

# Create deployment package
cd package && zip -r ../deployment.zip . && cd ..

# Upload to AWS Lambda
aws lambda create-function \
  --function-name rag-api \
  --runtime python3.10 \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://deployment.zip \
  --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --environment Variables={OPENAI_API_KEY=sk-proj-...} \
  --timeout 30 \
  --memory-size 1024
```

## Configuration Management

### Environment Variables

```bash
# .env.production
OPENAI_API_KEY=sk-proj-...
ENVIRONMENT=production
LOG_LEVEL=INFO
CACHE_DIR=/app/cache
VECTOR_STORE_DIR=/app/data/vector_stores
MAX_RETRIES=3
TIMEOUT=30
```

### Secrets Management

**AWS Secrets Manager:**

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Use in app
secrets = get_secret('rag-api-secrets')
OPENAI_API_KEY = secrets['OPENAI_API_KEY']
```

**Azure Key Vault:**

```python
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(vault_url="https://your-vault.vault.azure.net/", credential=credential)

OPENAI_API_KEY = client.get_secret("OPENAI-API-KEY").value
```

## Scaling Strategies

### Horizontal Scaling

**Load Balancer + Multiple Instances:**

```yaml
# docker-compose.yml
version: '3.8'
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

### Caching Strategy

**Redis for Response Caching:**

```python
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_rag_query(query: str, chain, ttl=3600):
    # Generate cache key
    cache_key = f"rag:{hashlib.md5(query.encode()).hexdigest()}"
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Query if not cached
    response = chain.invoke(query)
    
    # Cache result
    redis_client.setex(cache_key, ttl, json.dumps(response))
    
    return response
```

### Vector Store Optimization

**Pre-build and version vector stores:**

```bash
# Build script
python scripts/build_vector_stores.py --version v1.0

# Deploy
docker build --build-arg VECTOR_STORE_VERSION=v1.0 -t rag-api .
```

## Monitoring

### Logging

```python
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def monitored_rag_query(query: str, chain):
    start = datetime.now()
    logger.info(f"Query received: {query}")
    
    try:
        response = chain.invoke(query)
        latency = (datetime.now() - start).total_seconds()
        
        logger.info(f"Query successful. Latency: {latency:.2f}s")
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
```

### Metrics Collection

**Prometheus + Grafana:**

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
query_latency = Histogram('rag_query_latency_seconds', 'RAG query latency')
error_counter = Counter('rag_errors_total', 'Total RAG errors')

@query_latency.time()
def monitored_query(query: str, chain):
    query_counter.inc()
    try:
        return chain.invoke(query)
    except Exception as e:
        error_counter.inc()
        raise

# Start metrics server
start_http_server(9090)
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    checks = {
        "vectorstore": check_vectorstore(),
        "openai_api": check_openai_api(),
        "memory": check_memory_usage(),
    }
    
    if all(checks.values()):
        return {"status": "healthy", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail=checks)
```

## Security

### Input Validation

```python
from pydantic import BaseModel, validator

class Query(BaseModel):
    question: str
    
    @validator('question')
    def validate_question(cls, v):
        if len(v) > 1000:
            raise ValueError('Query too long')
        if not v.strip():
            raise ValueError('Query cannot be empty')
        # Add more validation
        return v
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query_rag(query: Query):
    # ...
```

### API Authentication

```python
from fastapi import Header, HTTPException

API_KEYS = set(os.getenv('API_KEYS', '').split(','))

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.post("/query", dependencies=[Depends(verify_api_key)])
async def query_rag(query: Query):
    # ...
```

## Cost Optimization

### 1. Implement Caching

- Cache responses for 1-24 hours
- Savings: 80-90% for repeated queries

### 2. Use Adaptive RAG

- Route simple queries to cheaper strategies
- Savings: 40-60% average cost reduction

### 3. Batch Processing

- Process multiple queries together
- Savings: 20-30% API overhead reduction

### 4. Monitor and Alert

```python
# Cost tracking
def track_costs(query, response):
    input_cost = estimate_tokens(query) * 0.15 / 1_000_000
    output_cost = estimate_tokens(response) * 0.60 / 1_000_000
    total_cost = input_cost + output_cost
    
    # Send to monitoring
    cost_metric.observe(total_cost)
    
    # Alert if exceeds threshold
    if total_cost > COST_THRESHOLD:
        send_alert(f"High cost query: ${total_cost}")
```

## Production Checklist

- [ ] Environment variables configured
- [ ] Secrets management implemented
- [ ] Logging enabled
- [ ] Monitoring dashboard created
- [ ] Health checks configured
- [ ] Rate limiting implemented
- [ ] Caching enabled
- [ ] Error handling robust
- [ ] Load testing completed
- [ ] Backup strategy defined
- [ ] Rollback plan ready
- [ ] Documentation updated

## See Also

- [PERFORMANCE.md](PERFORMANCE.md) - Optimization strategies
- [SECURITY.md](SECURITY.md) - Security best practices
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [EXAMPLES.md](EXAMPLES.md) - Integration patterns
