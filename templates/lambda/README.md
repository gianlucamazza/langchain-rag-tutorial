# AWS Lambda Template

Serverless deployment template for RAG on AWS Lambda.

## Architecture

```
API Gateway → Lambda → RAG Chain → OpenAI
                  ↓
             S3 (Vector Store)
```

## Deployment Steps

### 1. Prepare Lambda Layer

Create a layer with dependencies:

```bash
mkdir -p layer/python
pip install -r requirements.txt -t layer/python/
cd layer
zip -r layer.zip python/
```

### 2. Upload Vector Store to S3

```bash
aws s3 cp data/vector_stores/openai_embeddings/ \
    s3://your-bucket/vector_stores/openai_embeddings/ \
    --recursive
```

### 3. Create Lambda Function

```bash
zip function.zip lambda_handler.py

aws lambda create-function \
    --function-name rag-api \
    --runtime python3.11 \
    --role arn:aws:iam::ACCOUNT_ID:role/lambda-execution-role \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb://function.zip \
    --timeout 60 \
    --memory-size 512 \
    --environment Variables="{OPENAI_API_KEY=sk-proj-xxx,VECTOR_STORE_BUCKET=your-bucket}"
```

### 4. Attach Layer

```bash
aws lambda publish-layer-version \
    --layer-name rag-dependencies \
    --zip-file fileb://layer.zip \
    --compatible-runtimes python3.11

aws lambda update-function-configuration \
    --function-name rag-api \
    --layers arn:aws:lambda:REGION:ACCOUNT_ID:layer:rag-dependencies:1
```

### 5. Create API Gateway

Create REST API and integrate with Lambda.

## Testing

### Local Test

```bash
python lambda_handler.py
```

### Lambda Test

```bash
aws lambda invoke \
    --function-name rag-api \
    --payload '{"query": "What is RAG?"}' \
    response.json

cat response.json
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `VECTOR_STORE_BUCKET`: S3 bucket with vector store
- `VECTOR_STORE_KEY`: S3 key prefix for vector store

## Performance

- **Cold start**: ~3-5s (layer download + initialization)
- **Warm start**: ~1-2s (cached initialization)
- **Memory**: 512MB recommended
- **Timeout**: 60s

## Cost Optimization

- Use provisioned concurrency for critical endpoints
- Enable caching in API Gateway
- Use S3 for vector store instead of bundling
- Monitor and optimize memory allocation
