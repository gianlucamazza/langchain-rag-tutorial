# AWS Lambda Template

Serverless deployment template for the RAG system on AWS Lambda.

## Architecture

```text
API Gateway -> Lambda -> RAG chain -> OpenAI
                 |
                 v
          S3 (vector store)
```

The handler is self-contained (it does not import the `shared` module), so only
`lambda_handler.py` goes into the function package; dependencies go into a Lambda layer.

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

Build the store first with notebook 02 or `make vector-stores`, then upload it:

```bash
aws s3 cp data/vector_stores/openai__text-embedding-3-small/ \
    s3://your-bucket/vector_stores/openai__text-embedding-3-small/ \
    --recursive
```

`VECTOR_STORE_KEY` defaults to `vector_stores/openai__<OPENAI_EMBEDDING_MODEL>`, matching the local
layout; set it only if you upload elsewhere. `OPENAI_EMBEDDING_MODEL` must match the model the store
was built with. Without a bucket, the store is read from `/opt/vector_stores/openai__<model>` (Lambda layer).

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
    --environment Variables="{OPENAI_API_KEY=sk-proj-xxx,VECTOR_STORE_BUCKET=your-bucket,VECTOR_STORE_KEY=vector_stores/openai__text-embedding-3-small}"
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

Create a REST or HTTP API and integrate it with the Lambda function.

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

| Variable              | Default                           | Description                    |
| --------------------- | --------------------------------- | ------------------------------ |
| `OPENAI_API_KEY`         | (required)               | OpenAI API key                                   |
| `DEFAULT_MODEL`          | `gpt-4o-mini`            | Chat model used for answers                      |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model; must match the uploaded store   |
| `VECTOR_STORE_BUCKET`    | (empty)                  | S3 bucket holding the store                      |
| `VECTOR_STORE_KEY`       | `vector_stores/openai__<OPENAI_EMBEDDING_MODEL>` | S3 key prefix of the store |

Store the API key in AWS Secrets Manager or encrypted environment variables rather than in plain
text; see [docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md#secrets-management).

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
