# Usage Examples

Practical code patterns built on the `shared` module. Snippets assume the objects from example 1
(`embeddings`, `vectorstore`, `retriever`, `llm`, `chain`) are already defined.

## Quick Examples

### 1. Basic RAG Query

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from shared import RAG_PROMPT_TEMPLATE, VECTOR_STORE_DIR, format_docs, require_vector_store
from shared.config import DEFAULT_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_VECTOR_STORE_PATH

# Load vector store (built by notebook 02 or `make vector-stores`)
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Build RAG chain
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | RAG_PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)

# Query
response = chain.invoke("What is RAG?")
print(response)
```

### 2. Conversational RAG with Memory

```python
from operator import itemgetter

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from shared.prompts import MEMORY_RAG_PROMPT

# One in-memory history per session
store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# Build conversational chain
base_chain = (
    RunnablePassthrough.assign(context=itemgetter("input") | retriever | format_docs)
    | MEMORY_RAG_PROMPT
    | llm
    | StrOutputParser()
)

conversational_chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Multi-turn conversation
response1 = conversational_chain.invoke(
    {"input": "What is RAG?"},
    config={"configurable": {"session_id": "user_123"}}
)

response2 = conversational_chain.invoke(
    {"input": "What are its main components?"},  # References "RAG"
    config={"configurable": {"session_id": "user_123"}}
)
```

### 3. HyDe for Ambiguous Queries

```python
from shared.prompts import HYDE_PROMPT

# Generate hypothetical document
hyde_generator = HYDE_PROMPT | llm | StrOutputParser()

def hyde_retrieve(query: str):
    # Generate hypothetical answer
    hypo_doc = hyde_generator.invoke({"question": query})
    
    # Retrieve based on hypothetical doc
    docs = vectorstore.similarity_search(hypo_doc, k=4)
    return docs

# Use HyDe retrieval
query = "How do I make my system faster?"
docs = hyde_retrieve(query)
context = format_docs(docs)

# Generate final answer
response = (RAG_PROMPT_TEMPLATE | llm | StrOutputParser()).invoke({
    "context": context,
    "input": query
})
```

### 4. Adaptive RAG with Routing

```python
from shared.prompts import COMPLEXITY_CLASSIFIER_PROMPT, HYDE_PROMPT

# Classify query complexity
complexity_classifier = COMPLEXITY_CLASSIFIER_PROMPT | llm | StrOutputParser()

def adaptive_rag(query: str):
    complexity = complexity_classifier.invoke({"query": query}).strip()
    
    if "SIMPLE" in complexity:
        # Use simple similarity search
        docs = vectorstore.similarity_search(query, k=4)
        strategy = "Similarity"
    elif "MEDIUM" in complexity:
        # Use MMR for diversity
        docs = vectorstore.max_marginal_relevance_search(query, k=4)
        strategy = "MMR"
    else:  # COMPLEX
        # Use HyDe for better semantic matching
        hypo_doc = (HYDE_PROMPT | llm | StrOutputParser()).invoke({"question": query})
        docs = vectorstore.similarity_search(hypo_doc, k=4)
        strategy = "HyDe"
    
    context = format_docs(docs)
    response = (RAG_PROMPT_TEMPLATE | llm | StrOutputParser()).invoke({
        "context": context,
        "input": query
    })
    
    return {"response": response, "strategy": strategy}

# Test with different complexities
simple_query = "What is FAISS?"
complex_query = "How can I optimize semantic search latency while maintaining quality?"

result1 = adaptive_rag(simple_query)  # Similarity
result2 = adaptive_rag(complex_query)  # HyDE
```

### 5. Hybrid Search with Reranking

```python
from shared import build_hybrid_retriever, build_reranking_retriever
from shared.loaders import load_and_split

# BM25 needs the same chunks that were indexed in the vector store
_, chunks = load_and_split()

# BM25 + dense, fused with weighted Reciprocal Rank Fusion (up to 2 * k candidates)
hybrid = build_hybrid_retriever(chunks, vectorstore, k=10, bm25_weight=0.5)

# Cross-encoder reranker keeps the best 4 (DEFAULT_RERANKER_MODEL, ~90 MB download on first use)
reranked = build_reranking_retriever(hybrid, top_n=4)

hybrid_chain = (
    {"context": reranked | format_docs, "input": RunnablePassthrough()}
    | RAG_PROMPT_TEMPLATE
    | llm
    | StrOutputParser()
)

response = hybrid_chain.invoke("What does trim_messages do?")
```

## Advanced Patterns

### 6. Custom Document Loader

```python
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

from shared import VECTOR_STORE_DIR, save_vector_store, split_documents

def load_custom_documents(directory: str):
    """Load all .txt and .pdf files from directory."""
    docs = []
    
    for file_path in Path(directory).rglob("*"):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path))
            docs.extend(loader.load())
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs.extend(loader.load())
    
    # Add metadata
    for doc in docs:
        doc.metadata['source_type'] = 'custom'
        doc.metadata['directory'] = directory
    
    return docs

# Use custom loader
custom_docs = load_custom_documents("path/to/your/docs")
chunks = split_documents(custom_docs, chunk_size=1000)

# Create vector store
vectorstore_custom = FAISS.from_documents(chunks, embeddings)
save_vector_store(vectorstore_custom, VECTOR_STORE_DIR / "custom")
```

### 7. Metadata Filtering

```python
# Add rich metadata during loading
def load_with_metadata(urls, category):
    docs = load_langchain_docs(urls)
    for doc in docs:
        doc.metadata['category'] = category
        doc.metadata['indexed_at'] = datetime.now().isoformat()
    return docs

# Load different categories
tutorials = load_with_metadata(tutorial_urls, "tutorial")
api_docs = load_with_metadata(api_urls, "api")

# Create vector store
all_docs = tutorials + api_docs
chunks = split_documents(all_docs)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Filter by metadata
def filtered_retrieve(query: str, category: str):
    all_docs = vectorstore.similarity_search(query, k=20)
    filtered = [doc for doc in all_docs if doc.metadata.get('category') == category]
    return filtered[:4]

# Query specific category
tutorial_docs = filtered_retrieve("How to build RAG?", "tutorial")
api_docs = filtered_retrieve("What is the API for embeddings?", "api")
```

### 8. Batch Processing

```python
def batch_process_queries(queries: list, chain):
    """Process multiple queries efficiently."""
    results = []
    
    for query in queries:
        response = chain.invoke(query)
        results.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
    
    return results

# Process batch
queries = [
    "What is RAG?",
    "How do embeddings work?",
    "What is FAISS?",
]

results = batch_process_queries(queries, chain)

# Save results
import json
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### 9. Cost Tracking

```python
from shared.utils import estimate_tokens, estimate_embedding_cost

def track_costs(query: str, context: str, response: str):
    """Track API costs for monitoring."""
    # Estimate tokens
    input_tokens = estimate_tokens(context + query)
    output_tokens = estimate_tokens(response)
    
    # Calculate costs (GPT-4o-mini pricing)
    input_cost = input_tokens * 0.15 / 1_000_000
    output_cost = output_tokens * 0.60 / 1_000_000
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

# Use with chain
context = format_docs(retriever.invoke(query))
response = chain.invoke(query)
costs = track_costs(query, context, response)

print(f"Total cost: ${costs['total_cost']:.6f}")
print(f"Tokens: {costs['total_tokens']}")
```

### 10. Error Handling and Retry

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def robust_rag_query(query: str, chain):
    """RAG query with automatic retry on failure."""
    try:
        response = chain.invoke(query)
        return {"success": True, "response": response}
    except Exception as e:
        print(f"Error: {e}. Retrying...")
        raise  # Retry via @retry decorator

# Use robust query
result = robust_rag_query("What is RAG?", chain)
```

### 11. Async Parallel Processing

```python
import asyncio
from langchain_core.runnables import RunnablePassthrough

async def async_rag_query(query: str, chain):
    """Async RAG query for parallel processing."""
    response = await chain.ainvoke(query)
    return {"query": query, "response": response}

async def process_queries_parallel(queries: list, chain):
    """Process multiple queries in parallel."""
    tasks = [async_rag_query(q, chain) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

# Use async processing
queries = ["Query 1", "Query 2", "Query 3"]
results = asyncio.run(process_queries_parallel(queries, chain))
```

## Integration Examples

Complete, runnable versions of these integrations are in `templates/` (see
[DEPLOYMENT.md](DEPLOYMENT.md#deployment-templates)).

### FastAPI REST API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Initialize once
vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
retriever = vectorstore.as_retriever()
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
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn app:app --reload
```

### Streamlit UI

```python
import streamlit as st

st.title("RAG Chatbot")

# Initialize (cached)
@st.cache_resource
def load_chain():
    vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
    retriever = vectorstore.as_retriever()
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
    st.write(response)

# Run: streamlit run streamlit_app.py
```

## See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - Quick start
- [API_REFERENCE.md](API_REFERENCE.md) - Full API docs
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design patterns
- [DEPLOYMENT.md](DEPLOYMENT.md) - Starter templates
