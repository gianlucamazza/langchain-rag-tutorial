# Performance notes

Teaching estimates for this tutorial — **not** results from a checked-in, reproducible benchmark script. Latency, cost, and quality figures below are order-of-magnitude notes (token arithmetic + typical API wait) so you know what to measure on your own data. Notebook 11 prints the same kind of qualitative table; it does not time live chains.

## Execution Times

### First Run vs Subsequent Runs

| Notebook | First Run | Subsequent | Reason |
|---|---|---|---|
| 01_setup_and_basics | 3-5 min | 30s | Document loading + chunking |
| 02_embeddings_comparison | 8-10 min | 1 min | HF model download + embedding creation |
| 03_simple_rag | 2-3 min | 30s | Vector store loading + queries |
| 04_rag_with_memory | 2 min | 30s | Conversation history management |
| 05_branched_rag | 5-8 min | 2 min | Multiple LLM calls (3 sub-queries + 1 gen) |
| 06_hyde | 4-6 min | 1.5 min | Hypothetical doc generation + retrieval |
| 07_adaptive_rag | 3-5 min | 1 min | Classification + routing |
| 08_corrective_rag | 10-15 min | 3 min | Relevance grading + web search |
| 09_self_rag | 10-20 min | 4 min | Multiple iterations + self-critique |
| 10_agentic_rag | 20-40 min | 8 min | Agent loop with multiple tool calls |
| 11_comparison | 5-8 min | 2 min | Prints qualitative comparison tables (no live timing harness) |
| 12_contextual_rag | 8-12 min | 2 min | Context generation (one-time) + retrieval |
| 13_fusion_rag | 5-8 min | 2 min | Multiple query perspectives + RRF ranking |
| 14_sql_rag | 5-8 min | 1.5 min | Chinook DB setup + SQL generation |
| 15_graphrag | 10-15 min | 3 min | Entity extraction + graph construction |
| 16_evaluation_ragas | 15-20 min | 5 min | Evaluation dataset + metrics computation |
| 17_multimodal_rag | 10-20 min | 3 min | Vision/OCR setup + image queries |
| 18_finetuning_embeddings | 20-40 min | 5 min | Contrastive fine-tune (local) |
| 19_hybrid_search_reranking | 8-15 min | 2 min | BM25 + dense + first-run reranker download (~90 MB) |
| 20_parent_multivector_retrieval | 10-20 min | 3 min | Parent/multi-vector index (LLM summaries) |

**Key Insight:** First run includes model downloads, vector store creation, and database setup. Subsequent runs use cached data.

## Query latency (illustrative)

### Per-query response times (teaching estimates, not measured here)

| Architecture | Latency | API Calls | Token Usage |
|---|---|---|---|
| Simple RAG | 1-2s | 1 LLM call | ~1,500 tokens |
| Memory RAG | 2-3s | 1 LLM call | ~2,000 tokens (+ history) |
| Branched RAG | 5-8s | 4 LLM calls | ~6,000 tokens |
| HyDe | 4-6s | 2 LLM calls | ~3,000 tokens |
| Contextual RAG | 2-3s | 1 LLM call (+ upfront context) | ~1,800 tokens |
| Fusion RAG | 5-8s | 5-6 LLM calls | ~7,000 tokens |
| Hybrid + Rerank | 2-4s | 1 LLM call (+ local reranker) | ~1,500 tokens |
| Parent-Document / Multi-Vector | 2-3s | 1 LLM call (+ indexing-time LLM calls for multi-vector) | ~1,500-4,000 tokens (parents are larger) |
| Adaptive RAG | Variable | 2-3 LLM calls | 2,000-6,000 tokens (depends on route) |
| SQL RAG | 2-5s | 2-3 LLM calls | ~2,500 tokens |
| CRAG | 10-15s | 5-6 LLM calls | ~8,000 tokens |
| Self-RAG | 10-20s | 4-6 LLM calls | ~10,000 tokens |
| GraphRAG | 3-8s | 3-4 LLM calls + graph ops | ~4,500 tokens |
| Agentic RAG | 20-40s | 5-10 LLM calls | ~15,000 tokens |

**Factors Affecting Latency:**

- Network latency to OpenAI API
- Number of retrieved documents (k)
- LLM model speed
- Number of iterations (Self-RAG, Agentic)
- Cross-encoder reranking (notebooks 12, 19): about 1-4 s per query on a laptop CPU with the
  default MiniLM model and 20 candidates, ~5x more with `BAAI/bge-reranker-base`; latency grows with
  the number and length of candidates; load the model once and reuse it

## Cost Estimates

### OpenAI API Costs (GPT-4o-mini)

**Pricing:**

- Input: $0.15 / 1M tokens
- Output: $0.60 / 1M tokens

**Per-Query Costs:**

| Architecture | Input Tokens | Output Tokens | Cost per Query |
|---|---|---|---|
| Simple RAG | 1,200 | 300 | $0.00036 |
| Memory RAG | 1,800 | 400 | $0.00051 |
| Branched RAG | 4,800 | 1,200 | $0.00144 |
| HyDe | 2,400 | 600 | $0.00072 |
| Contextual RAG | 1,500 | 350 | $0.00043 |
| Fusion RAG | 5,500 | 1,400 | $0.00167 |
| Adaptive RAG | 2,000-5,000 | 500-1,000 | $0.00045-$0.00135 |
| SQL RAG | 2,000 | 500 | $0.00060 |
| CRAG | 6,500 | 1,500 | $0.00188 |
| Self-RAG | 8,000 | 2,000 | $0.00240 |
| GraphRAG | 3,500 | 1,000 | $0.00113 |
| Agentic RAG | 12,000 | 3,000 | $0.00360 |

**Monthly Cost Estimates (1000 queries/month):**

- Simple RAG: $0.36/month
- Adaptive RAG: $0.90/month (optimized)
- Agentic RAG: $3.60/month

### Embedding Costs

**OpenAI text-embedding-3-small:**

- $0.02 / 1M tokens
- 10,000 documents (~1M tokens): ~$0.02
- **One-time cost** (embeddings are cached)

**HuggingFace (Local):**

- **FREE** (runs locally)
- First download: ~90MB (one-time)
- Slower than OpenAI (CPU-bound)

## Resource Requirements

### System Requirements

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| RAM | 2GB | 4GB+ | HF embeddings need more |
| CPU | 2 cores | 4+ cores | For parallel processing |
| Disk | 1.5GB | 3GB+ | Dependencies + models |
| Network | 1 Mbps | 10+ Mbps | For API calls |

### Disk Space Breakdown

```text
venv/                 ~900 MB   Python dependencies
.cache/huggingface/   ~90 MB    Sentence-transformers model
.cache/huggingface/   ~90 MB    Reranker model (ms-marco-MiniLM-L-6-v2, notebooks 12, 19)
data/vector_stores/   1-5 MB    FAISS indexes
data/chinook.db       984 KB    Chinook SQLite database (SQL RAG)
notebooks/            ~700 KB   21 Jupyter notebooks
shared/               ~150 KB   Shared Python modules
Total:                ~1.2 GB
```

## Optimization Strategies

### 1. Vector Store Caching

**Problem:** Re-embedding on every run wastes time and money.

**Solution:**

```python
from shared.config import OPENAI_EMBEDDING_MODEL, OPENAI_VECTOR_STORE_PATH

embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

# Create once (notebook 02, or `make vector-stores`)
vectorstore = FAISS.from_documents(chunks, embeddings)
save_vector_store(vectorstore, OPENAI_VECTOR_STORE_PATH)

# Reuse everywhere (notebooks 03-18)
vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
```

**Savings:**

- Time: 8 minutes to 5 seconds
- Cost: $0.02 to $0.00

### 2. Reduce Retrieved Documents (k)

**Problem:** More documents = more tokens = higher cost + latency.

**Solution:**

```python
# Default: k=4
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Optimize for speed: k=2
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
```

**Impact:**

- Latency: -20-30%
- Cost: -20-30%
- Quality: -5-10% (depends on query)

### 3. Use Adaptive RAG for Mixed Workloads

**Problem:** Always using complex architectures wastes resources.

**Solution:**

- SIMPLE queries: similarity search (fast)
- MEDIUM queries: MMR (balanced)
- COMPLEX queries: HyDE (slower but more accurate)

**Savings:**

- Average cost: -40%
- Average latency: -50%
- Quality: Maintained

### 4. Batch Processing

**Problem:** Processing queries one-by-one is inefficient.

**Solution:**

```python
# Batch embed
texts = [doc.page_content for doc in docs]
embeddings_list = embeddings.embed_documents(texts)

# Batch retrieve
queries = ["query1", "query2", "query3"]
results = [retriever.invoke(q) for q in queries]
```

**Savings:**

- Latency: -30% (API overhead reduced)

### 5. Use Local Embeddings

**Problem:** OpenAI API calls add latency and cost.

**Solution:**

```python
from langchain_huggingface import HuggingFaceEmbeddings

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Trade-offs:**

- Cost: FREE
- Latency: +50% (CPU-bound)
- Quality: -10% (384d vs 1536d)

## Profiling

### Measure Query Time

```python
import time

start = time.time()
response = chain.invoke(query)
latency = time.time() - start

print(f"Latency: {latency:.2f}s")
```

### Measure Token Usage

```python
from shared.utils import estimate_tokens

input_tokens = estimate_tokens(context + query)
output_tokens = estimate_tokens(response)

print(f"Input: {input_tokens} tokens")
print(f"Output: {output_tokens} tokens")
print(f"Total: {input_tokens + output_tokens} tokens")
```

### Measure Cost

```python
from shared.utils import estimate_embedding_cost

# Embedding cost (one-time)
embedding_cost = estimate_embedding_cost(total_tokens)

# LLM cost (per query)
llm_cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

print(f"Embedding: ${embedding_cost:.4f} (one-time)")
print(f"LLM: ${llm_cost:.6f} (per query)")
```

## Benchmarking Results

### Quality vs Speed Trade-off

```
Relative quality (illustrative, unmeasured)
                |                    * Agentic RAG (~30s)
                |                  * Self-RAG (~15s)
                |                * GraphRAG (~6s)
                |                * CRAG (~12s)
                |              * Fusion RAG (~7s)
                |            * SQL RAG (~4s; high if SQL valid)
                |            * HyDe (~5s)
                |          * Branched RAG (~6s)
                |       * Adaptive RAG (variable)
                |       * Contextual RAG (~2.5s)
                |     * Memory RAG (~2.5s)
                |   * Simple RAG (~2s)
                |_________________________________
                         Latency (seconds)
```

This sketch is unlabeled teaching intuition, not a scored benchmark. SQL RAG quality is high when the generated SQL is valid and the schema is in scope; it is not guaranteed.

**Key Insight:** more elaborate loops usually cost a lot more latency than they gain in quality. Specialized architectures (Contextual, Fusion, SQL, GraphRAG) are worth trying when they match the question type.

## Performance Tips

**Do:**

- Cache vector stores (notebook 02 or `make vector-stores`)
- Use Adaptive RAG for mixed workloads
- Start with Simple RAG and upgrade only if needed
- Monitor token usage
- Use HuggingFace embeddings for demos

**Avoid:**

- Re-creating vector stores on every run
- Agentic RAG for simple queries
- Retrieving k=10 documents or more without a reason
- Skipping caching
- Larger models than needed for tutorial runs (`DEFAULT_MODEL=gpt-4o-mini` is the default)

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [FAQ.md](FAQ.md) - Performance FAQs
