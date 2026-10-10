# API Reference

Complete API documentation for the `shared` module - reusable utilities across all notebooks.

## Table of Contents

- [Module Overview](#module-overview)
- [config.py](#configpy) - Configuration and constants
- [utils.py](#utilspy) - Utility functions
- [loaders.py](#loaderspy) - Document loading
- [prompts.py](#promptspy) - Prompt templates
- [retrievers.py](#retrieverspy) - Hybrid search and reranking retrievers

## Module Overview

The `shared` module provides reusable functions, configurations, and prompts to avoid code duplication across notebooks.

**Import everything:**

```python
from shared import *
```

**Import selectively:**

```python
from shared.config import OPENAI_API_KEY, DEFAULT_CHUNK_SIZE
from shared.utils import format_docs, load_vector_store
from shared.loaders import load_langchain_docs
from shared.prompts import RAG_PROMPT_TEMPLATE
from shared.retrievers import build_hybrid_retriever
```

**Version:**

```python
import shared
print(shared.__version__)  # "1.4.0"
```

---

## config.py

Configuration management, API keys, paths, and default parameters.

### Constants

#### `OPENAI_API_KEY`

```python
OPENAI_API_KEY: str
```

OpenAI API key loaded from `.env` file.

**Raises**: Warning if not found.

---

#### `PROJECT_ROOT`

```python
PROJECT_ROOT: Path
```

Absolute path to project root directory.

---

#### `DATA_DIR`

```python
DATA_DIR: Path
```

Path to `data/` directory for generated files.

---

#### `VECTOR_STORE_DIR`

```python
VECTOR_STORE_DIR: Path
```

Path to vector stores directory (`data/vector_stores/`).

---

#### `OPENAI_VECTOR_STORE_PATH` / `HF_VECTOR_STORE_PATH`

```python
OPENAI_VECTOR_STORE_PATH: Path  # data/vector_stores/openai__<OPENAI_EMBEDDING_MODEL>
HF_VECTOR_STORE_PATH: Path      # data/vector_stores/hf__<HF_EMBEDDING_MODEL, "/" -> "__">
```

Vector store locations keyed by embedding model. Defaults: `openai__text-embedding-3-small` and
`hf__BAAI__bge-small-en-v1.5`. Changing `OPENAI_EMBEDDING_MODEL` or `HF_EMBEDDING_MODEL` points to
a different directory, so a stale index built with another model is never loaded. Import from
`shared.config`.

---

#### `CACHE_DIR`

```python
CACHE_DIR: Path
```

Path to cache directory (`data/cache/`).

---

#### `DEFAULT_CHUNK_SIZE`

```python
DEFAULT_CHUNK_SIZE: int = 1000
```

Default text chunk size for splitting.

---

#### `DEFAULT_CHUNK_OVERLAP`

```python
DEFAULT_CHUNK_OVERLAP: int = 200
```

Default overlap between chunks (20% of chunk size).

---

#### `DEFAULT_K`

```python
DEFAULT_K: int = 4
```

Default number of documents to retrieve.

---

#### `DEFAULT_MODEL`

```python
DEFAULT_MODEL: str = "gpt-4o-mini"
```

Default OpenAI model for chat completions.

---

#### `DEFAULT_TEMPERATURE`

```python
DEFAULT_TEMPERATURE: float = 0
```

Default temperature for LLM generation (deterministic).

---

#### `DEFAULT_RERANKER_MODEL`

```python
DEFAULT_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

Local HuggingFace cross-encoder used for reranking (notebooks 12 and 19). Downloaded on first use
(~90 MB, fast on CPU; `BAAI/bge-reranker-base` is stronger but ~5x slower on CPU). Override with the `DEFAULT_RERANKER_MODEL` environment variable. Also reported by
`get_project_info()` as `reranker_model`.

---

### Functions

#### `verify_api_key()`

```python
def verify_api_key() -> bool
```

Validates OpenAI API key.

**Returns**: `True` if valid, `False` otherwise.

**Example:**

```python
from shared.config import verify_api_key

if verify_api_key():
    print("API key is valid!")
```

---

#### `get_project_info()`

```python
def get_project_info() -> dict
```

Returns project metadata.

**Returns**: Dictionary with paths, version, and config.

**Example:**

```python
from shared.config import get_project_info

info = get_project_info()
print(f"Version: {info['version']}")
print(f"Root: {info['root']}")
```

---

## utils.py

Utility functions for document formatting, vector stores, and display.

### Functions

#### `format_docs()`

```python
def format_docs(docs: List[Document]) -> str
```

Formats list of documents as single string for prompt injection.

**Parameters:**

- `docs`: List of LangChain `Document` objects

**Returns**: Concatenated document content with double newlines.

**Example:**

```python
from shared.utils import format_docs

formatted = format_docs(retrieved_docs)
# "Document 1 content\n\nDocument 2 content\n\n..."
```

---

#### `load_vector_store()`

```python
def load_vector_store(
    path: str | Path,
    embeddings: Embeddings,
    verbose: bool = True
) -> FAISS | None
```

Loads FAISS vector store from disk.

**Parameters:**

- `path`: Path to vector store directory
- `embeddings`: Embeddings instance (must match stored embeddings)
- `verbose`: Print loading info

**Returns**: FAISS vector store instance, or `None` if the store is missing or cannot be loaded.
Use `require_vector_store()` when a missing store should be an error.

**Example:**

```python
from shared.utils import load_vector_store
from shared.config import OPENAI_EMBEDDING_MODEL, OPENAI_VECTOR_STORE_PATH
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vectorstore = load_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
```

---

#### `require_vector_store()`

```python
def require_vector_store(path: str | Path, embeddings: Embeddings) -> FAISS
```

Loads a FAISS vector store that must already exist.

**Raises**: `FileNotFoundError` if the store is missing or cannot be loaded; the message explains
how to build it (`make vector-stores` or notebook 02).

---

#### `save_vector_store()`

```python
def save_vector_store(
    vectorstore: FAISS,
    path: str | Path,
    verbose: bool = True
) -> None
```

Saves FAISS vector store to disk.

**Parameters:**

- `vectorstore`: FAISS instance to save
- `path`: Destination path
- `verbose`: Print saving info

**Example:**

```python
from shared.utils import save_vector_store

save_vector_store(vectorstore, "data/vector_stores/my_store")
# Saved to data/vector_stores/my_store/
```

---

#### `print_section_header()`

```python
def print_section_header(title: str, width: int = 80) -> None
```

Prints formatted section header for notebooks.

**Parameters:**

- `title`: Section title
- `width`: Total width in characters

**Example:**

```python
from shared.utils import print_section_header

print_section_header("Document Loading")
# ================================================================================
# DOCUMENT LOADING
# ================================================================================
```

---

#### `print_results()`

```python
def print_results(
    docs: list[Document],
    title: str = "Retrieved Documents",
    max_docs: int | None = None,
    preview_length: int = PREVIEW_LENGTH
) -> None
```

Pretty-prints retrieved documents: source, metadata and a content preview for each.

**Parameters:**

- `docs`: Retrieved documents
- `title`: Title of the results section
- `max_docs`: Maximum number of documents to display (`None` = all)
- `preview_length`: Characters shown per document (default `PREVIEW_LENGTH`, 300)

**Example:**

```python
from shared.utils import print_results

docs = retriever.invoke("What is RAG?")
print_results(docs, "Similarity Search Results", max_docs=3)
```

---

#### `print_comparison_table()`

```python
def print_comparison_table(data: list[list[str]], headers: list[str] | None = None) -> None
```

Prints comparison table (used in benchmarks).

**Parameters:**

- `data`: Table rows; when `headers` is omitted, the first row is used as the header
- `headers`: Optional header row

**Example:**

```python
from shared.utils import print_comparison_table

data = [
    ["Model", "Latency", "Cost"],
    ["GPT-4", "2s", "$0.03"],
    ["GPT-3.5", "1s", "$0.002"]
]
print_comparison_table(data)
```

---

#### `estimate_tokens()`

```python
def estimate_tokens(text: str, model: str = DEFAULT_MODEL) -> int
```

Estimates token count with the tiktoken encoding of `model`. Falls back to `len(text) // 4` if
tiktoken is missing or does not know the model.

**Parameters:**

- `text`: Input text
- `model`: Model whose tokenizer is used

**Returns**: Estimated token count.

**Example:**

```python
from shared.utils import estimate_tokens

tokens = estimate_tokens("Hello, world!")
print(f"Tokens: {tokens}")  # Tokens: 4
```

---

#### `estimate_embedding_cost()`

```python
def estimate_embedding_cost(
    texts: list[str],
    model: str = "text-embedding-3-small",
    cost_per_million: float = 0.02
) -> tuple[int, float]
```

Estimates OpenAI embedding cost for a list of texts.

**Parameters:**

- `texts`: Texts to embed
- `model`: Embedding model name
- `cost_per_million`: USD per million tokens (default: price of `text-embedding-3-small`)

**Returns**: Tuple `(total_tokens, estimated_cost_usd)`.

**Example:**

```python
from shared.utils import estimate_embedding_cost

tokens, cost = estimate_embedding_cost([doc.page_content for doc in chunks])
print(f"Estimated cost: ${cost:.4f} for {tokens:,} tokens")
```

---

## loaders.py

Document loading and text splitting utilities.

### Functions

#### `load_langchain_docs()`

```python
def load_langchain_docs(
    urls: Optional[List[str]] = None,
    add_metadata: bool = True,
    verbose: bool = True
) -> List[Document]
```

Loads LangChain documentation from web URLs.

**Parameters:**

- `urls`: List of URLs (default: preset LangChain docs)
- `add_metadata`: Add source_type, process_date, domain metadata
- `verbose`: Print loading progress

**Returns**: List of Document objects.

**Example:**

```python
from shared.loaders import load_langchain_docs

docs = load_langchain_docs()
print(f"Loaded {len(docs)} documents")
```

---

#### `split_documents()`

```python
def split_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    verbose: bool = True
) -> List[Document]
```

Splits documents into chunks using RecursiveCharacterTextSplitter.

**Parameters:**

- `docs`: List of documents to split
- `chunk_size`: Target chunk size
- `chunk_overlap`: Overlap between chunks
- `verbose`: Print splitting info

**Returns**: List of chunked documents.

**Example:**

```python
from shared.loaders import split_documents

chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
print(f"Split into {len(chunks)} chunks")
```

---

#### `compare_splitting_strategies()`

```python
def compare_splitting_strategies(
    docs: list[Document],
    strategies: list[tuple[int, int]],
    verbose: bool = True
) -> dict
```

Compares different splitting strategies.

**Parameters:**

- `docs`: Documents to split
- `strategies`: List of (chunk_size, overlap) tuples
- `verbose`: Print the comparison table

**Returns**: Dictionary with strategy comparison results.

**Example:**

```python
from shared.loaders import compare_splitting_strategies

strategies = [(1000, 200), (500, 100), (2000, 400)]
results = compare_splitting_strategies(docs, strategies)
```

---

#### `load_and_split()`

```python
def load_and_split(
    urls: list[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    verbose: bool = True
) -> tuple[list[Document], list[Document]]
```

Convenience function: load and split in one call.

**Parameters:**

- `urls`: URLs to load (defaults to `DEFAULT_LANGCHAIN_URLS`)
- `chunk_size`: Chunk size
- `chunk_overlap`: Overlap
- `verbose`: Print status messages

**Returns**: Tuple `(original_docs, chunks)`: the loaded documents and their chunks.

**Example:**

```python
from shared.loaders import load_and_split

docs, chunks = load_and_split(chunk_size=1000)
_, chunks = load_and_split()  # chunks only
```

---

## prompts.py

Prompt templates for all RAG architectures.

### Templates

#### `RAG_PROMPT_TEMPLATE`

```python
RAG_PROMPT_TEMPLATE: ChatPromptTemplate
```

Basic RAG prompt with context injection.

**Variables**: `context`, `input`

**Example:**

```python
from shared.prompts import RAG_PROMPT_TEMPLATE

chain = RAG_PROMPT_TEMPLATE | llm | StrOutputParser()
response = chain.invoke({"context": context, "input": query})
```

---

#### `RAG_WITH_METADATA_PROMPT`

```python
RAG_WITH_METADATA_PROMPT: ChatPromptTemplate
```

RAG prompt with metadata citation.

**Variables**: `context`, `input`

---

#### `MEMORY_RAG_PROMPT`

```python
MEMORY_RAG_PROMPT: ChatPromptTemplate
```

Conversational RAG with chat history.

**Variables**: `context`, `chat_history`, `input`

---

#### `HYDE_PROMPT`

```python
HYDE_PROMPT: ChatPromptTemplate
```

Hypothetical document generation for HyDe.

**Variables**: `question`

---

#### `COMPLEXITY_CLASSIFIER_PROMPT`

```python
COMPLEXITY_CLASSIFIER_PROMPT: ChatPromptTemplate
```

Query complexity classification (SIMPLE/MEDIUM/COMPLEX).

**Variables**: `query`

---

#### `ADAPTIVE_RAG_PROMPT`

```python
ADAPTIVE_RAG_PROMPT: ChatPromptTemplate
```

Adaptive RAG with strategy indication.

**Variables**: `context`, `input`, `strategy`

---

#### `RELEVANCE_GRADER_PROMPT`

```python
RELEVANCE_GRADER_PROMPT: ChatPromptTemplate
```

Document relevance grading (yes/no).

**Variables**: `question`, `document`

---

#### `CRAG_PROMPT`

```python
CRAG_PROMPT: ChatPromptTemplate
```

Corrective RAG with web search indication.

**Variables**: `context`, `input`

---

#### `RETRIEVAL_NEED_PROMPT`

```python
RETRIEVAL_NEED_PROMPT: ChatPromptTemplate
```

Self-RAG retrieval need classification.

**Variables**: `query`

---

#### `SELF_CRITIQUE_PROMPT`

```python
SELF_CRITIQUE_PROMPT: ChatPromptTemplate
```

Self-RAG response critique.

**Variables**: `query`, `context`, `response`

---

#### `CITATION_CHECK_PROMPT`

```python
CITATION_CHECK_PROMPT: ChatPromptTemplate
```

Self-RAG citation validation.

**Variables**: `context`, `response`

---

#### `MULTI_QUERY_PROMPT`

```python
MULTI_QUERY_PROMPT: ChatPromptTemplate
```

Branched RAG multi-query generation.

**Variables**: `question`

---

#### `REACT_AGENT_PROMPT`

```python
REACT_AGENT_PROMPT: ChatPromptTemplate
```

Agentic RAG ReAct agent prompt.

**Variables**: Defined by LangChain agent framework.

---

### Contextual, Fusion, SQL and GraphRAG Prompts

#### `DOCUMENT_SUMMARY_PROMPT`

```python
DOCUMENT_SUMMARY_PROMPT: ChatPromptTemplate
```

Contextual RAG document summarization.

**Variables**: `document_content`

---

#### `CONTEXTUAL_CHUNK_PROMPT`

```python
CONTEXTUAL_CHUNK_PROMPT: ChatPromptTemplate
```

Contextual RAG chunk contextualization (Anthropic technique).

**Variables**: `document_summary`, `chunk_content`

---

#### `CONTEXTUAL_RAG_ANSWER_PROMPT`

```python
CONTEXTUAL_RAG_ANSWER_PROMPT: ChatPromptTemplate
```

Contextual RAG answer generation.

**Variables**: `context`, `input`

---

#### `FUSION_QUERY_GENERATION_PROMPT`

```python
FUSION_QUERY_GENERATION_PROMPT: ChatPromptTemplate
```

Fusion RAG multi-perspective query generation.

**Variables**: `question`

---

#### `FUSION_RAG_ANSWER_PROMPT`

```python
FUSION_RAG_ANSWER_PROMPT: ChatPromptTemplate
```

Fusion RAG answer generation with RRF context.

**Variables**: `context`, `input`

---

#### `SQL_SCHEMA_SUMMARY_PROMPT`

```python
SQL_SCHEMA_SUMMARY_PROMPT: ChatPromptTemplate
```

SQL RAG schema summarization.

**Variables**: `schema_info`

---

#### `TEXT_TO_SQL_PROMPT`

```python
TEXT_TO_SQL_PROMPT: ChatPromptTemplate
```

SQL RAG natural language to SQL conversion.

**Variables**: `schema`, `question`

---

#### `SQL_RESULTS_INTERPRETATION_PROMPT`

```python
SQL_RESULTS_INTERPRETATION_PROMPT: ChatPromptTemplate
```

SQL RAG results interpretation.

**Variables**: `question`, `sql_query`, `results`

---

#### `SQL_ERROR_RECOVERY_PROMPT`

```python
SQL_ERROR_RECOVERY_PROMPT: ChatPromptTemplate
```

SQL RAG error recovery and query correction.

**Variables**: `question`, `failed_query`, `error_message`

---

#### `ENTITY_EXTRACTION_PROMPT`

```python
ENTITY_EXTRACTION_PROMPT: ChatPromptTemplate
```

GraphRAG entity extraction.

**Variables**: `text`

---

#### `RELATIONSHIP_EXTRACTION_PROMPT`

```python
RELATIONSHIP_EXTRACTION_PROMPT: ChatPromptTemplate
```

GraphRAG relationship extraction between entities.

**Variables**: `text`, `entities`

---

#### `ENTITY_DISAMBIGUATION_PROMPT`

```python
ENTITY_DISAMBIGUATION_PROMPT: ChatPromptTemplate
```

GraphRAG entity disambiguation and normalization.

**Variables**: `entities`

---

#### `GRAPH_SUMMARIZATION_PROMPT`

```python
GRAPH_SUMMARIZATION_PROMPT: ChatPromptTemplate
```

GraphRAG subgraph summarization.

**Variables**: `subgraph_info`

---

#### `GRAPHRAG_ANSWER_PROMPT`

```python
GRAPHRAG_ANSWER_PROMPT: ChatPromptTemplate
```

GraphRAG final answer generation with graph context.

**Variables**: `graph_context`, `input`

---

### Multi-Vector Retrieval Prompts

#### `CHUNK_SUMMARY_PROMPT`

```python
CHUNK_SUMMARY_PROMPT: ChatPromptTemplate
```

Short, retrieval-oriented summary of a chunk, indexed by `MultiVectorRetriever` (notebook 20).
Registered as `"chunk_summary"` in `get_prompt_by_name()`.

**Variables**: `chunk`

---

#### `HYPOTHETICAL_QUESTIONS_PROMPT`

```python
HYPOTHETICAL_QUESTIONS_PROMPT: ChatPromptTemplate
```

Generates questions a chunk answers, indexed by `MultiVectorRetriever` (notebook 20). Use with
`with_structured_output` to get a list of questions. Registered as `"hypothetical_questions"` in
`get_prompt_by_name()`.

**Variables**: `chunk`, `num_questions`

---

### Utility Functions

#### `get_prompt_by_name()`

```python
def get_prompt_by_name(name: str) -> ChatPromptTemplate
```

Retrieves prompt template by name.

**Parameters:**

- `name`: Prompt name (e.g., "RAG", "HYDE", "CRAG")

**Returns**: ChatPromptTemplate instance.

**Raises**: `ValueError` if name not found.

**Example:**

```python
from shared.prompts import get_prompt_by_name

prompt = get_prompt_by_name("HYDE")
```

---

## retrievers.py

Builders for hybrid (BM25 + dense) search and cross-encoder reranking, used by notebooks 12 and 19. The retriever classes come from `langchain-classic` and `langchain-community`; BM25 requires
`rank-bm25`.

### Functions

#### `bm25_tokenize()`

```python
def bm25_tokenize(text: str) -> list[str]
```

Lowercases and splits on non-word characters, keeping `snake_case` identifiers whole. The
default BM25 tokenizer in LangChain splits on whitespace only, so `max_retries=2,` would never
match the query `max_retries`.

**Example:**

```python
bm25_tokenize("Set max_retries=2.")  # ["set", "max_retries", "2"]
```

---

#### `build_bm25_retriever()`

```python
def build_bm25_retriever(
    docs: Sequence[Document],
    k: int = DEFAULT_K,
    tokenizer: Callable[[str], list[str]] = bm25_tokenize
) -> BM25Retriever
```

Builds a keyword (BM25) retriever over the given chunks. The index is kept in memory (no
persistence).

**Parameters:**

- `docs`: Chunks to index
- `k`: Number of documents to return
- `tokenizer`: Applied to both documents and queries

**Returns**: `BM25Retriever` instance.

---

#### `build_hybrid_retriever()`

```python
def build_hybrid_retriever(
    docs: Sequence[Document],
    vectorstore: VectorStore,
    k: int = DEFAULT_K,
    bm25_weight: float = 0.5,
    rrf_c: int = 60
) -> EnsembleRetriever
```

Combines BM25 and dense retrieval with weighted Reciprocal Rank Fusion. Both retrievers must index
the same chunks, since RRF matches documents by content.

**Parameters:**

- `docs`: Chunks indexed in the vector store (used to build the BM25 index)
- `vectorstore`: Dense vector store over the same chunks
- `k`: Documents fetched from each retriever before fusion
- `bm25_weight`: Weight of BM25 in [0, 1]; the dense retriever gets `1 - bm25_weight`
- `rrf_c`: RRF constant (60, as in the original paper)

**Returns**: `EnsembleRetriever` returning the fused ranking (up to `2 * k` unique documents).

**Raises**: `ValueError` if `bm25_weight` is not in [0, 1].

**Example:**

```python
from shared.retrievers import build_hybrid_retriever

hybrid = build_hybrid_retriever(chunks, vectorstore, k=10, bm25_weight=0.5)
docs = hybrid.invoke("What does max_retries control?")
```

---

#### `build_reranking_retriever()`

```python
def build_reranking_retriever(
    base_retriever: BaseRetriever,
    top_n: int = DEFAULT_K,
    model_name: str = DEFAULT_RERANKER_MODEL,
    cross_encoder: BaseCrossEncoder | None = None
) -> ContextualCompressionRetriever
```

Reranks a retriever's candidates with a cross-encoder and keeps the `top_n`. The base retriever
should over-fetch (e.g. 20 candidates) so the reranker has something to choose from.

**Parameters:**

- `base_retriever`: First-stage retriever producing candidates
- `top_n`: Documents kept after reranking
- `model_name`: HuggingFace cross-encoder model (downloaded on first use)
- `cross_encoder`: Pre-built cross-encoder; overrides `model_name` (reuse it across calls, loading
  is the slow part)

**Returns**: `ContextualCompressionRetriever` wrapping a `CrossEncoderReranker`.

**Example:**

```python
from shared.retrievers import build_hybrid_retriever, build_reranking_retriever

hybrid = build_hybrid_retriever(chunks, vectorstore, k=10)
reranked = build_reranking_retriever(hybrid, top_n=4)
docs = reranked.invoke("What does max_retries control?")
```

---

## Usage Examples

### Complete RAG Pipeline

```python
from shared import *
from shared.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, OPENAI_EMBEDDING_MODEL
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load and split documents
docs = load_langchain_docs()
chunks = split_documents(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save for reuse
save_vector_store(vectorstore, VECTOR_STORE_DIR / "my_store")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_K})

# Build RAG chain
llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
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

### Load Existing Vector Store

```python
from shared import *
from shared.config import OPENAI_EMBEDDING_MODEL, OPENAI_VECTOR_STORE_PATH
from langchain_openai import OpenAIEmbeddings

# Load pre-built vector store
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)

# Use immediately
retriever = vectorstore.as_retriever()
```

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
- [EXAMPLES.md](EXAMPLES.md) - Usage patterns
- [CONTRIBUTING.md](CONTRIBUTING.md) - Extend shared module
- [CHANGELOG.md](CHANGELOG.md) - Version history
