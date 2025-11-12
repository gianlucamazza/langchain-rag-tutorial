# API Reference

Complete API documentation for the `shared` module - reusable utilities across all notebooks.

## Table of Contents

- [Module Overview](#module-overview)
- [config.py](#configpy) - Configuration and constants
- [utils.py](#utilspy) - Utility functions
- [loaders.py](#loaderspy) - Document loading
- [prompts.py](#promptspy) - Prompt templates

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
```

**Version:**

```python
import shared
print(shared.__version__)  # "1.0.0"
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
    path: Union[str, Path],
    embeddings: Embeddings,
    verbose: bool = True
) -> FAISS
```

Loads FAISS vector store from disk.

**Parameters:**

- `path`: Path to vector store directory
- `embeddings`: Embeddings instance (must match stored embeddings)
- `verbose`: Print loading info

**Returns**: FAISS vector store instance.

**Raises**: `FileNotFoundError` if path doesn't exist.

**Example:**

```python
from shared.utils import load_vector_store
from shared.config import OPENAI_VECTOR_STORE_PATH
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = load_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)
```

---

#### `save_vector_store()`

```python
def save_vector_store(
    vectorstore: FAISS,
    path: Union[str, Path],
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
def print_results(results: dict, max_length: int = 200) -> None
```

Pretty-prints RAG chain results.

**Parameters:**

- `results`: Dictionary with `input`, `context`, `answer` keys
- `max_length`: Max length for context preview

**Example:**

```python
from shared.utils import print_results

results = chain.invoke({"input": "What is RAG?"})
print_results(results)
```

---

#### `print_comparison_table()`

```python
def print_comparison_table(data: List[List[str]]) -> None
```

Prints comparison table (used in benchmarks).

**Parameters:**

- `data`: 2D list, first row is header

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
def estimate_tokens(text: str) -> int
```

Estimates token count using tiktoken (cl100k_base).

**Parameters:**

- `text`: Input text

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
    num_tokens: int,
    model: str = "text-embedding-3-small"
) -> float
```

Estimates OpenAI embedding cost.

**Parameters:**

- `num_tokens`: Number of tokens
- `model`: Embedding model name

**Returns**: Estimated cost in USD.

**Example:**

```python
from shared.utils import estimate_embedding_cost

cost = estimate_embedding_cost(10000)
print(f"Cost: ${cost:.4f}")  # Cost: $0.0001
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
    docs: List[Document],
    strategies: List[Tuple[int, int]]
) -> dict
```

Compares different splitting strategies.

**Parameters:**

- `docs`: Documents to split
- `strategies`: List of (chunk_size, overlap) tuples

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
    urls: Optional[List[str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]
```

Convenience function: load and split in one call.

**Parameters:**

- `urls`: URLs to load
- `chunk_size`: Chunk size
- `chunk_overlap`: Overlap

**Returns**: List of chunked documents.

**Example:**

```python
from shared.loaders import load_and_split

chunks = load_and_split(chunk_size=1000)
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

## Usage Examples

### Complete RAG Pipeline

```python
from shared import *
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load and split documents
docs = load_langchain_docs()
chunks = split_documents(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
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
from langchain_openai import OpenAIEmbeddings

# Load pre-built vector store
embeddings = OpenAIEmbeddings()
vectorstore = load_vector_store(VECTOR_STORE_DIR / "openai_embeddings", embeddings)

# Use immediately
retriever = vectorstore.as_retriever()
```

---

## Version History

- **v1.0.0** (2024-11-12): Initial release
  - Core utilities (config, utils, loaders, prompts)
  - 13 prompt templates
  - Vector store persistence
  - Cost estimation utilities

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions
- [EXAMPLES.md](EXAMPLES.md) - Usage patterns
- [CONTRIBUTING.md](CONTRIBUTING.md) - Extend shared module
