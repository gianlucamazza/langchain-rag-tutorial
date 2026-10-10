# Troubleshooting

Common issues and solutions for LangChain RAG Tutorial.

## Table of Contents

- [Installation Issues](#installation-issues)
- [API Key Problems](#api-key-problems)
- [Import Errors](#import-errors)
- [Vector Store Issues](#vector-store-issues)
- [Performance Problems](#performance-problems)
- [Jupyter Issues](#jupyter-issues)
- [Advanced Issues](#advanced-issues)

## Installation Issues

### Python Version Mismatch

**Problem:**

```
ERROR: Package 'langchain' requires a different Python: 3.9.x not in '>=3.10'
```

**Solution:** the tutorial supports Python 3.10-3.13. Recreate the virtual environment with a
supported interpreter:

```bash
python --version
python3.12 -m venv venv
```

### Dependency Conflicts

**Problem:**

```
ERROR: Cannot install langchain and langchain-community
```

**Solution:**

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

If you manage dependencies yourself, keep all LangChain packages on the 1.x line
(`langchain`, `langchain-core`, `langchain-openai`, `langchain-text-splitters`, `langgraph`) and
`langchain-community` at `>=0.4.0,<0.4.2`.

### RAGAS Import Fails With `chat_models.vertexai`

**Problem:**

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

**Cause:** `langchain-community` 0.4.2 removed this module, but ragas 0.4.x imports it at module
load ([vibrantlabsai/ragas#2753](https://github.com/vibrantlabsai/ragas/issues/2753)).

**Solution:** install the pinned version from `requirements.txt`:

```bash
pip install "langchain-community>=0.4.0,<0.4.2"
```

The pin will be lifted once ragas ships a fix. Note that `langchain-community` is being sunset
upstream ([langchain-ai/langchain-community#674](https://github.com/langchain-ai/langchain-community/issues/674));
it is kept only because it is still the official home of the FAISS integration and `WebBaseLoader`.

### FAISS Installation Failed

**Problem (Windows):**

```
error: Microsoft Visual C++ 14.0 is required
```

**Solution:**

```bash
# Use pre-built wheels
pip install faiss-cpu --no-cache-dir

# Or use conda
conda install -c conda-forge faiss-cpu
```

**Problem (macOS M1/M2):**

```
ImportError: DLL load failed
```

**Solution:**

```bash
# Use conda on Apple Silicon
conda install -c conda-forge faiss-cpu
```

## API Key Problems

### API Key Not Found

**Problem:**

```
openai.OpenAIError: The api_key client option must be set
```

**Solutions:**

**1. Verify .env file:**

```bash
# Check if .env exists
ls -la .env

# Verify content (don't share output!)
cat .env
# Should show: OPENAI_API_KEY=sk-proj-...
```

**2. Restart Jupyter kernel:**

- In Jupyter: `Kernel > Restart`
- Environment variables load on kernel start

**3. Reload .env manually:**

```python
from dotenv import load_dotenv
load_dotenv(override=True)  # Force reload
```

### Invalid API Key

**Problem:**

```
AuthenticationError: Incorrect API key
```

**Solutions:**

1. Verify key at [OpenAI Platform](https://platform.openai.com/api-keys)
2. Check for extra spaces in .env
3. Ensure no quotes around key in .env:

   ```bash
   # Correct
   OPENAI_API_KEY=sk-proj-abc123

   # Wrong
   OPENAI_API_KEY="sk-proj-abc123"
   ```

### Rate Limit Exceeded

**Problem:**

```
RateLimitError: You exceeded your current quota
```

**Solutions:**

1. Check billing at [OpenAI Billing](https://platform.openai.com/account/billing)
2. Add payment method if needed
3. Wait 1 minute between retries
4. Use tier-appropriate rate limits

## Import Errors

### ModuleNotFoundError: langchain

**Problem:**

```python
ModuleNotFoundError: No module named 'langchain'
```

**Solutions:**

```bash
# 1. Verify venv is activated
which python  # Should point to venv/bin/python

# 2. Reinstall in venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Install Jupyter kernel
python -m ipykernel install --user --name=venv
# Restart Jupyter, select 'venv' kernel
```

### ImportError: shared

**Problem:**

```python
ImportError: No module named 'shared'
```

**Solution:**

```python
# Add project root to path
import sys
sys.path.append('../..')  # From notebooks/fundamentals/
# or
sys.path.append('..')      # From notebooks/

from shared import *
```

### ModuleNotFoundError: faiss

**Problem:**

```python
ImportError: DLL load failed while importing _swigfaiss
```

**Solution:**

```bash
# Reinstall faiss
pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir
```

## Vector Store Issues

### Vector Store Not Found

**Problem:**

```
FileNotFoundError: Vector store not found
```

**Solution:**

```bash
# Run notebook 02 first to create vector stores
jupyter notebook notebooks/fundamentals/02_embeddings_comparison.ipynb

# Verify creation
ls -la data/vector_stores/
# Should show openai__<OPENAI_EMBEDDING_MODEL>/ and hf__<HF_EMBEDDING_MODEL with / -> __>/,
# e.g. openai__text-embedding-3-small/ and hf__BAAI__bge-small-en-v1.5/
# Changing either model points to a new directory: rebuild the stores
# Alternatively: make vector-stores
```

### Dimension Mismatch

**Problem:**

```
RuntimeError: Embedding dimension mismatch
```

**Solution:**

```python
# Ensure same embeddings for save/load
# Save:
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
save_vector_store(vectorstore, path)

# Load:
embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)  # Same model!
vectorstore = load_vector_store(path, embeddings)
```

### Dangerous Deserialization Warning

**Problem:**

```
Warning: allow_dangerous_deserialization=True
```

**Solution:**

```python
# This is expected for local FAISS stores
# Safe if you created the vector store yourself
vectorstore = FAISS.load_local(
    path, 
    embeddings,
    allow_dangerous_deserialization=True  # OK for local files
)
```

## Performance Problems

### Slow First Run

**Problem:** Notebooks take 5-10 minutes on first run.

**Explanation:** Expected behavior

- HuggingFace model downloads (~90MB; plus ~90MB for the reranker in notebooks 12 and 19)
- Vector store creation (embedding all documents)
- Package initialization

**Solution:**

- Subsequent runs: ~30 seconds (uses cached vector stores)
- Be patient on first run!

### Out of Memory

**Problem:**

```
MemoryError: Unable to allocate array
```

**Solutions:**

```python
# Reduce chunk size
chunks = split_documents(docs, chunk_size=500)  # Instead of 1000

# Reduce k (retrieved documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})  # Instead of 4

# Process in batches
for i in range(0, len(docs), batch_size):
    batch = docs[i:i+batch_size]
    # Process batch
```

### Slow Queries

**Problem:** RAG queries taking > 5s

**Solutions:**

1. **Use cached vector stores** (don't recreate each time)
2. **Reduce k**: Retrieve fewer documents
3. **Optimize chunk size**: Test 500-2000 range
4. **Use faster embeddings**: HuggingFace (local) vs OpenAI (API call)

## Jupyter Issues

### Kernel Not Found

**Problem:** "No kernel named 'venv'"

**Solution:**

```bash
# Install kernel
python -m ipykernel install --user --name=venv --display-name="Python (LangChain RAG)"

# Restart Jupyter
# Select: Kernel > Change Kernel > Python (LangChain RAG)
```

### Kernel Keeps Dying

**Problem:** Kernel crashes repeatedly

**Solutions:**

```bash
# 1. Increase memory limit
# Add to jupyter config:
c.NotebookApp.max_buffer_size = 500000000

# 2. Restart kernel between notebooks
# In Jupyter: Kernel > Restart & Clear Output

# 3. Check system resources
htop  # or Activity Monitor on macOS
```

### Cell Output Not Showing

**Problem:** Cells execute but no output

**Solution:**

```python
# Explicitly print
result = chain.invoke(query)
print(result)  # Force output

# Or display
from IPython.display import display
display(result)
```

## Advanced Issues

### SSL Certificate Errors

**Problem:**

```
urllib.error.URLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**Solution (macOS):**

```bash
# Run Python certificate installer
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Solution (General):**

```bash
pip install --upgrade certifi
```

### Web Search Fails (Notebooks 08 and 10)

**Problem:**

```
ModuleNotFoundError: No module named 'langchain_tavily'
```

**Solution:**

```bash
pip install langchain-tavily
```

**Problem:** authentication or missing key errors from Tavily.

**Solution:** set `TAVILY_API_KEY` in `.env` and restart the kernel. Use the current tool:

```python
from langchain_tavily import TavilySearch

search = TavilySearch(max_results=3)
```

`TavilySearchResults` from `langchain_community` is deprecated and no longer used.

### Calculator Tool Fails (Notebook 10)

**Problem:**

```
ModuleNotFoundError: No module named 'numexpr'
```

**Solution:**

```bash
pip install numexpr
```

### Deprecated Imports After Upgrading to LangChain 1.x

**Problem:** `ImportError` or deprecation warnings for older import paths.

**Solution:** use the LangChain 1.x locations:

| Old import                                                                  | Current import                                            |
| --------------------------------------------------------------------------- | --------------------------------------------------------- |
| `langchain_community.chat_message_histories.ChatMessageHistory`             | `langchain_core.chat_history.InMemoryChatMessageHistory`  |
| `langchain_community.tools.tavily_search.TavilySearchResults`               | `langchain_tavily.TavilySearch`                           |
| `langchain.text_splitter.RecursiveCharacterTextSplitter`                    | `langchain_text_splitters.RecursiveCharacterTextSplitter` |
| `langchain.schema.Document`                                                 | `langchain_core.documents.Document`                       |
| `langchain.retrievers.EnsembleRetriever` (and other `langchain.retrievers`) | `langchain_classic.retrievers.EnsembleRetriever`          |

### LangGraph Issues (Agentic RAG)

**Problem:**

```
ModuleNotFoundError: No module named 'langgraph'
```

**Solution:**

```bash
pip install "langgraph>=1.0"
```

### Tokenizer Parallelism Warning

**Problem:**

```
huggingface/tokenizers: The current process just got forked...
```

**Solution:**

```bash
# Add to .env
echo "TOKENIZERS_PARALLELISM=false" >> .env

# Restart Jupyter kernel
```

### OpenAI Timeout

**Problem:**

```
OpenAIError: Request timed out
```

**Solution:**

```python
from langchain_openai import ChatOpenAI

# Increase timeout
llm = ChatOpenAI(
    model="gpt-4o-mini",
    timeout=60.0,  # 60 seconds
    max_retries=3
)
```

## Getting Help

If issues persist:

1. **Check logs:** Jupyter terminal shows detailed errors
2. **Search issues:** [GitHub Issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
3. **Create issue:** Include:
   - Error message (full traceback)
   - Python version (`python --version`)
   - OS and version
   - Steps to reproduce
4. **FAQ:** [FAQ.md](FAQ.md)
5. **Community:** [Discussions](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)

## Prevention Tips

**Recommended:**

- Activate the virtual environment before running anything
- Build vector stores first (notebook 02 or `make vector-stores`)
- Restart the kernel after changing `.env` or upgrading packages
- Install from `requirements.txt` rather than individual packages
- Use `.env.example` as the template for `.env`

**Common mistakes:**

- Hardcoding API keys
- Loading a vector store with different embeddings than it was built with
- Upgrading `langchain-community` past the pinned range
