# Installation Guide

Complete installation and setup instructions for LangChain RAG Tutorial.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [API Key Configuration](#api-key-configuration)
- [Validation](#validation)
- [Alternative Setups](#alternative-setups)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before installing, ensure you have:

### System Requirements

- **Python**: 3.9 or higher (3.10+ recommended)
- **RAM**: Minimum 2GB available
  - 4GB+ recommended for HuggingFace embeddings
  - 8GB+ for optimal performance with multiple architectures
- **Disk Space**: ~1.5GB
  - 500MB for dependencies
  - ~90MB for HuggingFace model (optional)
  - ~500MB for virtual environment
  - Remaining for vector stores and cache
- **Internet**: Required for initial setup and OpenAI API calls

### Required Accounts

1. **OpenAI Account** (Required)
   - Sign up: <https://platform.openai.com/signup>
   - Add billing: <https://platform.openai.com/account/billing>
   - Get API key: <https://platform.openai.com/api-keys>
   - Minimum credit: $5 recommended for tutorial completion

2. **GitHub Account** (Optional)
   - For cloning, issues, and contributions
   - Alternative: Download ZIP from repository

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository

```bash
# Using Git
git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
cd langchain-rag-tutorial

# Or download ZIP
# Unzip and navigate to folder
cd langchain-rag-tutorial-main
```

#### Step 2: Create Virtual Environment

**macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Verify activation:**

```bash
which python  # Should show path to venv/bin/python
# or on Windows
where python  # Should show path to venv\Scripts\python
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip first (recommended)
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

**Expected output:**

```
Successfully installed langchain-0.1.0+ openai-1.12.0+ ...
```

#### Step 4: Verify Installation

```bash
# Check LangChain version
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"

# Check OpenAI SDK
python -c "import openai; print(f'OpenAI SDK: {openai.__version__}')"

# Check FAISS
python -c "import faiss; print('FAISS: OK')"
```

### Method 2: Development Installation

For contributors or advanced users:

```bash
# Clone with dev dependencies
git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
cd langchain-rag-tutorial

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt  # If available
```

### Method 3: Google Colab

No local installation required!

```python
# In Colab notebook
!git clone https://github.com/gianlucamazza/langchain-rag-tutorial.git
%cd langchain-rag-tutorial

# Install dependencies
!pip install -q -r requirements.txt

# Configure API key via Colab secrets
from google.colab import userdata
import os
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
```

**Add secret in Colab:**

1. Click 🔑 icon (Secrets) in left sidebar
2. Add new secret: `OPENAI_API_KEY`
3. Paste your OpenAI API key

## API Key Configuration

### Required: OpenAI API Key

#### Step 1: Get Your API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Click "+ Create new secret key"
3. Name it (e.g., "langchain-rag-tutorial")
4. **Copy immediately** - you won't see it again!
5. Store securely (password manager recommended)

#### Step 2: Create `.env` File

**Option A: From Template**

```bash
cp .env.example .env
```

Then edit `.env` with your key:

```bash
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

**Option B: Command Line**

```bash
echo "OPENAI_API_KEY=sk-proj-your-actual-key-here" > .env
```

**Option C: Manual Creation**

1. Create file named `.env` in project root
2. Add single line: `OPENAI_API_KEY=sk-proj-...`
3. Save (no quotes needed around key)

#### Step 3: Security Verification

```bash
# Verify .env is in .gitignore
grep "^\.env$" .gitignore

# Should output: .env
```

**⚠️ CRITICAL SECURITY PRACTICES:**

- ✅ **DO**: Use `.env` files (already in `.gitignore`)
- ✅ **DO**: Use environment variables
- ✅ **DO**: Regenerate keys if accidentally committed
- ❌ **DON'T**: Hardcode keys in notebooks
- ❌ **DON'T**: Share `.env` files
- ❌ **DON'T**: Commit keys to version control
- ❌ **DON'T**: Include keys in screenshots

### Optional: HuggingFace Embeddings

**Local embeddings (no API key required):**

The tutorial uses `sentence-transformers/all-MiniLM-L6-v2` for local embeddings:

```python
from langchain_huggingface import HuggingFaceEmbeddings

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**First run:**

- Model downloads automatically (~90MB)
- Cached in `~/.cache/huggingface/`
- No internet needed for subsequent runs

**Suppress warnings:**
Add to `.env`:

```bash
TOKENIZERS_PARALLELISM=false
```

### Complete Configuration Reference

Your `.env` file supports many configuration options beyond just API keys. Here's a comprehensive guide:

#### Environment & Debug Settings

```bash
# Environment type: dev, test, prod
ENVIRONMENT=dev

# Enable debug mode (verbose logging)
DEBUG_MODE=false

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO
```

**When to adjust:**
- Use `ENVIRONMENT=prod` and `LOG_LEVEL=WARNING` in production
- Set `DEBUG_MODE=true` when troubleshooting issues

#### LLM Configuration

```bash
# OpenAI model to use
DEFAULT_MODEL=gpt-4o-mini

# Temperature (0.0 = deterministic, 1.0 = creative)
DEFAULT_TEMPERATURE=0

# Embeddings model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Model options:**
- `gpt-4o-mini` - Fast, cost-effective (recommended)
- `gpt-4o` - Most capable, slower, expensive
- `gpt-4-turbo` - Balanced performance
- `gpt-3.5-turbo` - Fastest, least capable

**Embedding options:**
- `text-embedding-3-small` - 1536 dimensions, faster, cheaper
- `text-embedding-3-large` - 3072 dimensions, more accurate

#### Retrieval Configuration

```bash
# Number of documents to retrieve
DEFAULT_K=4

# Documents to fetch before MMR filtering
DEFAULT_MMR_FETCH_K=20

# MMR Lambda: balance relevance vs diversity
DEFAULT_MMR_LAMBDA=0.5
```

**Tuning guidelines:**
- **DEFAULT_K**: Higher = more context but slower (3-5 recommended)
- **DEFAULT_MMR_FETCH_K**: Should be 4-5x DEFAULT_K
- **DEFAULT_MMR_LAMBDA**:
  - `1.0` = maximum relevance (may be redundant)
  - `0.5` = balanced (recommended)
  - `0.0` = maximum diversity (may lose relevance)

#### Text Processing

```bash
# Chunk size (characters)
DEFAULT_CHUNK_SIZE=1000

# Overlap between chunks
DEFAULT_CHUNK_OVERLAP=200
```

**Optimization tips:**
- Larger chunks (1500+) = more context, less granular
- Smaller chunks (500-) = more precise, may lose context
- Overlap should be 10-20% of chunk size

#### Display Settings

```bash
# Console output width
SECTION_WIDTH=80

# Document preview length
PREVIEW_LENGTH=300
```

#### Advanced Settings

```bash
# HuggingFace model for local embeddings
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Suppress tokenizer warnings
TOKENIZERS_PARALLELISM=false

# LangSmith tracing (requires LANGSMITH_API_KEY)
# LANGSMITH_TRACING=true
```

#### Configuration Profiles

**Development Profile (default):**
```bash
ENVIRONMENT=dev
DEBUG_MODE=false
LOG_LEVEL=INFO
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0
DEFAULT_K=4
```

**Production Profile:**
```bash
ENVIRONMENT=prod
DEBUG_MODE=false
LOG_LEVEL=WARNING
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0
DEFAULT_K=3  # Reduce for cost savings
```

**High-Accuracy Profile:**
```bash
ENVIRONMENT=prod
LOG_LEVEL=INFO
DEFAULT_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
DEFAULT_K=5
DEFAULT_MMR_LAMBDA=0.7  # Favor relevance
```

**Experimentation Profile:**
```bash
ENVIRONMENT=dev
DEBUG_MODE=true
LOG_LEVEL=DEBUG
DEFAULT_TEMPERATURE=0.7  # More creative responses
DEFAULT_K=10  # Maximum context
```

## Validation

### Validate OpenAI API Key

**Method 1: Python Script**

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    models = list(client.models.list().data)
    print(f"✅ API key is VALID! {len(models)} models available")
    print(f"✅ Account active, ready to use")
except Exception as e:
    print(f"❌ API key validation failed: {e}")
```

**Method 2: Quick Test**

```bash
# Create test script
cat > test_api.py << 'EOF'
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
print(client.models.list().data[0].id)
EOF

# Run test
python test_api.py

# Should output model name (e.g., "gpt-4o-mini")
```

### Validate Full Setup

Run the validation notebook:

```bash
jupyter notebook notebooks/00_index.ipynb
```

In the notebook, check:

- ✅ All imports successful
- ✅ API key validated
- ✅ Shared module accessible
- ✅ Vector stores directory created

## Alternative Setups

### Docker (Coming Soon)

```bash
# Build image
docker build -t langchain-rag-tutorial .

# Run container
docker run -p 8888:8888 -v $(pwd):/workspace langchain-rag-tutorial
```

### Conda Environment

```bash
# Create conda environment
conda create -n langchain-rag python=3.10
conda activate langchain-rag

# Install pip dependencies
pip install -r requirements.txt
```

### Poetry (Advanced)

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate shell
poetry shell
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**

```python
ModuleNotFoundError: No module named 'langchain'
```

**Solutions:**

```bash
# Verify virtual environment is activated
which python  # Should point to venv/

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Clear pip cache and reinstall
pip cache purge
pip install -r requirements.txt
```

#### 2. API Key Not Found

**Problem:**

```
openai.OpenAIError: The api_key client option must be set
```

**Solutions:**

```bash
# Verify .env file exists
ls -la .env

# Check .env content (don't share output!)
cat .env

# Restart Jupyter kernel
# In Jupyter: Kernel → Restart

# Reload environment in Python
from dotenv import load_dotenv
load_dotenv(override=True)
```

#### 3. FAISS Installation Issues

**Problem:**

```
ImportError: DLL load failed while importing _swigfaiss
```

**Solution (Windows):**

```bash
# Uninstall and reinstall faiss-cpu
pip uninstall faiss-cpu
pip install faiss-cpu --no-cache-dir
```

**Solution (macOS with M1/M2):**

```bash
# Use conda for FAISS on Apple Silicon
conda install -c conda-forge faiss-cpu
```

#### 4. SSL Certificate Errors

**Problem:**

```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```

**Solution (macOS):**

```bash
# Run Python certificates installer
/Applications/Python\ 3.x/Install\ Certificates.command
```

**Solution (General):**

```bash
pip install --upgrade certifi
```

#### 5. Jupyter Kernel Not Found

**Problem:**
Kernel "Python 3" not found in Jupyter

**Solution:**

```bash
# Install kernel
python -m ipykernel install --user --name=venv --display-name="Python (LangChain RAG)"

# Restart Jupyter
# In Jupyter: Select Kernel → Python (LangChain RAG)
```

### Getting Help

If issues persist:

1. **Check existing issues**: [GitHub Issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
2. **Create new issue**: Include error message, OS, Python version
3. **Review FAQ**: [FAQ.md](FAQ.md)
4. **Detailed troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## Next Steps

✅ Installation complete? Great!

- 🚀 **Quick start**: [GETTING_STARTED.md](GETTING_STARTED.md)
- 🏗️ **Understand architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- 📚 **API reference**: [API_REFERENCE.md](API_REFERENCE.md)
- 📖 **Start learning**: `notebooks/00_index.ipynb`

## Version Information

Current versions (as of 2024):

- Python: 3.9+
- LangChain: >=0.1.0
- OpenAI SDK: >=1.12.0
- FAISS: >=1.7.4

Check [CHANGELOG.md](CHANGELOG.md) for updates.
