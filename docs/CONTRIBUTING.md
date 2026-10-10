# Contributing to LangChain RAG Tutorial

Thank you for your interest in contributing! This document provides guidelines for contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Style Guide](#style-guide)

## Code of Conduct

Be respectful, inclusive, and constructive. We welcome contributions from developers of all skill levels.

## Getting Started

### Ways to Contribute

- **Bug reports**: [open an issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- **Feature requests**: [suggest an idea](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
- **Documentation**: improve docs, fix typos
- **Code**: fix bugs, add features, optimize performance
- **Educational content**: add notebooks, examples, use cases

### Before Contributing

1. Check [existing issues](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)
2. Read this guide completely
3. Fork the repository
4. Set up development environment

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub first, then clone your fork
git clone https://github.com/YOUR_USERNAME/langchain-rag-tutorial.git
cd langchain-rag-tutorial

# Add upstream remote
git remote add upstream https://github.com/gianlucamazza/langchain-rag-tutorial.git
```

### 2. Create Branch

```bash
# Update main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Install Development Dependencies

```bash
# Create a virtual environment (Python 3.10-3.13)
python3 -m venv venv
source venv/bin/activate

# Install runtime and development dependencies
make install
make install-dev   # pip install -r requirements-dev.txt && pre-commit install
```

`requirements-dev.txt` provides pytest (with pytest-cov), ruff, mypy and pre-commit. Tests use
fake embeddings and LLMs, so `make test` needs no network access or API key.

### 4. Configure Environment

```bash
# Copy .env template
cp .env.example .env

# Add your API keys
# Edit .env with your keys
```

## Contributing Guidelines

### Code Contributions

#### For Shared Module (shared/)

**Adding a new utility function:**

1. Add function to appropriate file:
   - `config.py` - Configuration constants
   - `utils.py` - General utilities
   - `loaders.py` - Document loading
   - `prompts.py` - Prompt templates
   - `retrievers.py` - Retriever builders (hybrid search, reranking)

2. Include docstring:

```python
def your_function(param1: str, param2: int = 10) -> dict:
    """
    Brief description.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        Description of return value

    Example:
        >>> result = your_function("test")
        >>> print(result)
    """
    # Implementation
    pass
```

3. Export in `__init__.py`:

```python
from .utils import your_function

__all__ = [
    # ... existing exports
    "your_function",
]
```

4. Update `docs/API_REFERENCE.md`

#### For Notebooks (notebooks/)

**Adding a new notebook:**

1. Choose appropriate directory:
   - `fundamentals/` - Core concepts
   - `advanced_architectures/` - Advanced patterns

2. Follow naming convention:
   - Number prefix: `##_descriptive_name.ipynb`
   - Use underscores, lowercase

3. Include standard sections:

   ```markdown
   # ## - Title

   **Complexity:** 3/5
   **Use Cases:** ...
   **Key Features:** ...

   ## 1. Setup
   ## 2. Implementation
   ## 3. Testing
   ## 4. Summary
   ```

4. Import from shared:

   ```python
   import sys
   sys.path.append('../..')

   from shared import *
   ```

### Documentation Contributions

**Fixing typos or improving docs:**

1. Edit appropriate file:
   - `docs/` - Modular documentation
   - `README.md` - Overview, quick start and links into `docs/`
   - Notebook READMEs - Category-specific

2. Use clear, concise language
3. Test all code examples
4. Update links if moving content

**Adding new documentation:**

1. Follow existing structure
2. Add to table of contents
3. Cross-link related docs
4. Include examples

## Pull Request Process

### 1. Pre-PR Checklist

Before submitting, ensure:

- [ ] Code runs without errors
- [ ] `make test` and `make lint` pass
- [ ] All notebooks execute successfully
- [ ] Documentation updated
- [ ] API_REFERENCE.md updated (if changed shared/)
- [ ] README links work
- [ ] No secrets committed (.env, API keys)
- [ ] Git history is clean

### 2. Running Tests and Quality Checks

```bash
make test      # pytest tests/ with coverage for shared/
make lint      # ruff check + ruff format --check + mypy
make format    # ruff format + ruff check --fix
```

Useful pytest variants:

```bash
pytest tests/test_utils.py -v   # single file
pytest -m "not slow"            # skip tests marked slow
```

Pre-commit runs the ruff hooks (`ruff-pre-commit`) plus basic file checks on every commit; run
them on the whole tree with:

```bash
pre-commit run --all-files
```

If you changed a notebook, execute it end to end:

```bash
jupyter nbconvert --to notebook --execute notebooks/fundamentals/01_setup_and_basics.ipynb
```

CI (GitHub Actions) runs the test suite on Python 3.10, 3.11, 3.12 and 3.13, and linting on
Python 3.12.

### 3. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: Add HyDe optimization for long queries

- Implement dynamic hypothetical doc generation
- Add caching for repeated queries
- Update API_REFERENCE.md with new function

Fixes #42"
```

**Commit message format:**

```
<type>: <subject>

<body>

<footer>
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

### 4. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then on GitHub:

1. Click "New Pull Request"
2. Fill in PR template
3. Link related issues
4. Wait for review

### 5. PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tested locally
- [ ] All notebooks execute successfully
- [ ] Documentation links checked

## Related Issues
Fixes #issue_number

## Screenshots (if applicable)
```

## Style Guide

### Python Code

Formatting and linting are enforced by ruff (configuration in `ruff.toml`); type checking uses
mypy. Run `make format` before committing. Conventions:

```python
# Imports: standard, third-party, local
import os
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from shared import format_docs

# Constants: UPPER_SNAKE_CASE
DEFAULT_CHUNK_SIZE = 1000

# Functions: snake_case
def load_vector_store(path: str) -> FAISS:
    """Docstring here."""
    pass

# Classes: PascalCase (if adding)
class CustomRetriever:
    pass
```

### Jupyter Notebooks

```python
# Cell 1: Markdown title
"""
# ## - Title

**Complexity:** 3/5
"""

# Cell 2: Imports
import sys
sys.path.append('../..')

from shared import *

# Cell 3: Setup with clear output
print_section_header("Setup")
# ... code ...
print("Setup complete")

# Keep cells focused: 5-15 lines of code max
# Add markdown cells between code for explanation
```

### Documentation

- Use **bold** for emphasis
- Use `code` for technical terms
- Do not use emoji; write status in words (for example Yes/No, 3/5)
- Include code examples with syntax highlighting
- Add links to related docs
- Keep lines under 100 characters

## Questions?

- **General questions**: open a [Discussion](https://github.com/gianlucamazza/langchain-rag-tutorial/discussions)
- **Bug reports**: [open an issue](https://github.com/gianlucamazza/langchain-rag-tutorial/issues)

Thank you for contributing.
