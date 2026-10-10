# Streamlit Template

Interactive demo UI for the tutorial RAG chain. Not a production app.

## Features

- Real-time query processing
- Source document display
- Architecture selection
- Performance metrics
- Sample queries

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Set your OpenAI API key, either in the project-root `.env` (loaded by the `shared` module) or in
the shell:

```bash
export OPENAI_API_KEY=sk-proj-your-key-here
export DEFAULT_MODEL=gpt-4o-mini   # optional, this is the default
```

The app loads the OpenAI vector store from `OPENAI_VECTOR_STORE_PATH`
(`data/vector_stores/openai__<OPENAI_EMBEDDING_MODEL>`, default
`openai__text-embedding-3-small`) using `OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)`. Build it first
with notebook 02 or `make vector-stores` from the project root.

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501.

## Configuration

| Variable         | Default       | Description                 |
| ---------------- | ------------- | --------------------------- |
| `OPENAI_API_KEY` | (required)    | OpenAI API key              |
| `DEFAULT_MODEL`  | `gpt-4o-mini` | Chat model used for answers |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model; selects the vector store |

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create an app at [share.streamlit.io](https://share.streamlit.io) pointing to
   `templates/streamlit/streamlit_app.py`.
3. Add `OPENAI_API_KEY` (and optionally `DEFAULT_MODEL`) as secrets.
4. Deploy.

### Docker

No template-specific Dockerfile is included. See [docs/DEPLOYMENT.md](../../docs/DEPLOYMENT.md)
for container guidance.
