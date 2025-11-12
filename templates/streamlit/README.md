# Streamlit Production Template

Interactive web UI for the RAG system.

## Features

- ✅ Beautiful, responsive UI
- ✅ Real-time query processing
- ✅ Source document display
- ✅ Architecture selection
- ✅ Performance metrics
- ✅ Sample queries

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-proj-your-key-here
```

### 3. Run Application

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at: http://localhost:8501

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add secrets in Streamlit dashboard
4. Deploy

### Docker

```bash
docker build -t rag-streamlit -f Dockerfile .
docker run -p 8501:8501 rag-streamlit
```
