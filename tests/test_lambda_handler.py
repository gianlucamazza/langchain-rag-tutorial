"""
Tests for templates/lambda/lambda_handler.py (no network: fake embeddings and LLM)
"""

import importlib.util
import json
from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser

HANDLER_PATH = Path(__file__).parent.parent / "templates" / "lambda" / "lambda_handler.py"


@pytest.fixture
def handler(monkeypatch):
    spec = importlib.util.spec_from_file_location("lambda_handler", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    docs = [Document(page_content=f"doc {i}", metadata={"source": f"s{i}"}) for i in range(6)]
    module.vectorstore = FAISS.from_documents(docs, DeterministicFakeEmbedding(size=16))
    llm = FakeListChatModel(responses=["answer"] * 10)
    module.answer_chain = module.RAG_PROMPT | llm | StrOutputParser()
    return module


def invoke(handler, payload):
    result = handler.lambda_handler({"body": json.dumps(payload)}, None)
    return result["statusCode"], json.loads(result["body"])


def test_custom_k_is_request_scoped(handler):
    status, body = invoke(handler, {"query": "q", "k": 2})
    assert status == 200
    assert len(body["sources"]) == 2

    # A later request without k must fall back to the default, not reuse k=2
    _, body = invoke(handler, {"query": "q"})
    assert len(body["sources"]) == handler.DEFAULT_K


def test_missing_query_returns_400(handler):
    status, body = invoke(handler, {})
    assert status == 400
    assert "Missing query" in body["error"]


def test_format_docs_joins_with_real_newlines(handler):
    docs = [Document(page_content="a"), Document(page_content="b")]
    assert handler.format_docs(docs) == "a\n\nb"
