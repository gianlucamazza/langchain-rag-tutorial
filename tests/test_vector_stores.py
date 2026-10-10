"""
Tests for vector store helpers and model-keyed store paths
"""

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import DeterministicFakeEmbedding

from shared import config
from shared.utils import require_vector_store, save_vector_store


def test_store_paths_are_keyed_by_embedding_model():
    assert config.OPENAI_EMBEDDING_MODEL in config.OPENAI_VECTOR_STORE_PATH.name
    assert config.HF_EMBEDDING_MODEL.replace("/", "__") in config.HF_VECTOR_STORE_PATH.name
    assert config.OPENAI_VECTOR_STORE_PATH.parent == config.VECTOR_STORE_DIR


def test_require_vector_store_roundtrip(tmp_path, sample_documents):
    embeddings = DeterministicFakeEmbedding(size=16)
    save_vector_store(FAISS.from_documents(sample_documents, embeddings), tmp_path / "store")

    store = require_vector_store(tmp_path / "store", embeddings)

    assert store.index.ntotal == len(sample_documents)


def test_require_vector_store_missing_raises_with_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="make vector-stores"):
        require_vector_store(tmp_path / "missing", DeterministicFakeEmbedding(size=16))
