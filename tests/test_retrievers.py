"""
Tests for hybrid search and reranking retriever builders
"""

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from shared.retrievers import (
    bm25_tokenize,
    build_bm25_retriever,
    build_hybrid_retriever,
    build_reranking_retriever,
)


@pytest.fixture
def corpus():
    return [
        Document(page_content="FAISS stores dense vectors for similarity search."),
        Document(page_content="BM25 ranks documents by exact keyword overlap."),
        Document(page_content="Cross-encoders score a query and a passage jointly."),
        Document(page_content="The InMemoryStore keeps parent documents by id."),
    ]


@pytest.fixture
def vectorstore(corpus):
    return FAISS.from_documents(corpus, DeterministicFakeEmbedding(size=16))


class KeywordCrossEncoder(BaseCrossEncoder):
    """Scores a pair by how many query words appear in the passage."""

    def score(self, text_pairs):
        return [
            float(sum(word in passage.lower() for word in query.lower().split()))
            for query, passage in text_pairs
        ]


def test_bm25_ranks_exact_keyword_first(corpus):
    retriever = build_bm25_retriever(corpus, k=2)

    results = retriever.invoke("InMemoryStore parent")

    assert len(results) == 2
    assert "InMemoryStore" in results[0].page_content


def test_bm25_tokenize_keeps_identifiers():
    assert bm25_tokenize("Set max_retries=2, then call init_chat_model().") == [
        "set",
        "max_retries",
        "2",
        "then",
        "call",
        "init_chat_model",
    ]


def test_bm25_matches_identifier_next_to_punctuation():
    docs = [
        Document(page_content="model = init(max_retries=2, timeout=30)"),
        Document(page_content="Retries are handled by the client."),
        # A third doc: with two docs, Okapi IDF of a term in one of them is log(1) = 0
        Document(page_content="Timeouts are set in seconds."),
    ]

    results = build_bm25_retriever(docs, k=1).invoke("max_retries")

    assert "max_retries" in results[0].page_content


def test_hybrid_fuses_both_retrievers(corpus, vectorstore):
    retriever = build_hybrid_retriever(corpus, vectorstore, k=2)

    results = retriever.invoke("InMemoryStore parent")
    contents = [doc.page_content for doc in results]

    # BM25 guarantees the keyword match survives fusion; no duplicates after RRF
    assert any("InMemoryStore" in c for c in contents)
    assert len(contents) == len(set(contents))
    assert 2 <= len(contents) <= 4


def test_hybrid_weights(corpus, vectorstore):
    retriever = build_hybrid_retriever(corpus, vectorstore, bm25_weight=0.3)

    assert retriever.weights == pytest.approx([0.3, 0.7])


def test_hybrid_rejects_invalid_weight(corpus, vectorstore):
    with pytest.raises(ValueError, match="bm25_weight"):
        build_hybrid_retriever(corpus, vectorstore, bm25_weight=1.5)


def test_reranker_reorders_and_truncates(corpus, vectorstore):
    base = vectorstore.as_retriever(search_kwargs={"k": 4})
    retriever = build_reranking_retriever(base, top_n=1, cross_encoder=KeywordCrossEncoder())

    results = retriever.invoke("cross-encoders score passage")

    assert len(results) == 1
    assert results[0].page_content.startswith("Cross-encoders")
