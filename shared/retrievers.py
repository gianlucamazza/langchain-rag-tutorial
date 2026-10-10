"""
Retriever builders for LangChain RAG Tutorial
Hybrid (BM25 + dense) search and cross-encoder reranking (notebooks 12, 19).
"""

import re
from collections.abc import Callable, Sequence

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from .config import DEFAULT_K, DEFAULT_RERANKER_MODEL


def bm25_tokenize(text: str) -> list[str]:
    """
    Lowercase and split on non-word characters, keeping snake_case identifiers whole.

    BM25Retriever's default splits on whitespace only, so "max_retries=2," would
    never match the query token "max_retries".
    """
    return re.findall(r"\w+", text.lower())


def build_bm25_retriever(
    docs: Sequence[Document],
    k: int = DEFAULT_K,
    tokenizer: Callable[[str], list[str]] = bm25_tokenize,
) -> BM25Retriever:
    """
    Build a keyword (BM25) retriever over the given chunks.

    Args:
        docs: Chunks to index (kept in memory; BM25 has no persistent index)
        k: Number of documents to return
        tokenizer: Applied to both documents and queries

    Returns:
        BM25Retriever
    """
    return BM25Retriever.from_documents(list(docs), k=k, preprocess_func=tokenizer)


def build_hybrid_retriever(
    docs: Sequence[Document],
    vectorstore: VectorStore,
    k: int = DEFAULT_K,
    bm25_weight: float = 0.5,
    rrf_c: int = 60,
) -> EnsembleRetriever:
    """
    Combine BM25 and dense retrieval with weighted Reciprocal Rank Fusion.

    Both retrievers must index the same chunks: RRF matches documents by content.

    Args:
        docs: Chunks indexed in the vector store (used to build the BM25 index)
        vectorstore: Dense vector store over the same chunks
        k: Documents fetched from each retriever before fusion
        bm25_weight: Weight of BM25 in [0, 1]; the dense retriever gets 1 - bm25_weight
        rrf_c: RRF constant (60 as in the original paper)

    Returns:
        EnsembleRetriever returning the fused ranking (up to 2 * k unique documents)
    """
    if not 0.0 <= bm25_weight <= 1.0:
        raise ValueError(f"bm25_weight must be in [0, 1], got {bm25_weight}")

    return EnsembleRetriever(
        retrievers=[
            build_bm25_retriever(docs, k=k),
            vectorstore.as_retriever(search_kwargs={"k": k}),
        ],
        weights=[bm25_weight, 1.0 - bm25_weight],
        c=rrf_c,
    )


def build_reranking_retriever(
    base_retriever: BaseRetriever,
    top_n: int = DEFAULT_K,
    model_name: str = DEFAULT_RERANKER_MODEL,
    cross_encoder: BaseCrossEncoder | None = None,
) -> ContextualCompressionRetriever:
    """
    Rerank a retriever's candidates with a cross-encoder and keep the top_n.

    The base retriever should over-fetch (e.g. 20 candidates) so the reranker has
    something to choose from.

    Args:
        base_retriever: First-stage retriever producing candidates
        top_n: Documents kept after reranking
        model_name: HuggingFace cross-encoder model (downloaded on first use)
        cross_encoder: Pre-built cross-encoder; overrides model_name (reuse across calls)

    Returns:
        ContextualCompressionRetriever
    """
    if cross_encoder is None:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder

        cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)

    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=top_n),
        base_retriever=base_retriever,
    )
