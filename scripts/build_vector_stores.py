"""
Build the shared FAISS vector stores used by notebooks 03-10 and the templates.

Usage:
    python scripts/build_vector_stores.py [--provider {openai,huggingface,all}] [--version TAG]

With --version, stores are written under data/vector_stores/<TAG>/ so that
indexes for a deployment can be built and shipped as an immutable artifact.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402

from shared.config import (  # noqa: E402
    HF_EMBEDDING_MODEL,
    HF_VECTOR_STORE_PATH,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_VECTOR_STORE_PATH,
    VECTOR_STORE_DIR,
    verify_api_key,
)
from shared.loaders import load_and_split  # noqa: E402
from shared.utils import save_vector_store  # noqa: E402


def target_path(default_path: Path, version: str | None) -> Path:
    """Place the store under a version directory when a tag is given."""
    if version is None:
        return default_path
    return VECTOR_STORE_DIR / version / default_path.name


def build_store(chunks: list[Document], embeddings: Embeddings, model: str, path: Path) -> None:
    print(f"Embedding {len(chunks)} chunks with {model}...")
    save_vector_store(FAISS.from_documents(chunks, embeddings), path)


def build(provider: str, version: str | None) -> None:
    if provider in ("openai", "all") and not verify_api_key():
        raise SystemExit("OPENAI_API_KEY is required to build the OpenAI store.")

    _, chunks = load_and_split()

    if provider in ("openai", "all"):
        from langchain_openai import OpenAIEmbeddings

        build_store(
            chunks,
            OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL),
            OPENAI_EMBEDDING_MODEL,
            target_path(OPENAI_VECTOR_STORE_PATH, version),
        )

    if provider in ("huggingface", "all"):
        from langchain_huggingface import HuggingFaceEmbeddings

        build_store(
            chunks,
            HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL),
            HF_EMBEDDING_MODEL,
            target_path(HF_VECTOR_STORE_PATH, version),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--provider", choices=["openai", "huggingface", "all"], default="all")
    parser.add_argument("--version", help="Optional tag: write stores to data/vector_stores/<TAG>/")
    args = parser.parse_args()
    build(args.provider, args.version)


if __name__ == "__main__":
    main()
