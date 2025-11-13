#!/usr/bin/env python3
"""
Build Vector Stores Script

This script pre-builds all vector stores for the RAG tutorial notebooks.
It loads documents, creates embeddings, and saves FAISS indices to disk.

Usage:
    python scripts/build_vector_stores.py [--stores STORE_NAMES] [--force]

Examples:
    # Build all vector stores
    python scripts/build_vector_stores.py

    # Build specific stores
    python scripts/build_vector_stores.py --stores openai_embeddings huggingface_embeddings

    # Force rebuild even if stores exist
    python scripts/build_vector_stores.py --force
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.config import VECTOR_STORE_DIR, DEFAULT_LANGCHAIN_URLS
from shared.loaders import load_and_split
from shared.utils import save_vector_store, print_section_header
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================================
# VECTOR STORE CONFIGURATIONS
# ============================================================================

VECTOR_STORE_CONFIGS = {
    "openai_embeddings": {
        "description": "OpenAI text-embedding-3-small",
        "embedding_class": OpenAIEmbeddings,
        "embedding_kwargs": {"model": "text-embedding-3-small"},
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "huggingface_embeddings": {
        "description": "HuggingFace all-MiniLM-L6-v2",
        "embedding_class": HuggingFaceEmbeddings,
        "embedding_kwargs": {"model_name": "all-MiniLM-L6-v2"},
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "ragas_evaluation": {
        "description": "RAGAS evaluation store (OpenAI)",
        "embedding_class": OpenAIEmbeddings,
        "embedding_kwargs": {"model": "text-embedding-3-small"},
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "fusion_rag": {
        "description": "Fusion RAG store (OpenAI)",
        "embedding_class": OpenAIEmbeddings,
        "embedding_kwargs": {"model": "text-embedding-3-small"},
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "contextual_rag_standard": {
        "description": "Contextual RAG standard chunks (OpenAI)",
        "embedding_class": OpenAIEmbeddings,
        "embedding_kwargs": {"model": "text-embedding-3-small"},
        "chunk_size": 800,
        "chunk_overlap": 200,
    },
    "contextual_rag_contextual": {
        "description": "Contextual RAG context-augmented chunks (OpenAI)",
        "embedding_class": OpenAIEmbeddings,
        "embedding_kwargs": {"model": "text-embedding-3-small"},
        "chunk_size": 800,
        "chunk_overlap": 200,
    },
}


# ============================================================================
# BUILD FUNCTIONS
# ============================================================================


def build_vector_store(
    store_name: str, config: dict, force: bool = False, verbose: bool = True
) -> bool:
    """
    Build a single vector store.

    Args:
        store_name: Name of the vector store
        config: Configuration dictionary
        force: Force rebuild even if store exists
        verbose: Print progress messages

    Returns:
        bool: True if successful, False otherwise
    """
    store_path = VECTOR_STORE_DIR / store_name

    # Check if store already exists
    if store_path.exists() and not force:
        if verbose:
            print(f"✓ {store_name} already exists (use --force to rebuild)")
        return True

    if verbose:
        print(f"\n{'='*80}")
        print(f"Building: {store_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*80}")

    try:
        # Load and split documents
        if verbose:
            print("\n📄 Loading and splitting documents...")

        documents = load_and_split(
            urls=DEFAULT_LANGCHAIN_URLS,
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            verbose=verbose,
        )

        if verbose:
            print(f"   → Loaded {len(documents)} document chunks")

        # Initialize embeddings
        if verbose:
            print(f"\n🔤 Initializing embeddings: {config['description']}")

        embeddings = config["embedding_class"](**config["embedding_kwargs"])

        # Create vector store
        if verbose:
            print("\n🔍 Creating FAISS vector store...")

        vector_store = FAISS.from_documents(documents, embeddings)

        # Save to disk
        if verbose:
            print(f"\n💾 Saving to: {store_path}")

        save_vector_store(vector_store, store_name)

        if verbose:
            print(f"\n✅ Successfully built {store_name}")

        return True

    except Exception as e:
        print(f"\n❌ Error building {store_name}: {str(e)}", file=sys.stderr)
        return False


def build_all_stores(
    store_names: Optional[List[str]] = None, force: bool = False, verbose: bool = True
) -> dict:
    """
    Build multiple vector stores.

    Args:
        store_names: List of store names to build (None = all)
        force: Force rebuild even if stores exist
        verbose: Print progress messages

    Returns:
        dict: Results with success/failure counts
    """
    if store_names is None:
        store_names = list(VECTOR_STORE_CONFIGS.keys())

    # Validate store names
    invalid_stores = [s for s in store_names if s not in VECTOR_STORE_CONFIGS]
    if invalid_stores:
        print(f"❌ Invalid store names: {', '.join(invalid_stores)}", file=sys.stderr)
        print(f"Available stores: {', '.join(VECTOR_STORE_CONFIGS.keys())}")
        sys.exit(1)

    print_section_header("Building Vector Stores")
    print(f"\nStores to build: {len(store_names)}")
    print(f"Force rebuild: {force}")

    results = {"success": [], "failed": [], "skipped": []}

    for store_name in store_names:
        config = VECTOR_STORE_CONFIGS[store_name]
        success = build_vector_store(store_name, config, force, verbose)

        if success:
            results["success"].append(store_name)
        else:
            results["failed"].append(store_name)

    # Print summary
    print(f"\n{'='*80}")
    print("BUILD SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {len(results['success'])}")
    print(f"❌ Failed: {len(results['failed'])}")

    if results["failed"]:
        print("\nFailed stores:")
        for store in results["failed"]:
            print(f"  - {store}")

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build vector stores for LangChain RAG Tutorial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build all vector stores
  python scripts/build_vector_stores.py

  # Build specific stores
  python scripts/build_vector_stores.py --stores openai_embeddings huggingface_embeddings

  # Force rebuild
  python scripts/build_vector_stores.py --force

  # List available stores
  python scripts/build_vector_stores.py --list
        """,
    )

    parser.add_argument(
        "--stores", nargs="+", help="Specific store names to build (default: all)"
    )

    parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if stores exist"
    )

    parser.add_argument(
        "--list", action="store_true", help="List available vector stores and exit"
    )

    parser.add_argument("--quiet", action="store_true", help="Minimize output")

    args = parser.parse_args()

    # List stores if requested
    if args.list:
        print("Available vector stores:")
        for name, config in VECTOR_STORE_CONFIGS.items():
            print(f"  - {name}: {config['description']}")
        sys.exit(0)

    # Build stores
    verbose = not args.quiet
    results = build_all_stores(
        store_names=args.stores, force=args.force, verbose=verbose
    )

    # Exit with appropriate code
    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
