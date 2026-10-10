"""
AWS Lambda Handler for LangChain RAG
Serverless deployment template
"""

import hashlib
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append("/opt/python")

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
VECTOR_STORE_BUCKET = os.environ.get("VECTOR_STORE_BUCKET", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# Same layout as shared/config.py: a store is keyed by the model that built it
STORE_NAME = f"openai__{OPENAI_EMBEDDING_MODEL}"
VECTOR_STORE_KEY = os.environ.get("VECTOR_STORE_KEY", f"vector_stores/{STORE_NAME}")
VECTOR_STORE_INDEX_SHA256 = os.environ.get("VECTOR_STORE_INDEX_SHA256", "").strip().lower()
ALLOW_UNVERIFIED_VECTOR_STORE = os.environ.get("ALLOW_UNVERIFIED_VECTOR_STORE", "").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_K = 4
MIN_K = 1
MAX_K = 10

RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {input}

Answer:"""
)

# Global variables (cached across invocations)
vectorstore = None
answer_chain = None


def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_index_pkl(index_pkl: Path) -> None:
    """Refuse to unpickle index.pkl unless its SHA-256 matches or opt-out is set."""
    if ALLOW_UNVERIFIED_VECTOR_STORE:
        print(
            "WARNING: loading FAISS index.pkl without SHA-256 verification "
            "(ALLOW_UNVERIFIED_VECTOR_STORE is set)"
        )
        return
    if not VECTOR_STORE_INDEX_SHA256:
        raise RuntimeError(
            "VECTOR_STORE_INDEX_SHA256 is unset. Refusing to unpickle index.pkl. "
            "Set the expected SHA-256, or ALLOW_UNVERIFIED_VECTOR_STORE=1 to opt out."
        )
    digest = _sha256_file(index_pkl)
    if digest != VECTOR_STORE_INDEX_SHA256:
        raise RuntimeError("index.pkl SHA-256 does not match VECTOR_STORE_INDEX_SHA256")


def parse_k(raw) -> int:
    """Parse k as an int in [1, 10]. Raises ValueError on bad input."""
    if raw is None:
        return DEFAULT_K
    if isinstance(raw, bool) or isinstance(raw, float):
        raise ValueError("k must be an integer")
    if isinstance(raw, int):
        k = raw
    elif isinstance(raw, str) and raw.strip().lstrip("-").isdigit():
        k = int(raw)
    else:
        raise ValueError("k must be an integer")
    if k < MIN_K or k > MAX_K:
        raise ValueError("k out of range")
    return k


def initialize():
    """Initialize RAG components (cold start)"""
    global vectorstore, answer_chain

    if answer_chain is not None:
        return  # Already initialized

    print("Initializing RAG components...")

    try:
        # Initialize embeddings
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

        # Load vector store from S3 or local
        if VECTOR_STORE_BUCKET:
            # Load from S3
            import boto3

            s3 = boto3.client("s3")

            # Download vector store files
            local_path = "/tmp/vector_store"
            Path(local_path).mkdir(parents=True, exist_ok=True)

            # Download index and pkl files
            for file in ["index.faiss", "index.pkl"]:
                s3.download_file(
                    VECTOR_STORE_BUCKET, f"{VECTOR_STORE_KEY}/{file}", f"{local_path}/{file}"
                )

            verify_index_pkl(Path(local_path) / "index.pkl")
            vectorstore = FAISS.load_local(
                local_path,
                embeddings,
                # Only after SHA-256 check (or explicit opt-out)
                allow_dangerous_deserialization=True,
            )
        else:
            layer_path = Path(f"/opt/vector_stores/{STORE_NAME}")
            verify_index_pkl(layer_path / "index.pkl")
            vectorstore = FAISS.load_local(
                str(layer_path),
                embeddings,
                # Only after SHA-256 check (or explicit opt-out)
                allow_dangerous_deserialization=True,
            )

        # Retrieval runs per request (k is request-scoped); the chain only answers
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)
        answer_chain = RAG_PROMPT | llm | StrOutputParser()

        print("Initialization complete")

    except Exception as e:
        print(f"Initialization failed: {e}")
        raise


def lambda_handler(event, context):
    """
    Lambda handler for RAG queries

    Event format:
    {
        "query": "What is RAG?",
        "k": 4  # optional
    }
    """

    # Initialize (only on cold start)
    initialize()

    try:
        # Parse input
        if isinstance(event, str):
            event = json.loads(event)

        # Handle API Gateway event format
        if "body" in event:
            body = json.loads(event["body"])
        else:
            body = event

        query = body.get("query")
        if not query:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing query parameter"})}

        try:
            k = parse_k(body.get("k", DEFAULT_K))
        except ValueError:
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid k parameter"})}

        print(f"Processing query: {query}")

        docs = vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
        answer = answer_chain.invoke({"context": format_docs(docs), "input": query})
        sources = [doc.metadata.get("source", "unknown") for doc in docs]

        # Return response
        response = {"answer": answer, "sources": sources, "query": query}

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps(response),
        }

    except Exception:
        print("Error processing query")
        return {"statusCode": 500, "body": json.dumps({"error": "Request failed"})}


# Local testing
if __name__ == "__main__":
    test_event = {"query": "What is RAG?", "k": 4}

    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
