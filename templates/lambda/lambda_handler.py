"""
AWS Lambda Handler for LangChain RAG
Serverless deployment template
"""

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
DEFAULT_K = 4

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

            vectorstore = FAISS.load_local(
                local_path,
                embeddings,
                allow_dangerous_deserialization=True,  # Trusted: our own bucket
            )
        else:
            # Load from local (Lambda layer)
            vectorstore = FAISS.load_local(
                f"/opt/vector_stores/{STORE_NAME}",
                embeddings,
                allow_dangerous_deserialization=True,  # Trusted: shipped in our Lambda layer
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

        k = int(body.get("k", DEFAULT_K))

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

    except Exception as e:
        print(f"Error processing query: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


# Local testing
if __name__ == "__main__":
    test_event = {"query": "What is RAG?", "k": 4}

    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
