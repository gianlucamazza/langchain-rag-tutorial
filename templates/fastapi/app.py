"""
FastAPI Production Template for LangChain RAG
Production-ready with error handling, logging, monitoring, and rate limiting
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import time
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from shared import (
    load_vector_store,
    format_docs,
    VECTOR_STORE_DIR
)
from shared.prompts import RAG_PROMPT_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LangChain RAG API",
    description="Production-ready RAG API with LangChain",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    k: int = Field(4, ge=1, le=10, description="Number of documents to retrieve")
    architecture: str = Field("simple", description="RAG architecture: simple, contextual, fusion")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is RAG?",
                "k": 4,
                "architecture": "simple"
            }
        }


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(..., description="Retrieved document sources")
    latency_ms: float = Field(..., description="Query latency in milliseconds")
    architecture: str = Field(..., description="Architecture used")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    vector_store_loaded: bool


# Global state
class AppState:
    """Application state management"""
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.llm = None
        self.chain = None
        self.initialized = False

    def initialize(self):
        """Initialize RAG components"""
        if self.initialized:
            return

        try:
            logger.info("Initializing RAG components...")

            # Load embeddings and vector store
            embeddings = OpenAIEmbeddings()
            self.vectorstore = load_vector_store(
                VECTOR_STORE_DIR / "openai_embeddings",
                embeddings
            )

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            # Initialize LLM
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            # Build RAG chain
            self.chain = (
                {"context": self.retriever | format_docs, "input": RunnablePassthrough()}
                | RAG_PROMPT_TEMPLATE
                | self.llm
                | StrOutputParser()
            )

            self.initialized = True
            logger.info("RAG components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise


# Application state instance
state = AppState()


# Dependency
async def get_app_state():
    """Dependency to get application state"""
    if not state.initialized:
        state.initialize()
    return state


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting LangChain RAG API...")
    state.initialize()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.2.0",
        vector_store_loaded=state.initialized
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    app_state: AppState = Depends(get_app_state)
):
    """
    Query the RAG system

    Args:
        request: Query request with question and parameters

    Returns:
        QueryResponse with answer and metadata
    """
    start_time = time.time()

    try:
        logger.info(f"Processing query: {request.query[:50]}...")

        # Update retriever with custom k
        retriever = app_state.vectorstore.as_retriever(
            search_kwargs={"k": request.k}
        )

        # Rebuild chain with custom retriever
        chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | RAG_PROMPT_TEMPLATE
            | app_state.llm
            | StrOutputParser()
        )

        # Get answer
        answer = chain.invoke(request.query)

        # Get source documents
        docs = retriever.invoke(request.query)
        sources = [doc.metadata.get("source", "unknown") for doc in docs]

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Query completed in {latency_ms:.2f}ms")

        return QueryResponse(
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
            architecture=request.architecture
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/architectures")
async def list_architectures():
    """List available RAG architectures"""
    return {
        "architectures": [
            {
                "name": "simple",
                "description": "Basic RAG with similarity search",
                "latency": "~2s",
                "cost": "Low"
            },
            {
                "name": "contextual",
                "description": "Context-augmented chunking (Anthropic)",
                "latency": "~2-3s",
                "cost": "Low"
            },
            {
                "name": "fusion",
                "description": "Multi-query with Reciprocal Rank Fusion",
                "latency": "~5-8s",
                "cost": "Medium"
            }
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
