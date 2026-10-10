"""
FastAPI starter template for the LangChain RAG tutorial.
Teaching app with basic error handling and logging; not a hardened service.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from shared import __version__, format_docs, require_vector_store
from shared.config import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_VECTOR_STORE_PATH,
)
from shared.prompts import RAG_PROMPT_TEMPLATE

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LangChain RAG API",
    description="Tutorial RAG API with LangChain",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict origins if you expose this beyond localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QueryRequest(BaseModel):
    """Request model for RAG queries"""

    query: str = Field(..., min_length=1, max_length=1000, description="User question")
    k: int = Field(4, ge=1, le=10, description="Number of documents to retrieve")
    architecture: Literal["simple"] = Field(
        "simple", description="Only the simple tutorial chain is implemented"
    )

    class Config:
        json_schema_extra = {"example": {"query": "What is RAG?", "k": 4, "architecture": "simple"}}


class QueryResponse(BaseModel):
    """Response model for RAG queries"""

    answer: str = Field(..., description="Generated answer")
    sources: list[str] = Field(..., description="Retrieved document sources")
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
            embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
            self.vectorstore = require_vector_store(OPENAI_VECTOR_STORE_PATH, embeddings)

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

            # Initialize LLM
            self.llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)

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
        status="healthy", version=__version__, vector_store_loaded=state.initialized
    )


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, app_state: AppState = Depends(get_app_state)):
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

        retriever = app_state.vectorstore.as_retriever(search_kwargs={"k": request.k})
        docs = retriever.invoke(request.query)
        answer = (RAG_PROMPT_TEMPLATE | app_state.llm | StrOutputParser()).invoke(
            {"context": format_docs(docs), "input": request.query}
        )
        sources = [doc.metadata.get("source", "unknown") for doc in docs]

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        logger.info(f"Query completed in {latency_ms:.2f}ms")

        return QueryResponse(
            answer=answer, sources=sources, latency_ms=latency_ms, architecture=request.architecture
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/architectures")
async def list_architectures():
    """List available RAG architectures"""
    return {
        "architectures": [
            {
                "name": "simple",
                "description": "Basic RAG with similarity search (the only implemented chain)",
                "latency": "~2s",
                "cost": "Low",
            },
        ]
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
