"""
routers/search.py
=================
Semantic search endpoint.

Current state: returns dummy results (Phase 1 mock).
Phase 2: replaced by real ChromaDB retrieval.

The route signature and response format NEVER change between phases —
only the internal implementation swaps out.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


# ── Request / Response Schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    source_filter: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How does backpropagation work?",
                "top_k": 3,
            }
        }
    }


class SearchResultItem(BaseModel):
    chunk: str
    score: float
    source: str
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    result_count: int
    mode: str  # "mock" or "real" — tells you which backend answered


# ── Dependency: Vector Store ──────────────────────────────────────────────────
# This function tries to import and use the real vector store.
# If ChromaDB / sentence-transformers aren't ready, it falls back
# to the mock implementation below.
# Phase 3: replace mock_search() with real_search().

def get_search_backend():
    """
    Returns the best available search function.

    Tries real ChromaDB first. Falls back to mock if unavailable.
    Logs clearly which mode is active so you always know.
    """
    try:
        from services.vector_store import search as real_search
        logger.info("Search backend: ChromaDB (real)")
        return real_search, "real"
    except Exception as e:
        logger.warning(f"Real search unavailable ({e}). Using mock backend.")
        return mock_search, "mock"


def mock_search(query: str, top_k: int = 3, source_filter=None) -> list[dict]:
    """
    Mock search — returns hardcoded results.
    Used when ChromaDB / embeddings are not yet available.

    Returns deterministic results so you can test the full
    API response pipeline without any ML infrastructure.
    """
    mock_results = [
        {
            "chunk": (
                "Machine learning is a subset of artificial intelligence "
                "that enables systems to learn from data without being "
                "explicitly programmed."
            ),
            "score": 0.92,
            "source": "ml_basics.txt",
            "chunk_index": 0,
        },
        {
            "chunk": (
                "Backpropagation is the algorithm used to train neural networks. "
                "It calculates gradients by working backwards through the network."
            ),
            "score": 0.87,
            "source": "ml_basics.txt",
            "chunk_index": 3,
        },
        {
            "chunk": (
                "FastAPI is a modern Python web framework for building APIs "
                "with automatic documentation and type validation."
            ),
            "score": 0.71,
            "source": "fastapi_guide.txt",
            "chunk_index": 1,
        },
    ]
    return mock_results[:top_k]


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=SearchResponse,
    summary="Semantic similarity search",
)
async def search_documents(request: SearchRequest) -> SearchResponse:
    """
    Find document chunks semantically similar to the query.

    Returns mock results until ChromaDB and embeddings are ready.
    The `mode` field in the response tells you which backend answered:
    - `"mock"` → dummy results, install dependencies to get real results
    - `"real"` → live ChromaDB results
    """
    search_fn, mode = get_search_backend()

    results_raw = search_fn(
        query=request.query,
        top_k=request.top_k,
        source_filter=request.source_filter,
    )

    results = [
        SearchResultItem(
            chunk=r["chunk"],
            score=r["score"],
            source=r["source"],
            chunk_index=r["chunk_index"],
        )
        for r in results_raw
    ]

    return SearchResponse(
        query=request.query,
        results=results,
        result_count=len(results),
        mode=mode,
    )


@router.get("/info", summary="Vector store statistics")
async def search_info() -> dict:
    """
    Returns info about the current search backend and collection stats.
    Use this to check whether the real backend is active.
    """
    try:
        from services.vector_store import get_collection_stats
        stats = get_collection_stats()
        return {"backend": "real", **stats}
    except Exception as e:
        return {
            "backend": "mock",
            "reason": str(e),
            "message": (
                "Install chromadb and sentence-transformers, "
                "then run python ingest.py to activate real search."
            ),
        }