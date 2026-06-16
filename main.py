"""
main.py
=======
Application entry point.

Registers routers, configures middleware, defines health check.
No ML imports here — this file should always start cleanly
regardless of which packages are installed.
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# These imports will work even if ML packages aren't installed yet
from routers import ask, search

app = FastAPI(
    title="FastAPI RAG API",
    description="""
A Retrieval-Augmented Generation API.

**Quick start:**
1. Run `python ingest.py` to populate the knowledge base
2. POST `/ask` with a question to get a grounded answer
3. POST `/search` to run raw semantic search
    """,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(ask.router)


@app.get("/ping", tags=["Health"])
async def ping() -> dict:
    """
    Liveness check. Returns immediately with no external dependencies.
    Use GET /search/info or GET /ask/info for deep health checks.
    """
    return {
        "status": "healthy",
        "version": app.version,
        "message": "FastAPI RAG API is running",
    }