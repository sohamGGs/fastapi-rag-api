"""
main.py
=======
Application entry point and factory.

This file's ONLY responsibilities:
1. Create the FastAPI app instance
2. Register routers
3. Register global exception handlers
4. Define lifecycle events (startup/shutdown)

It should NOT contain business logic, route handlers, or data access.
Think of it as the "front door" — it points to everything else.

Import tree (main.py is always the root — nothing imports it):
  main.py
  ├── config.py (settings)
  ├── routers/items.py
  │   ├── models.py
  │   ├── dependencies.py
  │   │   ├── config.py
  │   │   └── database.py
  └── (future: routers/documents.py, routers/queries.py, etc.)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from routers import items as items_router

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────

settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (replaces @app.on_event — modern FastAPI pattern)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Everything before `yield` runs on STARTUP.
    Everything after `yield` runs on SHUTDOWN.

    This is the modern replacement for @app.on_event("startup").
    On Day 4+, we'll load embedding models here.
    On Day 5+, we'll initialize ChromaDB here.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    logger.info(f"🚀 {settings.app_name} v{settings.app_version} starting...")
    logger.info(f"   Debug mode: {settings.debug}")
    logger.info(f"   Log file: {settings.log_file}")
    logger.info(f"   Docs: http://127.0.0.1:8000/docs")

    yield  # ← Application runs here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info(f"🛑 {settings.app_name} shutting down...")


# ─────────────────────────────────────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    description=(
        "A production-grade REST API and RAG (Retrieval-Augmented Generation) system. "
        "Built step-by-step as a 7-day structured learning project.\n\n"
        "**Days 1-3:** FastAPI foundations, CRUD, async, error handling, project structure\n"
        "**Days 4-6:** LangChain, embeddings, ChromaDB, RAG pipeline\n"
        "**Day 7:** Deployment with Docker"
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────
# CORS: Allows frontend apps (React, Vue) running on different ports/domains
# to call your API. Without this, browsers block cross-origin requests.

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Global Exception Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for any unhandled exceptions.

    Without this, FastAPI returns a raw 500 error that might expose
    internal details. This ensures all errors return our clean JSON format.

    In production, you'd also send an alert to PagerDuty, Sentry, etc.
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: "
        f"{type(exc).__name__}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
                "status_code": 500,
            }
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Router Registration
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(items_router.router)

# Future routers (Days 4-6):
# from routers import documents, queries
# app.include_router(documents.router)
# app.include_router(queries.router)


# ─────────────────────────────────────────────────────────────────────────────
# Root & Health Endpoints
# ─────────────────────────────────────────────────────────────────────────────
# These live in main.py (not a router) because they represent the app itself,
# not any specific resource domain.

@app.get("/", tags=["Root"], include_in_schema=False)
async def root() -> dict:
    """Redirect hint for the API root."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "health": "/ping",
    }


@app.get("/ping", tags=["Health"], summary="Liveness check")
async def ping() -> dict:
    """
    Liveness probe endpoint.
    Returns immediately with no external dependencies.
    Used by load balancers and monitoring tools.
    """
    from datetime import datetime
    return {
        "message": "pong",
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "timestamp": datetime.now().isoformat(),
    }