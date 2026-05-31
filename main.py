"""
FastAPI RAG Application - Day 1
================================
Building our first endpoints: ping, path params, query params.
"""

from fastapi import FastAPI, HTTPException, Query
from typing import Optional

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────

app = FastAPI(
    title="FastAPI RAG API",
    description="A production-grade RAG API built step by step.",
    version="0.1.0",
)

# ─────────────────────────────────────────────
# In-Memory "Database"
# We're using a plain Python list for now.
# Day 2 will introduce proper database patterns.
# ─────────────────────────────────────────────

items_db: list[dict] = [
    {"id": 1, "name": "Laptop", "price": 999.99, "in_stock": True},
    {"id": 2, "name": "Mouse", "price": 29.99, "in_stock": True},
    {"id": 3, "name": "Monitor", "price": 399.99, "in_stock": False},
]

# ─────────────────────────────────────────────
# SECTION 1: Health Check
# ─────────────────────────────────────────────

@app.get("/ping", tags=["Health"])
async def ping():
    """
    Health check endpoint.
    Used by load balancers and monitoring tools to verify the API is alive.
    Returns a simple pong response.
    """
    return {"message": "pong", "status": "healthy"}


# ─────────────────────────────────────────────
# SECTION 2: Path Parameters
# ─────────────────────────────────────────────

@app.get("/items/{item_id}", tags=["Items"])
async def get_item_by_id(item_id: int):
    """
    Retrieve a single item by its ID.

    - **item_id**: The unique integer ID of the item (path parameter)

    FastAPI automatically:
    1. Extracts item_id from the URL
    2. Converts it to int (or returns 422 if it's not a valid int)
    3. Passes it to this function

    Raises 404 if item is not found — this is the correct HTTP convention.
    """
    # Search our in-memory list for a matching ID
    for item in items_db:
        if item["id"] == item_id:
            return item

    # HTTP 404 = "Not Found" — the correct status code when a resource doesn't exist
    raise HTTPException(
        status_code=404,
        detail=f"Item with id {item_id} not found."
    )


# ─────────────────────────────────────────────
# SECTION 3: Query Parameters
# ─────────────────────────────────────────────

@app.get("/items/", tags=["Items"])
async def get_items(
    name: Optional[str] = Query(default=None, description="Filter items by name (case-insensitive)"),
    in_stock: Optional[bool] = Query(default=None, description="Filter by stock availability"),
    limit: int = Query(default=10, ge=1, le=100, description="Max number of results to return"),
):
    """
    Retrieve a list of items with optional filters.

    - **name**: Optional name filter (case-insensitive partial match)
    - **in_stock**: Optional filter for stock availability
    - **limit**: Max number of results (1-100, default 10)

    Query parameters are optional by default.
    Example: GET /items/?name=laptop&in_stock=true&limit=5
    """
    results = items_db.copy()

    # Apply name filter if provided
    if name is not None:
        results = [item for item in results if name.lower() in item["name"].lower()]

    # Apply stock filter if provided
    if in_stock is not None:
        results = [item for item in results if item["in_stock"] == in_stock]

    # Apply limit
    results = results[:limit]

    return {
        "count": len(results),
        "items": results,
    }