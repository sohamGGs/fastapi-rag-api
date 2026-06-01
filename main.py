"""
FastAPI RAG Application — Day 2
================================
Adds to Day 1:
  - Async patterns with proper await usage
  - Background tasks (async logging to file)
  - Comprehensive error handling
  - Custom exception handlers
  - Meaningful HTTP status codes
  - Duplicate detection (400 Conflict)
  - Response models controlling output shape
  - Full docstrings and type hints throughout
"""

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
# Python's built-in logging module — separate from our background task logging.
# This logs server-side events to the console (useful during development).
# In production, you'd ship these logs to a service like Datadog or CloudWatch.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Directory Setup
# ─────────────────────────────────────────────────────────────────────────────

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)  # Create logs/ folder if it doesn't exist
LOG_FILE = LOGS_DIR / "app.log"

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App Initialization
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FastAPI RAG API",
    description=(
        "A production-grade REST API and RAG system. "
        "Built step-by-step as a 7-day learning project."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ─────────────────────────────────────────────────────────────────────────────
# Define our own exception classes instead of always using HTTPException directly.
# This makes the code more readable: raise ItemNotFoundError(1) vs
# raise HTTPException(status_code=404, detail="Item with id=1 not found.")
#
# It also means if we ever want to change the error format globally,
# we change it in ONE place (the exception handler below).

class ItemNotFoundError(Exception):
    """Raised when an item with the given ID does not exist."""

    def __init__(self, item_id: int):
        self.item_id = item_id
        super().__init__(f"Item with id={item_id} not found.")


class DuplicateItemError(Exception):
    """Raised when trying to create an item with a name that already exists."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(f"An item named '{name}' already exists.")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Exception Handlers
# ─────────────────────────────────────────────────────────────────────────────
# These functions "catch" our custom exceptions anywhere in the app and
# return a consistent, clean JSON error response.
#
# The @app.exception_handler decorator registers them with FastAPI.
# When Python raises ItemNotFoundError anywhere, FastAPI intercepts it
# and calls this function instead of crashing.

@app.exception_handler(ItemNotFoundError)
async def item_not_found_handler(request: Request, exc: ItemNotFoundError) -> JSONResponse:
    """
    Handles ItemNotFoundError globally.
    Returns a consistent 404 JSON response.
    """
    logger.warning(f"404 Not Found: Item id={exc.item_id} | Path: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": {
                "code": "ITEM_NOT_FOUND",
                "message": str(exc),
                "status_code": 404,
            }
        },
    )


@app.exception_handler(DuplicateItemError)
async def duplicate_item_handler(request: Request, exc: DuplicateItemError) -> JSONResponse:
    """
    Handles DuplicateItemError globally.
    Returns a consistent 400 JSON response.
    """
    logger.warning(f"400 Duplicate: Item name='{exc.name}' | Path: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": "DUPLICATE_ITEM",
                "message": str(exc),
                "status_code": 400,
            }
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────────────────

class ItemCreate(BaseModel):
    """
    Schema for creating a new item.
    Used as the request body for POST /items/.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the item",
    )
    price: float = Field(
        ...,
        gt=0,
        description="Price in USD, must be greater than 0",
    )
    in_stock: bool = Field(
        default=True,
        description="Whether the item is currently available",
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_whitespace(cls, value: str) -> str:
        """Strip whitespace and reject blank names."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Name cannot be blank or whitespace only.")
        return stripped  # Store the cleaned version

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Mechanical Keyboard",
                "price": 149.99,
                "in_stock": True,
            }
        }
    }


class ItemUpdate(BaseModel):
    """
    Schema for updating an existing item.
    All fields are Optional to support partial updates.
    """

    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    price: Optional[float] = Field(default=None, gt=0)
    in_stock: Optional[bool] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "price": 129.99,
                "in_stock": False,
            }
        }
    }


class ItemResponse(BaseModel):
    """
    Schema for item data returned in API responses.

    This is the PUBLIC shape of an item.
    Any internal fields (cost, audit info, etc.) would NOT appear here,
    even if the underlying data has them — FastAPI filters them out.
    """

    id: int
    name: str
    price: float
    in_stock: bool
    created_at: str  # ISO 8601 timestamp string


class DeleteResponse(BaseModel):
    """Schema for delete operation confirmation."""

    message: str
    deleted_item_id: int


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory "Database"
# ─────────────────────────────────────────────────────────────────────────────

items_db: list[dict] = [
    {
        "id": 1,
        "name": "Laptop",
        "price": 999.99,
        "in_stock": True,
        "created_at": "2025-05-21T08:00:00",
    },
    {
        "id": 2,
        "name": "Mouse",
        "price": 29.99,
        "in_stock": True,
        "created_at": "2025-05-21T08:01:00",
    },
    {
        "id": 3,
        "name": "Monitor",
        "price": 399.99,
        "in_stock": False,
        "created_at": "2025-05-21T08:02:00",
    },
]

next_id: int = 4  # Auto-increment counter (a real DB handles this automatically)


# ─────────────────────────────────────────────────────────────────────────────
# Background Task Functions
# ─────────────────────────────────────────────────────────────────────────────
# These are regular functions called *after* the response is sent.
# They run in the same process/thread pool — not a separate worker.
# For heavy tasks (video processing, ML training), use Celery instead.

async def log_item_action(action: str, item_id: int, details: dict) -> None:
    """
    Async background task: write an action log entry to app.log.

    Called after item creation, update, or deletion.
    The client never waits for this — it happens after the response is sent.

    Args:
        action: The operation performed (e.g., "CREATE", "UPDATE", "DELETE")
        item_id: The ID of the item affected
        details: Additional context to log
    """
    # Simulate a small async delay (represents real async I/O like a DB write)
    # In production, you'd use aiofiles for async file writes
    await asyncio.sleep(0.01)

    timestamp = datetime.now().isoformat()
    log_entry = (
        f"{timestamp} | ACTION={action} | item_id={item_id} | details={details}\n"
    )

    # Write to our log file
    # Note: In production, use 'aiofiles' for truly async file I/O
    # pip install aiofiles → async with aiofiles.open(...) as f: await f.write(...)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # Also log to console so you can see it happening during development
    logger.info(f"Background log written: {action} item_id={item_id}")


async def simulate_notification(item_name: str, action: str) -> None:
    """
    Simulates sending a notification (email/Slack/webhook).

    In a real system, this would call an email service like SendGrid
    or post to a Slack webhook. Here we just simulate the delay.

    Args:
        item_name: Name of the item involved
        action: What happened to it
    """
    await asyncio.sleep(0.05)  # Simulate network call delay
    logger.info(f"[Simulated Notification] Item '{item_name}' was {action}.")


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def find_item_by_id(item_id: int) -> dict:
    """
    Search items_db for an item with the given ID.

    Args:
        item_id: The ID to look up

    Returns:
        The item dict if found

    Raises:
        ItemNotFoundError: If no item with that ID exists
    """
    for item in items_db:
        if item["id"] == item_id:
            return item
    raise ItemNotFoundError(item_id)


def find_item_by_name(name: str) -> Optional[dict]:
    """
    Search items_db for an item with the given name (case-insensitive).

    Args:
        name: The name to look up

    Returns:
        The item dict if found, None otherwise
    """
    name_lower = name.lower()
    for item in items_db:
        if item["name"].lower() == name_lower:
            return item
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Lifecycle Events
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """
    Runs once when the application starts.
    Good place for: DB connections, loading ML models, cache warming.
    On Day 4+, we'll load our embedding model here.
    """
    logger.info("🚀 FastAPI RAG API starting up...")
    logger.info(f"📁 Log file: {LOG_FILE.resolve()}")
    logger.info(f"📦 Items in memory: {len(items_db)}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    Runs once when the application shuts down (Ctrl+C).
    Good place for: closing DB connections, flushing buffers.
    """
    logger.info("🛑 FastAPI RAG API shutting down...")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

# ── Health ────────────────────────────────────────────────────────────────────

@app.get(
    "/ping",
    tags=["Health"],
    summary="Health check",
    response_description="Simple liveness check response",
)
async def ping() -> dict:
    """
    Liveness check endpoint.

    Used by load balancers, Kubernetes probes, and uptime monitors
    to verify the API process is running and responsive.
    Should always return quickly — no DB or external calls here.
    """
    return {
        "message": "pong",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
    }


# ── READ ──────────────────────────────────────────────────────────────────────

@app.get(
    "/items/",
    response_model=list[ItemResponse],
    tags=["Items"],
    summary="List all items",
)
async def get_items(
    name: Optional[str] = Query(
        default=None,
        description="Case-insensitive partial name filter",
        min_length=1,
    ),
    in_stock: Optional[bool] = Query(
        default=None,
        description="Filter by stock availability",
    ),
    min_price: Optional[float] = Query(
        default=None,
        gt=0,
        description="Minimum price filter",
    ),
    max_price: Optional[float] = Query(
        default=None,
        gt=0,
        description="Maximum price filter",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of items to return",
    ),
    skip: int = Query(
        default=0,
        ge=0,
        description="Number of items to skip (for pagination)",
    ),
) -> list[dict]:
    """
    Retrieve a paginated, filterable list of items.

    Supports filtering by name, stock status, and price range.
    Supports pagination via `skip` and `limit` query parameters.

    Example: `/items/?name=laptop&in_stock=true&min_price=100&limit=5`
    """
    results = items_db.copy()

    # Apply filters
    if name is not None:
        results = [item for item in results if name.lower() in item["name"].lower()]

    if in_stock is not None:
        results = [item for item in results if item["in_stock"] == in_stock]

    if min_price is not None:
        results = [item for item in results if item["price"] >= min_price]

    if max_price is not None:
        results = [item for item in results if item["price"] <= max_price]

    # Apply pagination (skip then limit)
    return results[skip : skip + limit]


@app.get(
    "/items/{item_id}",
    response_model=ItemResponse,
    tags=["Items"],
    summary="Get item by ID",
    responses={
        404: {
            "description": "Item not found",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "ITEM_NOT_FOUND",
                            "message": "Item with id=99 not found.",
                            "status_code": 404,
                        }
                    }
                }
            },
        }
    },
)
async def get_item(item_id: int) -> dict:
    """
    Retrieve a single item by its unique ID.

    Raises a 404 error with a structured error body if not found.
    The `responses` parameter above documents this in Swagger UI.
    """
    # find_item_by_id raises ItemNotFoundError if not found.
    # Our registered exception handler catches it and returns clean 404 JSON.
    return find_item_by_id(item_id)


# ── CREATE ────────────────────────────────────────────────────────────────────

@app.post(
    "/items/",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Items"],
    summary="Create a new item",
    responses={
        400: {
            "description": "Duplicate item name",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "DUPLICATE_ITEM",
                            "message": "An item named 'Laptop' already exists.",
                            "status_code": 400,
                        }
                    }
                }
            },
        }
    },
)
async def create_item(
    item: ItemCreate,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Create a new item.

    **Key behaviors:**
    - Returns `201 Created` (not 200) on success
    - Rejects duplicate names with `400 Bad Request`
    - Immediately returns the created item
    - Logs the action in the background (client doesn't wait for this)
    - Simulates sending a notification in the background

    **Background Tasks:**
    The logging and notification happen *after* the response is sent.
    The client gets their `201` response instantly, regardless of how
    long the background tasks take.
    """
    global next_id

    # ── Duplicate Check ──────────────────────────────────────────────────────
    # Business rule: item names must be unique.
    # We raise DuplicateItemError, which our handler converts to 400.
    existing = find_item_by_name(item.name)
    if existing is not None:
        raise DuplicateItemError(item.name)

    # ── Create the Item ──────────────────────────────────────────────────────
    new_item = {
        "id": next_id,
        "name": item.name,       # Already validated and stripped by Pydantic
        "price": item.price,
        "in_stock": item.in_stock,
        "created_at": datetime.now().isoformat(),
    }

    items_db.append(new_item)
    next_id += 1

    # ── Register Background Tasks ────────────────────────────────────────────
    # add_task(function, *args) — the function runs AFTER the response is sent.
    # We add TWO background tasks — both will run sequentially after response.
    background_tasks.add_task(
        log_item_action,
        action="CREATE",
        item_id=new_item["id"],
        details={"name": new_item["name"], "price": new_item["price"]},
    )
    background_tasks.add_task(
        simulate_notification,
        item_name=new_item["name"],
        action="created",
    )

    logger.info(f"Item created: id={new_item['id']} name='{new_item['name']}'")
    return new_item


# ── UPDATE ────────────────────────────────────────────────────────────────────

@app.put(
    "/items/{item_id}",
    response_model=ItemResponse,
    tags=["Items"],
    summary="Update an existing item",
)
async def update_item(
    item_id: int,
    item_update: ItemUpdate,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Update one or more fields of an existing item.

    - Partial updates supported: only send the fields you want to change
    - Raises 404 if the item doesn't exist
    - Raises 400 if the new name conflicts with another existing item
    - Logs the update in the background
    """
    # Will raise ItemNotFoundError (→ 404) if not found
    existing_item = find_item_by_id(item_id)

    # Get only the fields that were actually sent (exclude unset)
    update_data = item_update.model_dump(exclude_unset=True)

    if not update_data:
        # Client sent an empty body — nothing to update
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update. Send at least one field.",
        )

    # If name is being changed, check for duplicates
    if "name" in update_data:
        name_conflict = find_item_by_name(update_data["name"])
        if name_conflict is not None and name_conflict["id"] != item_id:
            raise DuplicateItemError(update_data["name"])

    # Apply updates: merge existing with changes
    updated_item = {**existing_item, **update_data}

    # Update in our "database"
    for index, item in enumerate(items_db):
        if item["id"] == item_id:
            items_db[index] = updated_item
            break

    # Background: log the update
    background_tasks.add_task(
        log_item_action,
        action="UPDATE",
        item_id=item_id,
        details=update_data,  # Log exactly what changed
    )

    logger.info(f"Item updated: id={item_id} changes={update_data}")
    return updated_item


# ── DELETE ────────────────────────────────────────────────────────────────────

@app.delete(
    "/items/{item_id}",
    response_model=DeleteResponse,
    tags=["Items"],
    summary="Delete an item",
)
async def delete_item(
    item_id: int,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Permanently delete an item by ID.

    - Returns a confirmation with the deleted item's ID
    - Raises 404 if the item doesn't exist
    - Logs the deletion in the background

    Note: In production systems, consider "soft delete" — marking items
    as deleted (with a `deleted_at` timestamp) rather than actually
    removing them. This preserves audit history and allows recovery.
    """
    # Will raise ItemNotFoundError (→ 404) if not found
    item_to_delete = find_item_by_id(item_id)

    # Remove from our "database"
    items_db.remove(item_to_delete)

    # Background: log the deletion
    background_tasks.add_task(
        log_item_action,
        action="DELETE",
        item_id=item_id,
        details={"name": item_to_delete["name"]},
    )
    background_tasks.add_task(
        simulate_notification,
        item_name=item_to_delete["name"],
        action="deleted",
    )

    logger.info(f"Item deleted: id={item_id} name='{item_to_delete['name']}'")
    return {
        "message": f"Item '{item_to_delete['name']}' deleted successfully.",
        "deleted_item_id": item_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stats Endpoint (Bonus from Day 1 extensions)
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/items/stats/summary",
    tags=["Items"],
    summary="Get inventory statistics",
)
async def get_stats() -> dict:
    """
    Returns summary statistics about the current inventory.

    Useful for dashboards and monitoring.
    Note the path is /items/stats/summary — using /items/stats alone
    would conflict with /items/{item_id} routing.
    """
    if not items_db:
        return {
            "total_items": 0,
            "in_stock_count": 0,
            "out_of_stock_count": 0,
            "average_price": 0.0,
            "most_expensive": None,
            "least_expensive": None,
        }

    prices = [item["price"] for item in items_db]
    in_stock_items = [item for item in items_db if item["in_stock"]]

    return {
        "total_items": len(items_db),
        "in_stock_count": len(in_stock_items),
        "out_of_stock_count": len(items_db) - len(in_stock_items),
        "average_price": round(sum(prices) / len(prices), 2),
        "most_expensive": max(items_db, key=lambda x: x["price"])["name"],
        "least_expensive": min(items_db, key=lambda x: x["price"])["name"],
    }