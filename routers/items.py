"""
routers/items.py
================
All item-related API endpoints.

This module is a self-contained "department" for the items resource.
It knows nothing about main.py — it only imports from:
  - models.py   (Pydantic schemas)
  - dependencies.py  (shared Depends functions)

Import tree (no cycles):
  main.py → routers/items.py → dependencies.py → database.py
                              ↘ models.py
                              ↘ config.py
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from dependencies import (
    AppSettings,
    DbSession,
    Pagination,
    log_item_action,
    simulate_notification,
)
from models import (
    DeleteResponse,
    ItemCreate,
    ItemResponse,
    ItemUpdate,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Router Configuration
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/items",
    tags=["Items"],
    # Router-level responses: these appear in Swagger for ALL routes in this router
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper — raises clean 404 with consistent format
# ─────────────────────────────────────────────────────────────────────────────

def _require_item(db: FakeDatabase, item_id: int) -> dict:
    """
    Fetch item by ID or raise a structured 404 HTTPException.

    Private helper (_name convention) — used only within this router.
    Centralizes the "not found" logic so we don't repeat it in every route.
    """
    from database import FakeDatabase  # avoid re-import at module level

    item = db.get_item_by_id(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "ITEM_NOT_FOUND",
                "message": f"Item with id={item_id} not found.",
                "status_code": 404,
            },
        )
    return item


# ─────────────────────────────────────────────────────────────────────────────
# READ Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=list[ItemResponse],
    summary="List items",
)
async def get_items(
    db: DbSession,                  # ← Injected: database session
    pagination: Pagination,         # ← Injected: {skip, limit} from query params
    settings: AppSettings,          # ← Injected: application config
    name: Optional[str] = Query(default=None, description="Name filter"),
    in_stock: Optional[bool] = Query(default=None, description="Stock filter"),
    min_price: Optional[float] = Query(default=None, gt=0, description="Min price"),
    max_price: Optional[float] = Query(default=None, gt=0, description="Max price"),
) -> list[dict]:
    """
    Retrieve a paginated, filtered list of items.

    **Dependency Injection in action:**
    - `db` is provided by `get_db()` — no manual session management
    - `pagination` is provided by `get_pagination()` — validated skip/limit
    - `settings` is provided by `get_settings()` — app configuration

    The route function focuses purely on business logic.
    """
    logger.info(
        f"GET /items/ | skip={pagination['skip']} limit={pagination['limit']} "
        f"name={name} in_stock={in_stock}"
    )

    return db.get_all_items(
        skip=pagination["skip"],
        limit=pagination["limit"],
        name_filter=name,
        in_stock_filter=in_stock,
        min_price=min_price,
        max_price=max_price,
    )


@router.get(
    "/stats/summary",
    summary="Inventory statistics",
    # Note: This route is defined BEFORE /{item_id} to avoid path conflicts
)
async def get_stats(db: DbSession) -> dict:
    """
    Returns summary statistics for the current inventory.
    No pagination needed — always returns a single summary object.
    """
    return db.get_stats()


@router.get(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Get item by ID",
)
async def get_item(item_id: int, db: DbSession) -> dict:
    """
    Retrieve a single item by its unique integer ID.
    Returns 404 if no item with that ID exists.
    """
    return _require_item(db, item_id)


# ─────────────────────────────────────────────────────────────────────────────
# CREATE Route
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=ItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new item",
)
async def create_item(
    item: ItemCreate,
    db: DbSession,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Create a new item in the inventory.

    - Validates uniqueness of item name
    - Returns 201 with the created item (including server-generated ID)
    - Logs the action asynchronously in the background
    """
    # Duplicate check
    if db.get_item_by_name(item.name) is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "DUPLICATE_ITEM",
                "message": f"An item named '{item.name}' already exists.",
                "status_code": 400,
            },
        )

    # Create the item via the DB session
    new_item = db.create_item(
        name=item.name,
        price=item.price,
        in_stock=item.in_stock,
    )

    # Register background tasks (run after response is sent)
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

    return new_item


# ─────────────────────────────────────────────────────────────────────────────
# UPDATE Route
# ─────────────────────────────────────────────────────────────────────────────

@router.put(
    "/{item_id}",
    response_model=ItemResponse,
    summary="Update an item",
)
async def update_item(
    item_id: int,
    item_update: ItemUpdate,
    db: DbSession,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Update one or more fields of an existing item.

    Only the fields included in the request body are changed.
    Raises 404 if item doesn't exist, 400 if new name is a duplicate.
    """
    # Verify item exists
    _require_item(db, item_id)

    # Get only the fields that were actually sent
    update_data = item_update.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "EMPTY_UPDATE",
                "message": "No fields provided for update.",
                "status_code": 400,
            },
        )

    # Check name uniqueness if name is being changed
    if "name" in update_data:
        conflict = db.get_item_by_name(update_data["name"])
        if conflict is not None and conflict["id"] != item_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "DUPLICATE_ITEM",
                    "message": f"An item named '{update_data['name']}' already exists.",
                    "status_code": 400,
                },
            )

    updated_item = db.update_item(item_id, update_data)

    background_tasks.add_task(
        log_item_action,
        action="UPDATE",
        item_id=item_id,
        details=update_data,
    )

    return updated_item


# ─────────────────────────────────────────────────────────────────────────────
# DELETE Route
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/{item_id}",
    response_model=DeleteResponse,
    summary="Delete an item",
)
async def delete_item(
    item_id: int,
    db: DbSession,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Permanently delete an item by ID.
    Raises 404 if the item doesn't exist.
    """
    # Verify it exists first (raises 404 if not)
    item = _require_item(db, item_id)

    db.delete_item(item_id)

    background_tasks.add_task(
        log_item_action,
        action="DELETE",
        item_id=item_id,
        details={"name": item["name"]},
    )
    background_tasks.add_task(
        simulate_notification,
        item_name=item["name"],
        action="deleted",
    )

    return {
        "message": f"Item '{item['name']}' deleted successfully.",
        "deleted_item_id": item_id,
    }