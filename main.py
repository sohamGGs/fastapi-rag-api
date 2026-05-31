"""
FastAPI RAG Application - Day 1
================================
Building our first endpoints with full CRUD operations.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
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
# Pydantic Models
# These define the "shape" of our data.
# ─────────────────────────────────────────────

class ItemCreate(BaseModel):
    """
    Model for creating a new item.
    Used as the request body for POST /items.

    'Create' models typically don't have an ID —
    the server generates the ID, the client shouldn't set it.
    """
    name: str = Field(..., min_length=1, max_length=100, description="Name of the item")
    price: float = Field(..., gt=0, description="Price must be greater than 0")
    in_stock: bool = Field(default=True, description="Whether the item is currently in stock")

    class Config:
        # This enables example data in Swagger UI
        json_schema_extra = {
            "example": {
                "name": "Mechanical Keyboard",
                "price": 149.99,
                "in_stock": True,
            }
        }


class ItemUpdate(BaseModel):
    """
    Model for updating an existing item.
    All fields are Optional — allows partial updates.

    This is the correct pattern: you shouldn't need to send
    ALL fields just to update ONE field.
    """
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    price: Optional[float] = Field(default=None, gt=0)
    in_stock: Optional[bool] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "price": 129.99,
                "in_stock": False,
            }
        }


class ItemResponse(BaseModel):
    """
    Model for returning item data in responses.
    Includes the server-generated ID.

    Having a separate Response model is best practice:
    it lets you control exactly what gets returned
    (e.g., you might hide internal fields like passwords).
    """
    id: int
    name: str
    price: float
    in_stock: bool


# ─────────────────────────────────────────────
# In-Memory "Database"
# ─────────────────────────────────────────────

items_db: list[dict] = [
    {"id": 1, "name": "Laptop", "price": 999.99, "in_stock": True},
    {"id": 2, "name": "Mouse", "price": 29.99, "in_stock": True},
    {"id": 3, "name": "Monitor", "price": 399.99, "in_stock": False},
]

# Simple counter for generating unique IDs
# (In a real DB, this is handled automatically)
next_id: int = 4


# ─────────────────────────────────────────────
# SECTION 1: Health Check
# ─────────────────────────────────────────────

@app.get("/ping", tags=["Health"])
async def ping():
    """
    Health check endpoint.
    Used by load balancers and monitoring tools to verify the API is alive.
    """
    return {"message": "pong", "status": "healthy"}


# ─────────────────────────────────────────────
# SECTION 2: READ Operations (GET)
# ─────────────────────────────────────────────

@app.get("/items/", response_model=list[ItemResponse], tags=["Items"])
async def get_items(
    name: Optional[str] = Query(default=None, description="Filter by name (case-insensitive)"),
    in_stock: Optional[bool] = Query(default=None, description="Filter by stock availability"),
    limit: int = Query(default=10, ge=1, le=100, description="Max results (1-100)"),
):
    """
    Retrieve all items with optional filtering.

    - Filter by name (partial, case-insensitive match)
    - Filter by stock status
    - Limit result count
    """
    results = items_db.copy()

    if name is not None:
        results = [item for item in results if name.lower() in item["name"].lower()]

    if in_stock is not None:
        results = [item for item in results if item["in_stock"] == in_stock]

    return results[:limit]


@app.get("/items/{item_id}", response_model=ItemResponse, tags=["Items"])
async def get_item(item_id: int):
    """
    Retrieve a single item by ID.

    Raises HTTP 404 if the item doesn't exist.
    """
    for item in items_db:
        if item["id"] == item_id:
            return item

    raise HTTPException(
        status_code=404,
        detail=f"Item with id={item_id} not found."
    )


# ─────────────────────────────────────────────
# SECTION 3: CREATE Operation (POST)
# ─────────────────────────────────────────────

@app.post("/items/", response_model=ItemResponse, status_code=201, tags=["Items"])
async def create_item(item: ItemCreate):
    """
    Create a new item.

    - Request body is validated automatically by Pydantic
    - Returns the created item with its server-generated ID
    - Status 201 = "Created" (more precise than 200 for creation)

    Note: `item` parameter tells FastAPI to expect a JSON request body
    matching the ItemCreate schema.
    """
    global next_id  # Access the module-level counter

    # Build the full item dict (adding server-generated ID)
    new_item = {
        "id": next_id,
        "name": item.name,
        "price": item.price,
        "in_stock": item.in_stock,
    }

    items_db.append(new_item)
    next_id += 1

    return new_item


# ─────────────────────────────────────────────
# SECTION 4: UPDATE Operation (PUT)
# ─────────────────────────────────────────────

@app.put("/items/{item_id}", response_model=ItemResponse, tags=["Items"])
async def update_item(item_id: int, item_update: ItemUpdate):
    """
    Update an existing item (partial update supported).

    - Only the fields you send will be updated
    - Fields not included remain unchanged
    - Raises 404 if the item doesn't exist

    This pattern (partial update with Optional fields) is
    sometimes called a PATCH, but we're using PUT here for simplicity.
    """
    for index, item in enumerate(items_db):
        if item["id"] == item_id:
            # .model_dump(exclude_unset=True) returns ONLY the fields
            # that were actually sent in the request body.
            # This is crucial for partial updates — we don't want to
            # overwrite existing values with None.
            update_data = item_update.model_dump(exclude_unset=True)

            # Merge: start with existing item, overwrite changed fields
            updated_item = {**item, **update_data}
            items_db[index] = updated_item

            return updated_item

    raise HTTPException(
        status_code=404,
        detail=f"Item with id={item_id} not found."
    )


# ─────────────────────────────────────────────
# SECTION 5: DELETE Operation (DELETE)
# ─────────────────────────────────────────────

@app.delete("/items/{item_id}", tags=["Items"])
async def delete_item(item_id: int):
    """
    Delete an item by ID.

    - Returns a confirmation message (not the deleted item)
    - Raises 404 if the item doesn't exist
    - In production, consider "soft delete" (marking as deleted)
      instead of actually removing data.
    """
    for index, item in enumerate(items_db):
        if item["id"] == item_id:
            deleted_item = items_db.pop(index)
            return {
                "message": f"Item '{deleted_item['name']}' (id={item_id}) deleted successfully."
            }

    raise HTTPException(
        status_code=404,
        detail=f"Item with id={item_id} not found."
    )