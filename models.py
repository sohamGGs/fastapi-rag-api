"""
models.py
=========
All Pydantic models (schemas) for the application.

Separating models from routes is important because:
1. Multiple routers may use the same model
2. Models grow complex — they deserve their own file
3. Avoids circular imports (routes import models, not vice versa)

Naming conventions:
  - XCreate  → Request body for POST (creating a new resource)
  - XUpdate  → Request body for PUT/PATCH (updating a resource)
  - XResponse → Response body (what the client receives)
  - XInDB    → Full internal representation (includes hidden fields)
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Item Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ItemCreate(BaseModel):
    """Request body schema for creating a new item (POST /items/)."""

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
        """Strip leading/trailing whitespace and reject blank names."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Name cannot be blank or whitespace only.")
        return stripped

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
    Request body schema for updating an item (PUT /items/{id}).
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
    Response schema for item data.
    This is the PUBLIC shape — only safe fields are included.
    FastAPI automatically filters out any extra fields from the raw data.
    """

    id: int
    name: str
    price: float
    in_stock: bool
    created_at: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 1,
                "name": "Laptop",
                "price": 999.99,
                "in_stock": True,
                "created_at": "2025-05-23T09:00:00",
            }
        }
    }


class DeleteResponse(BaseModel):
    """Response schema for a successful delete operation."""

    message: str
    deleted_item_id: int


# ─────────────────────────────────────────────────────────────────────────────
# Shared / Generic Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PaginationParams(BaseModel):
    """
    Reusable pagination parameters.
    Used as a dependency response model — not a request body.
    """

    skip: int = Field(default=0, ge=0, description="Items to skip")
    limit: int = Field(default=10, ge=1, le=100, description="Max items to return")


class ErrorDetail(BaseModel):
    """Standard error response structure used across all error handlers."""

    code: str
    message: str
    status_code: int


class ErrorResponse(BaseModel):
    """Wrapper for all error responses — consistent format API-wide."""

    error: ErrorDetail