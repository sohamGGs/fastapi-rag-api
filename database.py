"""
database.py
===========
Simulated database layer for Day 3.

Right now this is an in-memory Python structure.
On Day 5, this file will be replaced with a real
SQLAlchemy / ChromaDB / async database setup.

The key design choice: ALL data access goes through this module.
Routes NEVER access `items_db` directly — they use the DB session.
This makes it trivial to swap the backend later without touching routes.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# In-Memory Storage
# ─────────────────────────────────────────────────────────────────────────────
# Module-level variables act as our "database tables".
# These persist for the lifetime of the running process.

_items_store: list[dict] = [
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

_next_id: int = 4


# ─────────────────────────────────────────────────────────────────────────────
# Database Session Class
# ─────────────────────────────────────────────────────────────────────────────

class FakeDatabase:
    """
    Simulated database session.

    Mimics the interface of a real database ORM session (like SQLAlchemy).
    Each method represents a database operation.

    Why a class instead of functions?
    - Mirrors real DB session objects (SQLAlchemy Session, Motor AsyncIOMotorClient)
    - Can hold connection state (in a real DB, the active transaction)
    - Easy to mock in tests: replace with FakeDatabase() that returns test data

    On Day 5, this class will be replaced or wrapped with real DB calls.
    The dependency injection means routes won't need to change at all.
    """

    def __init__(self) -> None:
        """
        Initialize the database session.
        In a real DB, this is where you'd get a connection from the pool.
        """
        logger.debug("FakeDatabase session opened")

    # ── Read Operations ───────────────────────────────────────────────────────

    def get_all_items(
        self,
        skip: int = 0,
        limit: int = 10,
        name_filter: Optional[str] = None,
        in_stock_filter: Optional[bool] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> list[dict]:
        """
        Retrieve items with optional filtering and pagination.

        Args:
            skip: Number of records to skip (offset)
            limit: Maximum records to return
            name_filter: Case-insensitive partial name match
            in_stock_filter: Filter by stock status
            min_price: Minimum price inclusive
            max_price: Maximum price inclusive

        Returns:
            List of item dicts matching the filters
        """
        results = _items_store.copy()

        if name_filter is not None:
            results = [
                item for item in results
                if name_filter.lower() in item["name"].lower()
            ]

        if in_stock_filter is not None:
            results = [
                item for item in results
                if item["in_stock"] == in_stock_filter
            ]

        if min_price is not None:
            results = [item for item in results if item["price"] >= min_price]

        if max_price is not None:
            results = [item for item in results if item["price"] <= max_price]

        return results[skip: skip + limit]

    def get_item_by_id(self, item_id: int) -> Optional[dict]:
        """
        Find a single item by ID.

        Returns:
            Item dict if found, None otherwise
        """
        for item in _items_store:
            if item["id"] == item_id:
                return item
        return None

    def get_item_by_name(self, name: str) -> Optional[dict]:
        """
        Find a single item by exact name (case-insensitive).

        Returns:
            Item dict if found, None otherwise
        """
        name_lower = name.lower()
        for item in _items_store:
            if item["name"].lower() == name_lower:
                return item
        return None

    # ── Write Operations ──────────────────────────────────────────────────────

    def create_item(self, name: str, price: float, in_stock: bool) -> dict:
        """
        Insert a new item into the store.

        Args:
            name: Item name (already validated by Pydantic)
            price: Item price (already validated)
            in_stock: Stock status

        Returns:
            The newly created item dict with server-generated ID
        """
        global _next_id

        new_item = {
            "id": _next_id,
            "name": name,
            "price": price,
            "in_stock": in_stock,
            "created_at": datetime.now().isoformat(),
        }

        _items_store.append(new_item)
        _next_id += 1

        logger.debug(f"Created item id={new_item['id']}")
        return new_item

    def update_item(self, item_id: int, update_data: dict) -> Optional[dict]:
        """
        Update fields of an existing item.

        Args:
            item_id: ID of the item to update
            update_data: Dict of fields to update (only provided fields)

        Returns:
            Updated item dict, or None if not found
        """
        for index, item in enumerate(_items_store):
            if item["id"] == item_id:
                updated = {**item, **update_data}
                _items_store[index] = updated
                logger.debug(f"Updated item id={item_id} fields={list(update_data.keys())}")
                return updated
        return None

    def delete_item(self, item_id: int) -> Optional[dict]:
        """
        Remove an item from the store.

        Args:
            item_id: ID of the item to delete

        Returns:
            The deleted item dict, or None if not found
        """
        for index, item in enumerate(_items_store):
            if item["id"] == item_id:
                deleted = _items_store.pop(index)
                logger.debug(f"Deleted item id={item_id}")
                return deleted
        return None

    def get_stats(self) -> dict:
        """Return summary statistics about the inventory."""
        if not _items_store:
            return {
                "total_items": 0,
                "in_stock_count": 0,
                "out_of_stock_count": 0,
                "average_price": 0.0,
                "most_expensive": None,
                "least_expensive": None,
            }

        prices = [item["price"] for item in _items_store]
        in_stock = [item for item in _items_store if item["in_stock"]]

        return {
            "total_items": len(_items_store),
            "in_stock_count": len(in_stock),
            "out_of_stock_count": len(_items_store) - len(in_stock),
            "average_price": round(sum(prices) / len(prices), 2),
            "most_expensive": max(_items_store, key=lambda x: x["price"])["name"],
            "least_expensive": min(_items_store, key=lambda x: x["price"])["name"],
        }

    def close(self) -> None:
        """
        Close the database session.
        In a real DB, this returns the connection to the pool.
        Called automatically by the dependency's finally block.
        """
        logger.debug("FakeDatabase session closed")