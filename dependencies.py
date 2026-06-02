"""
dependencies.py
===============
Shared FastAPI dependencies used across multiple routers.

Think of this file as the "shared services" layer.
Any logic that multiple routes need — pagination, auth, DB access,
rate limiting, current user — lives here as a Depends() function.

Why a dedicated file?
- Prevents circular imports (routers import this, not main.py)
- Single place to update shared logic
- Easy to mock in tests
"""

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Generator

from fastapi import Depends, Query

from config import Settings, get_settings
from database import FakeDatabase

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Database Session Dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> Generator[FakeDatabase, None, None]:
    """
    Provides a database session to route functions.

    This is a GENERATOR dependency — it uses 'yield' to:
    1. Create the session (before yield)
    2. Hand it to the route function (at yield)
    3. Clean up after the route completes (after yield / in finally)

    FastAPI guarantees the finally block runs even if an exception
    occurs inside the route. This prevents connection leaks.

    Usage in routes:
        async def my_route(db: DbSession):

    When Day 5 arrives, we replace FakeDatabase() with a real
    SQLAlchemy/async session — routes don't need to change at all.
    """
    db = FakeDatabase()
    try:
        yield db       # ← Route function receives this db object
    finally:
        db.close()     # ← Always runs, even on exceptions


# Type alias for cleaner route signatures
# Instead of: db: FakeDatabase = Depends(get_db)
# You can write: db: DbSession
DbSession = Annotated[FakeDatabase, Depends(get_db)]


# ─────────────────────────────────────────────────────────────────────────────
# Pagination Dependency
# ─────────────────────────────────────────────────────────────────────────────

def get_pagination(
    skip: int = Query(
        default=0,
        ge=0,
        description="Number of records to skip (offset for pagination)",
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of records to return",
    ),
) -> dict:
    """
    Extracts and validates pagination parameters from query string.

    Centralizes pagination logic so every list endpoint has
    consistent behavior and validation without copy-pasting.

    Usage in routes:
        async def list_items(pagination: Pagination):

    Returns:
        Dict with 'skip' and 'limit' keys

    Example URL: /items/?skip=20&limit=5
    """
    return {"skip": skip, "limit": limit}


# Type alias for pagination
Pagination = Annotated[dict, Depends(get_pagination)]


# ─────────────────────────────────────────────────────────────────────────────
# Settings Dependency
# ─────────────────────────────────────────────────────────────────────────────

# Type alias for settings
# Instead of: settings: Settings = Depends(get_settings)
# You can write: settings: AppSettings
AppSettings = Annotated[Settings, Depends(get_settings)]


# ─────────────────────────────────────────────────────────────────────────────
# Background Task Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def log_item_action(action: str, item_id: int, details: dict) -> None:
    """
    Background task: write an action log entry to the log file.

    Moved here from main.py — background task helpers belong in
    dependencies or a dedicated tasks.py, not in main.py.

    Args:
        action: Operation type (CREATE, UPDATE, DELETE)
        item_id: ID of the affected item
        details: Additional context (name, changed fields, etc.)
    """
    from datetime import datetime  # Local import to avoid any circular issues

    settings = get_settings()
    log_file = Path(settings.log_file)
    log_file.parent.mkdir(exist_ok=True)

    await asyncio.sleep(0.01)  # Simulates async I/O

    timestamp = datetime.now().isoformat()
    log_entry = (
        f"{timestamp} | ACTION={action} | item_id={item_id} | details={details}\n"
    )

    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        logger.info(f"Background log: {action} item_id={item_id}")
    except OSError as e:
        logger.error(f"Failed to write log entry: {e}")


async def simulate_notification(item_name: str, action: str) -> None:
    """
    Background task: simulates sending a notification.

    In production: SendGrid email, Slack webhook, SNS message, etc.
    """
    await asyncio.sleep(0.05)
    logger.info(f"[Simulated Notification] '{item_name}' was {action}.")