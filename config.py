"""
config.py
=========
Centralized application configuration using Pydantic Settings.

How it works:
1. Pydantic reads environment variables (case-insensitive)
2. Falls back to .env file if the variable isn't in the environment
3. Falls back to the default value defined here
4. Validates types automatically (e.g., DEBUG=true → bool True)

This means the SAME code works in:
- Local dev: reads from .env file
- Docker: reads from container environment variables
- AWS Lambda / Cloud Run: reads from cloud environment variables
No code changes needed between environments.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    Each field corresponds to an environment variable with the same name
    (case-insensitive). E.g., `app_name` reads from env var `APP_NAME`.
    """

    # ── App Identity ──────────────────────────────────────────────────────────
    app_name: str = "FastAPI RAG API"
    app_version: str = "0.3.0"
    debug: bool = False

    # ── API Keys (will be used on Day 4+) ─────────────────────────────────────
    openai_api_key: str = ""

    # ── Database (placeholder for Day 5) ─────────────────────────────────────
    database_url: str = "sqlite:///./fastapi_rag.db"

    # ── CORS & Server ─────────────────────────────────────────────────────────
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    # ── Pagination Defaults ───────────────────────────────────────────────────
    default_page_size: int = 10
    max_page_size: int = 100

    # ── Logging ───────────────────────────────────────────────────────────────
    log_file: str = "logs/app.log"

    # ── Pydantic Settings Configuration ──────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",           # Load from .env file
        env_file_encoding="utf-8",
        case_sensitive=False,      # APP_NAME and app_name both work
        extra="ignore",            # Ignore unknown env vars (don't crash)
    )


# ── Singleton Pattern with lru_cache ─────────────────────────────────────────
# lru_cache ensures Settings() is only created ONCE for the entire app lifetime.
# Without this, every call to get_settings() would re-read the .env file —
# wasteful and potentially inconsistent.
#
# This is the recommended FastAPI pattern for settings.
# See: https://fastapi.tiangolo.com/advanced/settings/

@lru_cache
def get_settings() -> Settings:
    """
    Returns the cached application settings singleton.

    Use as a FastAPI dependency:
        settings: Settings = Depends(get_settings)

    Or import directly in non-route code:
        from config import get_settings
        settings = get_settings()
    """
    return Settings()