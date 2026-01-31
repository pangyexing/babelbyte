"""Common utilities for CLI commands."""

from rich.console import Console

from src.storage.database import SyncDatabase

console = Console()


def get_db() -> SyncDatabase:
    """Get database connection."""
    db = SyncDatabase()
    db.connect()
    return db
