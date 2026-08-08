"""Storage factory — selects an implementation behind one shared interface."""

from __future__ import annotations

from config import SETTINGS
from services.storage.base import ConversationStore, Message, UsageRecord
from services.storage.memory_store import InMemoryConversationStore
from services.storage.sql_store import SqlConversationStore

__all__ = [
    "ConversationStore",
    "Message",
    "UsageRecord",
    "InMemoryConversationStore",
    "SqlConversationStore",
    "create_store",
]


def create_store(kind: str | None = None, *, database_url: str | None = None) -> ConversationStore:
    kind = (kind or SETTINGS.conversation_store).lower()
    if kind == "memory":
        return InMemoryConversationStore()
    if kind in {"sqlite", "sql", "postgres"}:
        return SqlConversationStore(database_url or SETTINGS.database_url)
    raise ValueError(f"unknown CONVERSATION_STORE: {kind!r}")
