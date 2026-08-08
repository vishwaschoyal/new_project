"""Bounded process-local conversation storage.

Suitable for development and tests. It is bounded on three axes — turns per
thread, characters per thread, and total threads — so a long-running process
cannot grow without limit. It is *not* durable: Phase 8 deployments use
``SqlConversationStore`` instead.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from config import LIMITS
from services.storage.base import ConversationStore, Message, UsageRecord

MAX_THREADS = 200
MAX_USAGE_RECORDS = 5_000


class InMemoryConversationStore(ConversationStore):
    def __init__(
        self,
        *,
        max_messages: int = LIMITS.max_history_messages,
        max_chars: int = LIMITS.max_history_chars,
        max_threads: int = MAX_THREADS,
    ):
        self._max_messages = max_messages
        self._max_chars = max_chars
        self._max_threads = max_threads
        # OrderedDict gives LRU eviction of whole threads for free.
        self._threads: OrderedDict[str, list[Message]] = OrderedDict()
        self._usage: list[UsageRecord] = []
        self._lock = threading.Lock()

    # -- history ---------------------------------------------------------
    def append(self, message: Message) -> Message:
        with self._lock:
            turns = self._threads.get(message.thread_id)
            if turns is None:
                turns = []
                self._threads[message.thread_id] = turns
            turns.append(message)
            self._trim_thread(turns)
            self._threads.move_to_end(message.thread_id)
            while len(self._threads) > self._max_threads:
                self._threads.popitem(last=False)
            return message

    def _trim_thread(self, turns: list[Message]) -> None:
        """Drop oldest turns until both the count and character caps hold."""
        while len(turns) > self._max_messages:
            turns.pop(0)
        total = sum(len(t.content) for t in turns)
        while total > self._max_chars and len(turns) > 1:
            total -= len(turns.pop(0).content)

    def history(self, thread_id: str, *, limit: int | None = None) -> list[Message]:
        with self._lock:
            turns = list(self._threads.get(thread_id, ()))
        return turns[-limit:] if limit else turns

    def threads(self, *, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._threads.items())
        summaries = []
        for thread_id, turns in items:
            if not turns:
                continue
            first_user = next((t for t in turns if t.role == "user"), turns[0])
            summaries.append(
                {
                    "thread_id": thread_id,
                    "title": first_user.content[:80],
                    "message_count": len(turns),
                    "updated_at": turns[-1].created_at,
                }
            )
        summaries.sort(key=lambda s: s["updated_at"], reverse=True)
        return summaries[:limit]

    def delete_thread(self, thread_id: str) -> bool:
        with self._lock:
            return self._threads.pop(thread_id, None) is not None

    # -- usage -----------------------------------------------------------
    def record_usage(self, record: UsageRecord) -> UsageRecord:
        with self._lock:
            self._usage.append(record)
            if len(self._usage) > MAX_USAGE_RECORDS:
                del self._usage[: len(self._usage) - MAX_USAGE_RECORDS]
            return record

    def usage_since(self, *, user_id: str, since: float) -> list[UsageRecord]:
        with self._lock:
            return [
                r for r in self._usage if r.user_id == user_id and r.created_at >= since
            ]

    # -- test helper -----------------------------------------------------
    def clear(self) -> None:
        with self._lock:
            self._threads.clear()
            self._usage.clear()


def utcnow() -> float:
    return time.time()
