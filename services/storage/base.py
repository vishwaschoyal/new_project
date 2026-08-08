"""The conversation storage contract.

Phase 1 ships a bounded process-local implementation; Phase 8 swaps in a durable
SQL implementation. Because both satisfy this interface, the swap is a
configuration change rather than a rewrite of the routes or the agent loop.

Only bounded user/assistant turns are persisted here. System prompts, raw tool
output, and per-request evidence are deliberately *not* stored: they are
request-local, they are large, and replaying them would corrupt the cacheable
prompt prefix on a follow-up question.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Role = Literal["user", "assistant"]


@dataclass
class Message:
    thread_id: str
    role: Role
    content: str
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Citations, token usage, cost, and termination reason for assistant turns.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UsageRecord:
    """One model request's accounting, for billing and operational reporting."""

    thread_id: str
    user_id: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cost_usd: float
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConversationStore(ABC):
    """Storage for bounded conversation history and usage accounting."""

    @abstractmethod
    def append(self, message: Message) -> Message:
        """Persist one turn and return it."""

    @abstractmethod
    def history(self, thread_id: str, *, limit: int | None = None) -> list[Message]:
        """Return turns oldest-first, capped at ``limit``."""

    @abstractmethod
    def threads(self, *, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return thread summaries, most recently active first."""

    @abstractmethod
    def delete_thread(self, thread_id: str) -> bool:
        """Remove a thread and its turns. Returns whether anything was removed."""

    @abstractmethod
    def record_usage(self, record: UsageRecord) -> UsageRecord:
        """Persist one usage record."""

    @abstractmethod
    def usage_since(self, *, user_id: str, since: float) -> list[UsageRecord]:
        """Usage records for a user after ``since``, for quota enforcement."""

    def cost_since(self, *, user_id: str, since: float) -> float:
        return sum(r.cost_usd for r in self.usage_since(user_id=user_id, since=since))
