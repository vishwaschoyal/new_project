"""The shared tool result contract.

Every tool returns a ``ToolResult`` — success or failure, never a raised
exception crossing into the agent loop. A tool failure is information the model
should act on (search again, widen the range, pick a different file), not a
request-ending error. Errors are therefore returned as content the model reads.

Truncation is always reported. A silently truncated result is the mechanism by
which an agent confidently cites something that was never there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.safety import redact_secrets


@dataclass
class ToolResult:
    ok: bool
    content: str
    tool: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, tool: str, content: str, **metadata: Any) -> "ToolResult":
        return cls(True, redact_secrets(content), tool, metadata)

    @classmethod
    def failure(cls, tool: str, message: str, **metadata: Any) -> "ToolResult":
        return cls(False, redact_secrets(message), tool, metadata)

    def to_model_string(self) -> str:
        """Exactly what the model sees as the tool observation."""
        if not self.ok:
            return f"ERROR: {self.content}"
        if self.metadata.get("truncated"):
            shown = self.metadata.get("shown")
            limit = self.metadata.get("limit")
            note = (
                f"\n\n[truncated: showing {shown} of at least {limit}+ results;"
                " narrow the pattern or path to see more]"
                if shown is not None
                else "\n\n[truncated: output exceeded the size limit]"
            )
            return self.content + note
        return self.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "content": self.content,
            "metadata": self.metadata,
        }


class Timer:
    """Wall-clock timing for the tool trail the user inspects."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        self.ms = round((time.perf_counter() - self._start) * 1000, 1)

    ms: float = 0.0


def clamp_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Cut text to a character budget, reporting whether anything was lost."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip(), True
