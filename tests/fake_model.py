"""A scripted stand-in for ChatOpenAI.

The orchestrator's contract with the provider is small: ``bind_tools`` then
``stream``, yielding chunks that aggregate into a message with optional
``tool_calls`` and ``usage_metadata``. Implementing exactly that lets the whole
loop be tested — including budget exhaustion, provider failure, and
citation verification — without a network call or a cent of spend.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable

from langchain_core.messages import AIMessageChunk


def tool_call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "args": arguments, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def turn(
    *,
    text: str = "",
    calls: list[dict[str, Any]] | None = None,
    input_tokens: int = 1_000,
    cached_tokens: int = 0,
    output_tokens: int = 120,
) -> dict[str, Any]:
    """One scripted model response."""
    return {
        "text": text,
        "calls": calls or [],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cached_tokens},
            "output_token_details": {"reasoning": 0},
        },
    }


class FakeModel:
    """Replays a script. Raises if the loop asks for more turns than scripted."""

    def __init__(self, script: Iterable[dict[str, Any]], *, fail_with: Exception | None = None):
        self.script = list(script)
        self.calls_received: list[list] = []
        self.bound_tools: list[dict] = []
        self.index = 0
        self.fail_with = fail_with

    def bind_tools(self, tools, **_kwargs):
        self.bound_tools = list(tools)
        return self

    def stream(self, messages, **_kwargs):
        if self.fail_with is not None:
            raise self.fail_with

        self.calls_received.append(list(messages))
        if self.index >= len(self.script):
            # Ending the script means "answer now"; a loop that keeps calling
            # tools past its script is the bug this surfaces.
            step = turn(text="I have run out of scripted turns.")
        else:
            step = self.script[self.index]
            self.index += 1

        yield AIMessageChunk(
            content=step["text"],
            tool_calls=step["calls"],
            usage_metadata=step["usage"],
        )

    def invoke(self, messages, **kwargs):
        aggregate = None
        for chunk in self.stream(messages, **kwargs):
            aggregate = chunk if aggregate is None else aggregate + chunk
        return aggregate


def install(monkeypatch, model: FakeModel) -> FakeModel:
    """Route every model construction in the loop to this fake."""
    import agents.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "create_read_loop_model", lambda **_kw: model)
    return model
