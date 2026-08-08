"""Token accounting and cost estimation.

Two distinct jobs, deliberately not conflated:

1. **Estimation** (``estimate_tokens``) — a local ``tiktoken`` count used
   *before* a request to decide whether the next call fits in the budget.
   Approximate by nature; it must never be reported to the user as actual cost.
2. **Accounting** (``usage_from_response``) — the provider's own usage metadata,
   read *after* a request. This is what the user is shown and billed against,
   and it is the only source that knows how many input tokens were cache hits.

Cached input tokens are billed at a large discount, so a run that reports a high
input count but a high cache-hit rate can be much cheaper than it looks. Keeping
them as separate fields is what makes the caching work visible instead of
hidden inside one blended number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any

import tiktoken

log = logging.getLogger(__name__)

# USD per 1M tokens. Update alongside provider pricing changes.
# `cached_input` is the discounted rate for prompt-cache hits.
PRICING: dict[str, dict[str, float]] = {
    "gpt-5.4":       {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.4-mini":  {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5":         {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini":    {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-4.1":       {"input": 2.00, "cached_input": 0.50,  "output": 8.00},
    "gpt-4.1-mini":  {"input": 0.40, "cached_input": 0.10,  "output": 1.60},
    "gpt-4o":        {"input": 2.50, "cached_input": 1.25,  "output": 10.00},
    "gpt-4o-mini":   {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}

_FALLBACK_PRICING = {"input": 1.00, "cached_input": 0.10, "output": 4.00}
_ENCODING_CACHE: dict[str, Any] = {}


def pricing_for(model: str) -> dict[str, float]:
    """Exact match, then longest known prefix, then a conservative fallback."""
    if model in PRICING:
        return PRICING[model]
    candidates = [name for name in PRICING if model.startswith(name)]
    if candidates:
        return PRICING[max(candidates, key=len)]
    log.warning("no pricing entry for model", extra={"model": model})
    return _FALLBACK_PRICING


def _encoding(model: str):
    if model not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Newer models are not in tiktoken's table yet; o200k_base is the
            # right base encoding for the current generation.
            _ENCODING_CACHE[model] = tiktoken.get_encoding("o200k_base")
    return _ENCODING_CACHE[model]


def estimate_tokens(text: str, model: str = "gpt-5.4-mini") -> int:
    if not text:
        return 0
    try:
        return len(_encoding(model).encode(text, disallowed_special=()))
    except Exception:
        # Never let token estimation break a request; 4 chars/token is close.
        return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[Any], model: str = "gpt-5.4-mini") -> int:
    """Estimate a message list, including tool calls and a per-message overhead."""
    total = 0
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, str):
            total += estimate_tokens(content, model)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total += estimate_tokens(str(block.get("text", "")), model)
                else:
                    total += estimate_tokens(str(block), model)

        for call in getattr(message, "tool_calls", None) or []:
            total += estimate_tokens(str(call), model)

        total += 4  # role + delimiter overhead
    return total


@dataclass
class Usage:
    """One request's accounting. Additive across a multi-step run."""

    model: str = ""
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0
    cost_usd: float = 0.0
    # Populated by the loop, not the provider.
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        return max(0, self.input_tokens - self.cached_input_tokens)

    @property
    def cache_hit_rate(self) -> float:
        return (self.cached_input_tokens / self.input_tokens) if self.input_tokens else 0.0

    def add(self, other: "Usage") -> "Usage":
        self.model = self.model or other.model
        self.input_tokens += other.input_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.output_tokens += other.output_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.requests += other.requests
        self.cost_usd = round(self.cost_usd + other.cost_usd, 6)
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            total_tokens=self.total_tokens,
            uncached_input_tokens=self.uncached_input_tokens,
            cache_hit_rate=round(self.cache_hit_rate, 4),
        )
        return data


def compute_cost(
    model: str,
    *,
    input_tokens: int,
    cached_input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Cost in USD. Reasoning tokens are billed as output and already counted there."""
    rates = pricing_for(model)
    uncached = max(0, input_tokens - cached_input_tokens)
    cost = (
        uncached * rates["input"]
        + cached_input_tokens * rates["cached_input"]
        + output_tokens * rates["output"]
    ) / 1_000_000
    return round(cost, 6)


def _dig(source: Any, *names: str) -> int:
    """Read the first present key across the shapes providers actually return."""
    if not isinstance(source, dict):
        return 0
    for name in names:
        value = source.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def usage_from_response(response: Any, model: str) -> Usage:
    """Extract provider usage metadata from a ChatOpenAI response.

    LangChain surfaces this in several places depending on version and whether
    the call streamed, so all of them are checked rather than assuming one.
    """
    raw = (
        getattr(response, "usage_metadata", None)
        or (getattr(response, "response_metadata", None) or {}).get("usage")
        or (getattr(response, "response_metadata", None) or {}).get("token_usage")
        or {}
    )

    input_tokens = _dig(raw, "input_tokens", "prompt_tokens")
    output_tokens = _dig(raw, "output_tokens", "completion_tokens")

    details = raw.get("input_token_details") or raw.get("prompt_tokens_details") or {}
    cached = _dig(details, "cache_read", "cached_tokens", "cached")

    output_details = raw.get("output_token_details") or raw.get("completion_tokens_details") or {}
    reasoning = _dig(output_details, "reasoning", "reasoning_tokens")

    usage = Usage(
        model=model,
        input_tokens=input_tokens,
        cached_input_tokens=min(cached, input_tokens),
        output_tokens=output_tokens,
        reasoning_tokens=reasoning,
        requests=1,
    )
    usage.cost_usd = compute_cost(
        model,
        input_tokens=usage.input_tokens,
        cached_input_tokens=usage.cached_input_tokens,
        output_tokens=usage.output_tokens,
    )
    return usage
