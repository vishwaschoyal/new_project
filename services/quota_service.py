"""Authentication, rate limiting, and cost quotas.

Identity is resolved once, here, and every other module asks this service rather
than reading headers itself — so there is exactly one place where "who is this"
can be got wrong.

With ``AUTH_ENABLED=false`` the whole application runs as a single local user.
That is the correct default for local development and an unacceptable one for a
public deployment, which is why ``/readyz`` reports it and the deployment guide
calls it out.

Two independent limits, because they fail differently: a **rate limit** stops a
runaway client hammering the API, and a **daily cost ceiling** stops a slow,
polite client quietly spending a fortune.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque

from flask import g, request

import app_state
from config import SETTINGS
from core.errors import AuthError, QuotaExceededError

log = logging.getLogger(__name__)

LOCAL_USER_ID = "local"
_SECONDS_PER_DAY = 86_400

# user_id -> timestamps of recent requests
_request_times: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


# --------------------------------------------------------------------------
# Identity
# --------------------------------------------------------------------------
def _api_key_from_request() -> str:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("X-API-Key", "").strip()


def current_user_id() -> str:
    """Resolve the calling user, or raise AuthError.

    The user ID is a hash of the presented key, never the key itself: it ends up
    in logs, usage records, and workspace directory names.
    """
    if not SETTINGS.auth_enabled:
        return LOCAL_USER_ID

    cached = g.get("user_id") if _in_request_context() else None
    if cached:
        return cached

    api_key = _api_key_from_request()
    if not api_key:
        raise AuthError("An API key is required. Send it as 'Authorization: Bearer <key>'.")
    if len(api_key) < 20:
        raise AuthError("Invalid API key.")

    user_id = "u_" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:24]
    if _in_request_context():
        g.user_id = user_id
    return user_id


def _in_request_context() -> bool:
    try:
        return bool(request)
    except RuntimeError:
        return False


def owns_thread(user_id: str, thread_id: str) -> bool:
    """Whether ``user_id`` may access ``thread_id``.

    A thread with no stored turns is unclaimed — the first user to write to it
    owns it. An existing thread belongs to whoever wrote its first message.
    """
    if not SETTINGS.auth_enabled:
        return True
    history = app_state.conversations.history(thread_id, limit=1)
    if not history:
        return True
    return str(history[0].metadata.get("user_id", "")) in {"", user_id}


def require_thread_access(user_id: str, thread_id: str) -> None:
    if not owns_thread(user_id, thread_id):
        from core.errors import ForbiddenError

        raise ForbiddenError("This conversation belongs to another user.")


# --------------------------------------------------------------------------
# Limits
# --------------------------------------------------------------------------
def check_rate_limit(user_id: str) -> None:
    """Sliding-window rate limit.

    In-process, so it is per-worker. Multi-worker deployments need a shared
    store (Redis) — flagged in docs/deployment.md rather than pretended away.
    """
    limit = SETTINGS.rate_limit_per_minute
    if limit <= 0:
        return

    now = time.time()
    with _lock:
        window = _request_times[user_id]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= limit:
            retry_after = int(60 - (now - window[0])) + 1
            raise QuotaExceededError(
                f"Rate limit reached ({limit} requests/minute). Retry in {retry_after}s.",
                details={"retry_after_seconds": retry_after},
            )
        window.append(now)


def check_cost_quota(user_id: str) -> None:
    """Daily spend ceiling, measured from recorded provider usage."""
    ceiling = SETTINGS.daily_cost_limit_usd
    if ceiling <= 0:
        return

    spent = app_state.conversations.cost_since(
        user_id=user_id, since=time.time() - _SECONDS_PER_DAY
    )
    if spent >= ceiling:
        raise QuotaExceededError(
            f"Daily cost limit reached (${spent:.2f} of ${ceiling:.2f}). "
            "It resets 24 hours after your earliest recorded usage.",
            details={"spent_usd": round(spent, 4), "limit_usd": ceiling},
        )


def enforce(user_id: str) -> None:
    check_rate_limit(user_id)
    check_cost_quota(user_id)


def usage_summary(user_id: str) -> dict:
    since = time.time() - _SECONDS_PER_DAY
    records = app_state.conversations.usage_since(user_id=user_id, since=since)
    spent = sum(r.cost_usd for r in records)
    return {
        "user_id": user_id,
        "window_hours": 24,
        "requests": len(records),
        "input_tokens": sum(r.input_tokens for r in records),
        "cached_input_tokens": sum(r.cached_input_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
        "cost_usd": round(spent, 4),
        "limit_usd": SETTINGS.daily_cost_limit_usd,
        "remaining_usd": round(max(0.0, SETTINGS.daily_cost_limit_usd - spent), 4),
    }


def reset_for_tests() -> None:
    with _lock:
        _request_times.clear()
