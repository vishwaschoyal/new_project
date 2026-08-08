"""Structured JSON logging with per-request correlation IDs.

Logs are emitted as one JSON object per line so a log shipper can index them
without regex parsing. Every record inside a request carries the same
``request_id``, which is also returned to the client in the ``X-Request-ID``
header — that is what makes a user-reported failure traceable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from typing import Any

from flask import Flask, g, request

# Values that must never reach a log line, matched case-insensitively against
# the key name of any structured field.
_SENSITIVE_KEYS = {
    "api_key",
    "openai_api_key",
    "authorization",
    "token",
    "github_token",
    "password",
    "secret",
    "secret_key",
    "cookie",
}

_STANDARD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


def _redact(key: str, value: Any) -> Any:
    if key.lower() in _SENSITIVE_KEYS:
        return "***redacted***"
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None) or _current_request_id()
        if request_id:
            payload["request_id"] = request_id

        # Anything passed via logger.info("...", extra={...}) lands here.
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            if key == "request_id":
                continue
            payload[key] = _redact(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _current_request_id() -> str | None:
    try:
        return g.get("request_id")  # type: ignore[attr-defined]
    except Exception:
        # Outside an application context (background threads, CLI scripts).
        return None


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # These are noisy at INFO and say nothing we do not already log ourselves.
    for noisy in ("werkzeug", "urllib3", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def install_request_logging(app: Flask) -> None:
    """Attach a request ID and one access log line per request."""

    log = logging.getLogger("access")

    @app.before_request
    def _start_request() -> None:
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        g.request_started = time.perf_counter()

    @app.after_request
    def _finish_request(response):
        started = g.get("request_started")
        duration_ms = round((time.perf_counter() - started) * 1000, 2) if started else None
        request_id = g.get("request_id")
        if request_id:
            response.headers["X-Request-ID"] = request_id

        # Streaming responses log at start; their real duration is the stream's.
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
