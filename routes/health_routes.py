"""Health and readiness endpoints for deployment probes."""

from __future__ import annotations

import time

from flask import Blueprint, jsonify

import app_state
from config import READ_LOOP_MODEL, SETTINGS
from services import git_service
from services.sandbox_service import sandbox_available

health_bp = Blueprint("health", __name__)

_STARTED_AT = time.time()


@health_bp.get("/healthz")
def liveness():
    """Liveness: the process is up. Must stay cheap — no I/O.

    A liveness probe that touches a dependency will restart a healthy process
    when that dependency blips, so this deliberately checks nothing external.
    """
    return jsonify({"status": "ok", "uptime_seconds": round(time.time() - _STARTED_AT, 1)})


@health_bp.get("/readyz")
def readiness():
    """Readiness: every dependency needed to serve traffic is usable."""
    checks: dict[str, object] = {}

    checks["model_configured"] = SETTINGS.model_configured
    checks["model"] = READ_LOOP_MODEL

    git_ok = git_service._run(["--version"]).ok
    checks["git"] = git_ok

    try:
        app_state.conversations.threads(limit=1)
        checks["storage"] = True
    except Exception:
        checks["storage"] = False

    checks["workspace_root_writable"] = app_state.workspaces.root.exists()
    # Informational: the sandbox degrades to a bounded subprocess runner, so an
    # unreachable Docker daemon is not a readiness failure.
    checks["sandbox"] = sandbox_available()

    required = ("model_configured", "git", "storage", "workspace_root_writable")
    ready = all(bool(checks[name]) for name in required)
    return jsonify({"ready": ready, "checks": checks}), (200 if ready else 503)
