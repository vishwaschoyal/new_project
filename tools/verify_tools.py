"""The ``run_check`` tool: run tests or a build inside the sandbox.

The result the model sees is deliberately shaped for *acting on a failure*: the
status line, then the tail of the output where the assertion and traceback are.
A failing check is not an error to report upward — it is the diagnostic the loop
uses to fix its own change and try again.
"""

from __future__ import annotations

from pathlib import Path

from core.errors import SafetyError
from services import sandbox_service
from tools.base import ToolResult


def run_check(workspace: Path, *, command: str, timeout: int | None = None) -> ToolResult:
    try:
        result = sandbox_service.run_check(workspace, command, timeout=timeout)
    except SafetyError as exc:
        return ToolResult.failure("run_check", str(exc), refused=True)

    payload = result.to_model_string()
    metadata = {
        "command": result.command,
        "exit_code": result.exit_code,
        "backend": result.backend,
        "isolated": result.isolated,
        "duration_seconds": result.duration_seconds,
        "timed_out": result.timed_out,
        "truncated": result.truncated,
    }

    # A failing check returns ok=False so the trail shows it red, but the body
    # is the full diagnostic rather than a one-line error.
    if result.ok:
        return ToolResult.success("run_check", payload, **metadata)
    return ToolResult.failure("run_check", payload, **metadata)
