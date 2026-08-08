"""Isolated execution of repository tests and builds.

Running a cloned repository's test suite means running **code we did not write
and have not read**. A `conftest.py` executes on collection; a `package.json`
lifecycle script runs on install. So execution never happens in the application
process, and never with the application's credentials.

Two backends:

- **docker** — the real boundary. A disposable container per run, with no
  network, a read-only-ish bind of the workspace, a memory cap, a CPU cap, a
  process cap, dropped capabilities, and `--rm` so nothing survives.
- **subprocess** — the fallback for when no daemon is reachable (the common
  local-development case). It bounds *time and output* and strips credentials
  from the environment, but it does **not** isolate the filesystem or the
  network. It is honest about this: every result reports which backend ran and
  whether it was isolated, and the UI shows that to the user.

``SANDBOX_MODE`` selects: ``docker`` (required), ``subprocess`` (forced), or
``auto`` (docker when reachable, else subprocess).
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import LIMITS, SETTINGS
from core.errors import SafetyError
from core.safety import redact_secrets
from tools.base import clamp_text

log = logging.getLogger(__name__)

_DOCKER_PROBE_CACHE: dict[str, Any] = {"checked_at": 0.0, "available": False}
_PROBE_TTL_SECONDS = 30.0

# Allow-list of check runners. A repository must not be able to name an
# arbitrary executable just because its command string reached us.
_ALLOWED_RUNNERS = frozenset(
    {"pytest", "python", "python3", "npm", "npx", "yarn", "pnpm", "node",
     "go", "cargo", "make", "mvn", "gradle", "ruff", "mypy", "eslint", "tsc"}
)

_SHELL_METACHARACTERS = set(";&|><`$\n\r")


@dataclass
class CheckResult:
    ok: bool
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    backend: str
    isolated: bool
    timed_out: bool = False
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "command": self.command,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "backend": self.backend,
            "isolated": self.isolated,
            "timed_out": self.timed_out,
            "truncated": self.truncated,
            **self.metadata,
        }

    def to_model_string(self) -> str:
        """Bounded diagnostics for the loop to read and act on.

        Failure output is trimmed from the *front*: the assertion and traceback
        that explain a failure are at the end of pytest output, and keeping the
        head while dropping the tail would hand the model the least useful half.
        """
        status = "PASSED" if self.ok else ("TIMED OUT" if self.timed_out else "FAILED")
        header = f"{status}: {self.command} (exit {self.exit_code}, {self.duration_seconds:.1f}s, {self.backend})"
        if not self.isolated:
            header += "\n[warning: ran without container isolation]"

        body = (self.stdout or "") + (f"\n[stderr]\n{self.stderr}" if self.stderr else "")
        if len(body) > LIMITS.sandbox_max_output_chars:
            body = "[earlier output trimmed]\n" + body[-LIMITS.sandbox_max_output_chars:]
        return f"{header}\n\n{body.strip() or '(no output)'}"


# --------------------------------------------------------------------------
# Command validation
# --------------------------------------------------------------------------
def validate_check_command(command: str) -> list[str]:
    """Parse and allow-list a check command, or raise SafetyError."""
    raw = (command or "").strip()
    if not raw:
        raise SafetyError("A check command is required.")
    if any(character in _SHELL_METACHARACTERS for character in raw):
        raise SafetyError(
            "Shell operators are not permitted in a check command. Run one command."
        )

    try:
        parts = shlex.split(raw)
    except ValueError as exc:
        raise SafetyError(f"Could not parse command: {exc}") from exc
    if not parts:
        raise SafetyError("A check command is required.")

    runner = Path(parts[0]).name
    if runner not in _ALLOWED_RUNNERS:
        raise SafetyError(
            f"{runner!r} is not an approved check runner. "
            f"Approved: {', '.join(sorted(_ALLOWED_RUNNERS))}."
        )

    for part in parts[1:]:
        if part.startswith("/") or ".." in Path(part).parts:
            raise SafetyError(f"Check arguments must stay inside the workspace: {part!r}")

    return parts


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------
def sandbox_available() -> bool:
    """Whether a Docker daemon is reachable. Cached — probing is slow."""
    now = time.time()
    if now - _DOCKER_PROBE_CACHE["checked_at"] < _PROBE_TTL_SECONDS:
        return bool(_DOCKER_PROBE_CACHE["available"])

    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, text=True, timeout=8, shell=False,
        )
        available = proc.returncode == 0 and bool(proc.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        available = False

    _DOCKER_PROBE_CACHE.update(checked_at=now, available=available)
    return available


def active_backend() -> str:
    mode = SETTINGS.sandbox_mode
    if mode == "subprocess":
        return "subprocess"
    if mode == "docker":
        return "docker"
    return "docker" if sandbox_available() else "subprocess"


def _clean_environment() -> dict[str, str]:
    """A minimal environment with every credential removed.

    Repository code has no business seeing our OpenAI key or GitHub token, and
    the fallback backend shares this process's environment by default.
    """
    keep = {"PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE"}
    env = {name: value for name, value in os.environ.items() if name in keep}
    env.update(
        CI="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        NO_COLOR="1",
        npm_config_yes="true",
    )
    return env


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------
def run_check(
    workspace: Path,
    command: str,
    *,
    timeout: int | None = None,
    backend: str | None = None,
) -> CheckResult:
    """Run one verification command against the workspace."""
    parts = validate_check_command(command)
    timeout = timeout or LIMITS.sandbox_timeout_seconds
    chosen = backend or active_backend()

    if chosen == "docker":
        if not sandbox_available():
            if SETTINGS.sandbox_mode == "docker":
                return CheckResult(
                    ok=False, command=command, exit_code=-1, stdout="",
                    stderr=(
                        "SANDBOX_MODE=docker but no Docker daemon is reachable. "
                        "Start Docker Desktop, or set SANDBOX_MODE=auto to allow the "
                        "bounded subprocess fallback."
                    ),
                    duration_seconds=0.0, backend="docker", isolated=True,
                )
            chosen = "subprocess"

    if chosen == "docker":
        return _run_in_docker(workspace, parts, command, timeout)
    return _run_in_subprocess(workspace, parts, command, timeout)


def _run_in_docker(workspace: Path, parts: list[str], command: str, timeout: int) -> CheckResult:
    container_args = [
        "docker", "run", "--rm",
        "--network", "none",              # no exfiltration, no dependency fetch
        "--memory", f"{LIMITS.sandbox_memory_mb}m",
        "--memory-swap", f"{LIMITS.sandbox_memory_mb}m",
        "--cpus", str(LIMITS.sandbox_cpus),
        "--pids-limit", "256",            # fork bombs
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "1000:1000",
        "--workdir", "/repo",
        "--volume", f"{workspace.resolve()}:/repo",
        "--env", "CI=1",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        SETTINGS.sandbox_image,
        *parts,
    ]

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            container_args, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout + 15, shell=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            ok=False, command=command, exit_code=-1, stdout="",
            stderr=f"Check timed out after {timeout}s.",
            duration_seconds=round(time.perf_counter() - started, 2),
            backend="docker", isolated=True, timed_out=True,
        )
    except OSError as exc:
        return CheckResult(
            ok=False, command=command, exit_code=-1, stdout="",
            stderr=f"Could not start container: {exc}",
            duration_seconds=0.0, backend="docker", isolated=True,
        )

    return _build_result(proc, command, started, backend="docker", isolated=True)


def _run_in_subprocess(workspace: Path, parts: list[str], command: str, timeout: int) -> CheckResult:
    """Bounded local execution. Bounds time, output, and environment — not the
    filesystem. Every result says so."""
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            parts, cwd=str(workspace), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
            env=_clean_environment(), shell=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            ok=False, command=command, exit_code=-1, stdout="",
            stderr=f"Check timed out after {timeout}s.",
            duration_seconds=round(time.perf_counter() - started, 2),
            backend="subprocess", isolated=False, timed_out=True,
        )
    except (OSError, FileNotFoundError) as exc:
        return CheckResult(
            ok=False, command=command, exit_code=-1, stdout="",
            stderr=f"Could not run {parts[0]!r}: {exc}",
            duration_seconds=0.0, backend="subprocess", isolated=False,
        )

    return _build_result(proc, command, started, backend="subprocess", isolated=False)


def _build_result(
    proc: subprocess.CompletedProcess,
    command: str,
    started: float,
    *,
    backend: str,
    isolated: bool,
) -> CheckResult:
    stdout, out_truncated = clamp_text(proc.stdout or "", LIMITS.sandbox_max_output_chars)
    stderr, err_truncated = clamp_text(proc.stderr or "", 8_000)
    return CheckResult(
        ok=proc.returncode == 0,
        command=command,
        exit_code=proc.returncode,
        stdout=redact_secrets(stdout),
        stderr=redact_secrets(stderr),
        duration_seconds=round(time.perf_counter() - started, 2),
        backend=backend,
        isolated=isolated,
        truncated=out_truncated or err_truncated,
    )
