"""Git CLI wrapper.

All Git access goes through ``_run``, which uses a fixed argument list (never a
shell string), a timeout, and an environment that cannot prompt for credentials.
A hung credential prompt inside a web request is a production outage, so
``GIT_TERMINAL_PROMPT=0`` is not optional.

Read operations are unrestricted. Write operations (commit, push, branch
creation) exist here but are only ever reached through the approval gate in
``services/publish_service.py`` — this module enforces no policy of its own
beyond refusing to operate on a dirty tree where that would lose work.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from config import LIMITS
from core.errors import ValidationError, WorkspaceError

# A conservative branch-name policy. Git itself allows more, but everything
# outside this set has caused a quoting or path bug somewhere.
_BRANCH_RE = re.compile(r"^(?!-)[A-Za-z0-9._/\-]{1,100}$")


@dataclass
class GitResult:
    ok: bool
    stdout: str
    stderr: str
    returncode: int

    @property
    def output(self) -> str:
        return self.stdout if self.ok else (self.stderr or self.stdout)


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",   # never block on a credential prompt
            "GCM_INTERACTIVE": "never",
            "GIT_ASKPASS": "",
            "SSH_ASKPASS": "",
            "GIT_CONFIG_NOSYSTEM": "1",
            "LC_ALL": "C",                # stable, parseable output
        }
    )
    return env


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 60,
) -> GitResult:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=_git_env(),
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return GitResult(False, "", f"git {args[0]} timed out after {timeout}s", -1)
    except FileNotFoundError:
        return GitResult(False, "", "git executable not found on PATH", -1)

    return GitResult(
        proc.returncode == 0,
        (proc.stdout or "").strip(),
        (proc.stderr or "").strip(),
        proc.returncode,
    )


def validate_branch_name(name: str) -> str:
    branch = (name or "").strip()
    if not _BRANCH_RE.match(branch):
        raise ValidationError(f"Invalid branch name: {name!r}")
    if ".." in branch or branch.endswith((".lock", "/")):
        raise ValidationError(f"Invalid branch name: {name!r}")
    return branch


# --------------------------------------------------------------------------
# Read operations
# --------------------------------------------------------------------------
def clone(url: str, destination: Path, *, token: str | None = None, depth: int = 1) -> GitResult:
    """Shallow-clone a repository.

    A token is injected into the URL only for the duration of this call and is
    scrubbed from any error text before it can reach a log or the browser.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    clone_url = url
    if token and url.startswith("https://github.com/"):
        clone_url = url.replace("https://", f"https://x-access-token:{token}@", 1)

    result = _run(
        ["clone", "--depth", str(depth), "--no-single-branch", clone_url, str(destination)],
        timeout=LIMITS.clone_timeout_seconds,
    )
    if token:
        result = GitResult(
            result.ok,
            result.stdout.replace(token, "***"),
            result.stderr.replace(token, "***"),
            result.returncode,
        )
    return result


def current_branch(repo: Path) -> str:
    result = _run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    return result.stdout if result.ok else ""


def head_sha(repo: Path) -> str:
    result = _run(["rev-parse", "HEAD"], cwd=repo)
    return result.stdout if result.ok else ""


def list_branches(repo: Path) -> dict[str, list[str] | str]:
    local = _run(["branch", "--format=%(refname:short)"], cwd=repo)
    remote = _run(
        ["branch", "-r", "--format=%(refname:short)"], cwd=repo
    )
    remote_names = []
    for line in remote.stdout.splitlines():
        line = line.strip()
        if not line or "->" in line:
            continue
        remote_names.append(line.split("/", 1)[-1] if "/" in line else line)

    local_names = [b.strip() for b in local.stdout.splitlines() if b.strip()]
    combined = sorted({*local_names, *remote_names})
    return {
        "current": current_branch(repo),
        "local": local_names,
        "all": combined,
    }


def is_clean(repo: Path) -> bool:
    result = _run(["status", "--porcelain"], cwd=repo)
    return result.ok and not result.stdout.strip()


def status_short(repo: Path) -> str:
    return _run(["status", "--porcelain"], cwd=repo).stdout


def diff(repo: Path, *, staged: bool = False, base: str | None = None) -> str:
    args = ["diff", "--no-color"]
    if staged:
        args.append("--cached")
    if base:
        args.append(f"{base}...HEAD")
    return _run(args, cwd=repo, timeout=30).stdout


def diff_stat(repo: Path, *, base: str | None = None) -> str:
    args = ["diff", "--no-color", "--stat"]
    if base:
        args.append(f"{base}...HEAD")
    return _run(args, cwd=repo, timeout=30).stdout


def default_branch(repo: Path) -> str:
    """Best-effort default branch: origin/HEAD, else main, else master."""
    result = _run(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=repo)
    if result.ok and "/" in result.stdout:
        return result.stdout.rsplit("/", 1)[-1]
    branches = list_branches(repo)
    for candidate in ("main", "master"):
        if candidate in branches["all"]:
            return candidate
    return current_branch(repo) or "main"


# --------------------------------------------------------------------------
# Write operations — reached only through the approval gate
# --------------------------------------------------------------------------
def checkout_branch(repo: Path, branch: str, *, create: bool = False) -> GitResult:
    branch = validate_branch_name(branch)
    if not is_clean(repo):
        raise WorkspaceError(
            "The workspace has uncommitted changes. Commit or discard them before switching branches."
        )
    args = ["checkout", "-b", branch] if create else ["checkout", branch]
    return _run(args, cwd=repo)


def create_task_branch(repo: Path, branch: str) -> GitResult:
    """Create and switch to a task branch, tolerating one that already exists."""
    branch = validate_branch_name(branch)
    existing = list_branches(repo)
    if branch in existing["local"]:
        return _run(["checkout", branch], cwd=repo)
    return _run(["checkout", "-b", branch], cwd=repo)


def stage_all(repo: Path) -> GitResult:
    return _run(["add", "-A"], cwd=repo)


def commit(repo: Path, message: str, *, author_name: str, author_email: str) -> GitResult:
    if not message.strip():
        raise ValidationError("A commit message is required.")
    return _run(
        [
            "-c", f"user.name={author_name}",
            "-c", f"user.email={author_email}",
            "commit", "-m", message,
        ],
        cwd=repo,
    )


def push(repo: Path, branch: str, *, token: str | None = None, remote: str = "origin") -> GitResult:
    """Push a branch, injecting the token into the remote URL for this call only."""
    branch = validate_branch_name(branch)

    push_args = ["push", "--set-upstream", remote, branch]
    if token:
        url_result = _run(["remote", "get-url", remote], cwd=repo)
        url = url_result.stdout
        if url.startswith("https://github.com/"):
            authed = url.replace("https://", f"https://x-access-token:{token}@", 1)
            push_args = ["push", "--set-upstream", authed, branch]

    result = _run(push_args, cwd=repo, timeout=120)
    if token:
        result = GitResult(
            result.ok,
            result.stdout.replace(token, "***"),
            result.stderr.replace(token, "***"),
            result.returncode,
        )
    return result


def remote_url(repo: Path, remote: str = "origin") -> str:
    return _run(["remote", "get-url", remote], cwd=repo).stdout
