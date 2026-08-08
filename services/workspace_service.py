"""Thread-owned repository workspaces.

Each thread gets a directory derived from a hash of its ``thread_id``. Nothing
resolves a workspace path from client input directly, so one thread cannot name
another thread's directory: the only way to reach a workspace is to present the
thread ID that hashes to it.

Deletion is the sharpest edge in this module. ``unload`` refuses any target that
is not a direct child of the workspace root, which is what stops a malformed
thread ID from turning into ``rmtree`` on something that matters.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import GITHUB_TOKEN, LIMITS, SETTINGS
from core.errors import NotFoundError, SafetyError, ValidationError, WorkspaceError
from core.safety import (
    BLOCKED_DIRECTORIES,
    is_binary_file,
    is_secret_path,
    redact_secrets,
    relative_to_workspace,
    resolve_in_workspace,
)
from services import git_service, github_service
from services.github_service import RepoRef

log = logging.getLogger(__name__)

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# Filesystem errors that mean "something else was holding the file", not
# "this repository cannot be cloned". Worth exactly one retry.
_TRANSIENT_CLONE_ERRORS = (
    "could not lock config file",
    "no such file or directory",
    "permission denied",
    "access is denied",
    "unable to create",
    "file exists",
    "resource temporarily unavailable",
)


def _looks_transient(error_text: str) -> bool:
    lowered = (error_text or "").lower()
    # Auth and network failures share some wording but never recover on retry.
    if any(
        fatal in lowered
        for fatal in ("authentication failed", "repository not found", "could not resolve host")
    ):
        return False
    return any(marker in lowered for marker in _TRANSIENT_CLONE_ERRORS)


def _clone_error_message(error_text: str, workspace_root: Path) -> str:
    """Turn a git failure into something the user can act on."""
    from config import synced_folder

    message = f"Clone failed: {error_text}"
    client = synced_folder(workspace_root)

    if client and _looks_transient(error_text):
        message += (
            f"\n\nThe workspace directory is inside {client}, which opens handles on "
            f"files as git creates them and makes clones fail at random. Set "
            f"WORKSPACE_ROOT in .env to a path outside {client} and restart."
        )
    return message


def validate_thread_id(thread_id: str) -> str:
    tid = (thread_id or "").strip()
    if not _THREAD_ID_RE.match(tid):
        raise ValidationError("Invalid thread id.")
    return tid


@dataclass
class Workspace:
    thread_id: str
    repo: RepoRef
    path: Path
    default_branch: str
    branch: str
    head_sha: str
    loaded_at: float = field(default_factory=time.time)
    # Set by Phase 6 once a coding task starts editing.
    task_branch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "repo": self.repo.full_name,
            "html_url": self.repo.html_url,
            "branch": self.branch,
            "default_branch": self.default_branch,
            "task_branch": self.task_branch,
            "head_sha": self.head_sha[:8],
            "loaded_at": self.loaded_at,
        }


class WorkspaceService:
    """Owns the lifecycle of every cloned repository."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or SETTINGS.workspace_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._workspaces: dict[str, Workspace] = {}
        self._lock = threading.RLock()

    # -- location --------------------------------------------------------
    def _thread_dir(self, thread_id: str) -> Path:
        """Directory for a thread: a hash, never client-controlled text."""
        thread_id = validate_thread_id(thread_id)
        digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()[:24]
        return self.root / f"ws_{digest}"

    def get(self, thread_id: str) -> Workspace | None:
        with self._lock:
            return self._workspaces.get(validate_thread_id(thread_id))

    def require(self, thread_id: str) -> Workspace:
        workspace = self.get(thread_id)
        if workspace is None:
            raise NotFoundError("No repository is loaded for this session.")
        return workspace

    # -- lifecycle -------------------------------------------------------
    def load(self, thread_id: str, repo_url: str) -> Workspace:
        """Clone a repository into this thread's workspace, replacing any existing one."""
        thread_id = validate_thread_id(thread_id)
        ref = github_service.parse_repo_url(repo_url)

        metadata = github_service.fetch_repository(ref)
        size_bytes = int(metadata.get("size_kb") or 0) * 1024
        if size_bytes > LIMITS.max_repo_bytes:
            raise WorkspaceError(
                f"Repository is too large to load "
                f"({size_bytes // (1024 * 1024)} MB; limit "
                f"{LIMITS.max_repo_bytes // (1024 * 1024)} MB)."
            )

        thread_dir = self._thread_dir(thread_id)
        self._remove_tree(thread_dir)
        destination = thread_dir / ref.name

        result = git_service.clone(ref.clone_url, destination, token=GITHUB_TOKEN or None)

        if not result.ok and _looks_transient(result.stderr or result.stdout):
            # A sync client or scanner held a handle while git was writing. The
            # partial clone is unusable, so start from a clean directory rather
            # than letting git resume into it.
            log.warning(
                "clone hit a transient filesystem error; retrying once",
                extra={"thread_id": thread_id, "repo": ref.full_name},
            )
            self._remove_tree(thread_dir)
            time.sleep(1.0)
            result = git_service.clone(ref.clone_url, destination, token=GITHUB_TOKEN or None)

        if not result.ok:
            self._remove_tree(thread_dir)
            raise WorkspaceError(_clone_error_message(result.stderr or result.stdout, self.root))

        branch = git_service.current_branch(destination)
        workspace = Workspace(
            thread_id=thread_id,
            repo=ref,
            path=destination,
            default_branch=metadata.get("default_branch") or git_service.default_branch(destination),
            branch=branch,
            head_sha=git_service.head_sha(destination),
        )
        with self._lock:
            self._workspaces[thread_id] = workspace

        log.info(
            "workspace loaded",
            extra={"thread_id": thread_id, "repo": ref.full_name, "branch": branch},
        )
        return workspace

    def unload(self, thread_id: str) -> bool:
        """Delete this thread's workspace directory and forget it."""
        thread_id = validate_thread_id(thread_id)
        thread_dir = self._thread_dir(thread_id)
        with self._lock:
            existed = self._workspaces.pop(thread_id, None) is not None
        self._remove_tree(thread_dir)
        log.info("workspace unloaded", extra={"thread_id": thread_id})
        return existed

    def _remove_tree(self, target: Path) -> None:
        """Delete a workspace directory, refusing anything that is not one.

        The guard is the point of this method: ``rmtree`` on a bad path is
        unrecoverable, so the target must be a direct child of the workspace
        root with the expected prefix before a single file is removed.
        """
        if not target.exists():
            return

        resolved = target.resolve()
        root = self.root.resolve()
        if resolved == root:
            raise SafetyError("Refusing to delete the workspace root.")
        if resolved.parent != root:
            raise SafetyError(f"Refusing to delete a path outside the workspace root: {resolved}")
        if not resolved.name.startswith("ws_"):
            raise SafetyError(f"Refusing to delete an unmanaged directory: {resolved.name}")

        # Cloned .git objects are read-only on Windows; clear the bit and retry.
        def _on_error(func, path, _exc_info):
            import os
            import stat

            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except OSError:
                log.warning("could not remove path", extra={"path": str(path)})

        # Windows keeps a directory name in "delete pending" while any process
        # still holds a handle to something inside it — a sync client, antivirus,
        # or an open editor. Recreating that exact path immediately then fails in
        # confusing ways, including ENOENT for files created inside it.
        #
        # Renaming first is atomic and frees the original name at once. If the
        # subsequent delete then fails, the cost is a leaked directory rather
        # than a broken clone.
        doomed = resolved.with_name(f"{resolved.name}.deleting-{uuid.uuid4().hex[:8]}")
        try:
            resolved.rename(doomed)
        except OSError:
            doomed = resolved

        shutil.rmtree(doomed, onerror=_on_error)

    def switch_branch(self, thread_id: str, branch: str) -> Workspace:
        """Human-requested branch switch. Refuses to discard uncommitted work."""
        workspace = self.require(thread_id)
        branch = git_service.validate_branch_name(branch)

        available = git_service.list_branches(workspace.path)
        if branch not in available["all"]:
            raise ValidationError(f"Branch {branch!r} does not exist in this repository.")

        result = git_service.checkout_branch(workspace.path, branch)
        if not result.ok:
            raise WorkspaceError(f"Could not switch branch: {result.stderr or result.stdout}")

        workspace.branch = git_service.current_branch(workspace.path)
        workspace.head_sha = git_service.head_sha(workspace.path)
        return workspace

    # -- inspection ------------------------------------------------------
    def status(self, thread_id: str) -> dict[str, Any]:
        workspace = self.get(thread_id)
        if workspace is None:
            return {"loaded": False}
        return {
            "loaded": True,
            **workspace.to_dict(),
            "clean": git_service.is_clean(workspace.path),
        }

    def branches(self, thread_id: str) -> dict[str, Any]:
        workspace = self.require(thread_id)
        return git_service.list_branches(workspace.path)

    def tree(self, thread_id: str, relative_dir: str = "", *, max_entries: int = 500) -> dict[str, Any]:
        """One directory level, with secret and blocked entries filtered out."""
        workspace = self.require(thread_id)

        if relative_dir in {"", ".", "/"}:
            target = workspace.path
        else:
            target = resolve_in_workspace(workspace.path, relative_dir)

        if not target.is_dir():
            raise ValidationError(f"Not a directory: {relative_dir!r}")

        directories: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []

        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            rel = relative_to_workspace(workspace.path, entry)

            if entry.is_dir():
                if entry.name in BLOCKED_DIRECTORIES:
                    continue
                directories.append({"name": entry.name, "path": rel, "type": "dir"})
            else:
                if is_secret_path(rel):
                    continue
                try:
                    size = entry.stat().st_size
                except OSError:
                    continue
                files.append(
                    {"name": entry.name, "path": rel, "type": "file", "size": size}
                )

            if len(directories) + len(files) >= max_entries:
                break

        return {
            "path": relative_to_workspace(workspace.path, target) if target != workspace.path else "",
            "entries": directories + files,
            "truncated": len(directories) + len(files) >= max_entries,
        }

    def read_file(self, thread_id: str, relative_path: str) -> dict[str, Any]:
        """Read a file for the browser file viewer.

        Refuses secrets, binaries, and oversized files. Redaction still runs on
        the way out: a repository may contain a leaked credential in a file we
        were entitled to read.
        """
        workspace = self.require(thread_id)
        path = resolve_in_workspace(workspace.path, relative_path)

        if not path.is_file():
            raise ValidationError(f"Not a file: {relative_path!r}")

        size = path.stat().st_size
        if size > LIMITS.max_viewable_file_bytes:
            raise ValidationError(
                f"File is too large to display ({size // 1024} KB; limit "
                f"{LIMITS.max_viewable_file_bytes // 1024} KB)."
            )
        if is_binary_file(path):
            raise ValidationError("Binary files cannot be displayed.")

        text = path.read_text(encoding="utf-8", errors="replace")
        return {
            "path": relative_to_workspace(workspace.path, path),
            "size": size,
            "lines": text.count("\n") + 1,
            "content": redact_secrets(text),
        }
