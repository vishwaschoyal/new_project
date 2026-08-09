"""Task branches, review diffs, and the publication approval gate.

The rule this module exists to enforce: **nothing reaches a remote, or even a
commit, without an explicit human decision made against the actual diff.**

Two structural guarantees back that up rather than relying on prompt wording:

1. The agent's tools cannot commit, push, or open a pull request. Those verbs
   exist only here, and only behind ``approve``.
2. Editing requires a task branch, and a task branch is never the default
   branch. An agent that misbehaves damages a scratch branch, not `main`.

Every state change is appended to an audit trail: which files were edited, which
checks ran and what they returned, and who approved what and when.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from config import GITHUB_TOKEN
from core.errors import ForbiddenError, NotFoundError, ValidationError, WorkspaceError
from services import git_service, github_service
from services.workspace_service import Workspace

log = logging.getLogger(__name__)

COMMIT_AUTHOR_NAME = "AI Coding Workspace"
COMMIT_AUTHOR_EMAIL = "noreply@ai-coding-workspace.local"

VALID_ACTIONS = frozenset({"commit", "push", "pull_request"})


def slugify(text: str, *, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:max_length].rstrip("-")) or "task"


@dataclass
class AuditEvent:
    kind: str
    detail: str
    at: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "at": self.at, **self.data}


@dataclass
class TaskSession:
    """One coding task on one task branch."""

    thread_id: str
    description: str
    branch: str
    base_branch: str
    created_at: float = field(default_factory=time.time)
    edited_files: set[str] = field(default_factory=set)
    checks: list[dict[str, Any]] = field(default_factory=list)
    audit: list[AuditEvent] = field(default_factory=list)
    committed: bool = False
    pushed: bool = False
    pull_request_url: str = ""

    def log(self, kind: str, detail: str, **data: Any) -> None:
        self.audit.append(AuditEvent(kind, detail, data=data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "description": self.description,
            "branch": self.branch,
            "base_branch": self.base_branch,
            "created_at": self.created_at,
            "edited_files": sorted(self.edited_files),
            "checks": self.checks,
            "committed": self.committed,
            "pushed": self.pushed,
            "pull_request_url": self.pull_request_url,
            "audit": [event.to_dict() for event in self.audit],
        }


class PublishService:
    def __init__(self):
        self._tasks: dict[str, TaskSession] = {}
        self._lock = threading.RLock()

    # -- task lifecycle --------------------------------------------------
    def start_task(self, workspace: Workspace, description: str) -> TaskSession:
        """Create and switch to a task branch. Required before any edit."""
        if not (description or "").strip():
            raise ValidationError("A task description is required.")

        base = git_service.current_branch(workspace.path) or workspace.default_branch
        branch = f"ai/{slugify(description)}-{int(time.time()) % 100000}"

        if not git_service.is_clean(workspace.path):
            raise WorkspaceError(
                "The workspace has uncommitted changes. Review or discard them before "
                "starting a new task."
            )

        result = git_service.create_task_branch(workspace.path, branch)
        if not result.ok:
            raise WorkspaceError(f"Could not create task branch: {result.output}")

        workspace.task_branch = branch
        workspace.branch = git_service.current_branch(workspace.path)

        task = TaskSession(
            thread_id=workspace.thread_id,
            description=description.strip(),
            branch=branch,
            base_branch=base,
        )
        task.log("task_started", f"Created task branch {branch} from {base}")
        with self._lock:
            self._tasks[workspace.thread_id] = task

        log.info(
            "task started",
            extra={"thread_id": workspace.thread_id, "branch": branch, "base": base},
        )
        return task

    def get(self, thread_id: str) -> TaskSession | None:
        with self._lock:
            return self._tasks.get(thread_id)

    def require(self, thread_id: str) -> TaskSession:
        task = self.get(thread_id)
        if task is None:
            raise NotFoundError("No coding task is active for this session.")
        return task

    def ensure_editable(self, workspace: Workspace) -> TaskSession:
        """Guard called before an editing run. Refuses the default branch."""
        task = self.get(workspace.thread_id)
        if task is None:
            raise ForbiddenError(
                "Editing requires an active task branch. Start a coding task first."
            )
        current = git_service.current_branch(workspace.path)
        if current == workspace.default_branch:
            raise ForbiddenError(
                f"Refusing to edit the default branch ({workspace.default_branch})."
            )
        return task

    # -- progress recording ----------------------------------------------
    def record_edit(
        self, thread_id: str, path: str, line: int | None = None, *, created: bool = False
    ) -> None:
        task = self.get(thread_id)
        if task is None:
            return
        task.edited_files.add(path)
        task.log(
            "edit",
            f"{'Created' if created else 'Edited'} {path}"
            + (f" at line {line}" if line else ""),
            path=path,
        )

    def record_check(self, thread_id: str, check: dict[str, Any]) -> None:
        task = self.get(thread_id)
        if task is None:
            return
        task.checks.append(check)
        status = "passed" if check.get("ok") else "failed"
        task.log(
            "check",
            f"{check.get('command', 'check')} {status}",
            ok=bool(check.get("ok")),
            backend=check.get("backend"),
        )

    # -- review -----------------------------------------------------------
    def review(self, workspace: Workspace) -> dict[str, Any]:
        """The diff the user approves or rejects."""
        task = self.require(workspace.thread_id)

        # Files `create` wrote are untracked, and untracked files are absent
        # from `git diff`. Register them first so the reviewer sees every file
        # the run produced, not only the ones it modified.
        git_service.mark_intent_to_add(workspace.path)

        working_diff = git_service.diff(workspace.path)
        committed_diff = git_service.diff(workspace.path, base=task.base_branch) if task.committed else ""

        return {
            "task": task.to_dict(),
            "diff": working_diff or committed_diff,
            "diff_stat": git_service.diff_stat(workspace.path, base=task.base_branch if task.committed else None),
            "status": git_service.status_short(workspace.path),
            "has_changes": bool(working_diff.strip() or committed_diff.strip()),
            "checks_passed": all(c.get("ok") for c in task.checks) if task.checks else None,
        }

    # -- the approval gate -------------------------------------------------
    def approve(
        self,
        workspace: Workspace,
        *,
        action: str,
        commit_message: str = "",
        pr_title: str = "",
        pr_body: str = "",
        approved_by: str = "user",
    ) -> dict[str, Any]:
        """Publish an approved change. The only path to a commit or a remote.

        Actions escalate: ``commit`` is local, ``push`` implies commit, and
        ``pull_request`` implies push. Each step is recorded before the next
        runs, so a failure halfway leaves an accurate trail.
        """
        if action not in VALID_ACTIONS:
            raise ValidationError(f"Unknown action {action!r}. Expected one of {sorted(VALID_ACTIONS)}.")

        task = self.require(workspace.thread_id)
        outcome: dict[str, Any] = {"action": action, "branch": task.branch}

        # -- commit
        if not task.committed:
            if not git_service.status_short(workspace.path).strip():
                raise ValidationError("There are no changes to publish.")

            message = (commit_message or f"{task.description}").strip()
            staged = git_service.stage_all(workspace.path)
            if not staged.ok:
                raise WorkspaceError(f"Could not stage changes: {staged.output}")

            committed = git_service.commit(
                workspace.path, message,
                author_name=COMMIT_AUTHOR_NAME, author_email=COMMIT_AUTHOR_EMAIL,
            )
            if not committed.ok:
                raise WorkspaceError(f"Commit failed: {committed.output}")

            task.committed = True
            task.log("commit", message, approved_by=approved_by)
            outcome["commit"] = message
            outcome["head_sha"] = git_service.head_sha(workspace.path)[:8]

        if action == "commit":
            log.info("change committed", extra={"thread_id": workspace.thread_id, "branch": task.branch})
            return outcome

        # -- push
        if not task.pushed:
            pushed = git_service.push(workspace.path, task.branch, token=GITHUB_TOKEN or None)
            if not pushed.ok:
                raise WorkspaceError(f"Push failed: {pushed.output}")
            task.pushed = True
            task.log("push", f"Pushed {task.branch} to origin", approved_by=approved_by)
            outcome["pushed"] = True
            outcome["branch_url"] = f"{workspace.repo.html_url}/tree/{task.branch}"

        if action == "push":
            return outcome

        # -- pull request
        if task.pull_request_url:
            outcome["pull_request_url"] = task.pull_request_url
            return outcome

        pull_request = github_service.create_pull_request(
            workspace.repo,
            title=(pr_title or task.description)[:250],
            body=pr_body or _default_pr_body(task),
            head=task.branch,
            base=task.base_branch,
        )
        task.pull_request_url = pull_request["html_url"]
        task.log("pull_request", f"Opened PR #{pull_request['number']}", approved_by=approved_by)
        outcome["pull_request_url"] = task.pull_request_url
        outcome["pull_request_number"] = pull_request["number"]

        log.info(
            "pull request opened",
            extra={"thread_id": workspace.thread_id, "url": task.pull_request_url},
        )
        return outcome

    def discard(self, workspace: Workspace) -> dict[str, Any]:
        """Abandon a task: reset the branch and return to the base branch."""
        task = self.require(workspace.thread_id)

        git_service._run(["reset", "--hard"], cwd=workspace.path)
        git_service._run(["clean", "-fd"], cwd=workspace.path)
        git_service._run(["checkout", task.base_branch], cwd=workspace.path)

        task.log("discarded", f"Discarded {task.branch}; returned to {task.base_branch}")
        with self._lock:
            self._tasks.pop(workspace.thread_id, None)

        workspace.task_branch = None
        workspace.branch = git_service.current_branch(workspace.path)
        return {"discarded": True, "branch": workspace.branch}


def _default_pr_body(task: TaskSession) -> str:
    lines = [task.description, "", "## Changes", ""]
    lines += [f"- `{path}`" for path in sorted(task.edited_files)] or ["- (no files recorded)"]

    if task.checks:
        lines += ["", "## Verification", ""]
        for check in task.checks:
            status = "✅ passed" if check.get("ok") else "❌ failed"
            isolation = "" if check.get("isolated") else " _(ran without container isolation)_"
            lines.append(f"- `{check.get('command')}` — {status}{isolation}")

    lines += ["", "---", "", "🤖 Generated with an AI coding workspace and approved by a human reviewer."]
    return "\n".join(lines)


publisher = PublishService()
