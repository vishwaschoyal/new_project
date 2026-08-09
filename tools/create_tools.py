"""The ``create`` tool: write one file that does not exist yet.

Kept separate from ``edit`` deliberately. ``edit`` is defined by never
destroying anything it did not first quote back exactly, and the way to keep
that guarantee is to give it no path at all to a whole-file write. So creation
is a second tool with the mirror-image rule:

- ``edit`` refuses when the target is **missing** — the model must re-read.
- ``create`` refuses when the target **exists** — the model must use ``edit``.

Between the two there is no call that can silently replace a file's contents,
which is what keeps every change reviewable as a diff of known lines.

Directories are made along the way rather than by a tool of their own, because
git does not track an empty directory. A folder is not a thing you can commit;
it exists once a file is in it. ``create`` on ``pkg/parsers/json.py`` is the
honest way to ask for the folder as well.
"""

from __future__ import annotations

from pathlib import Path

from core.errors import SafetyError
from core.safety import relative_to_workspace, resolve_in_workspace
from tools.base import ToolResult

# A new file the model writes in one call. Large enough for a real module,
# small enough that "generate the whole app" is not on the table.
MAX_NEW_FILE_CHARS = 32_000
PREVIEW_LINES = 40


def _missing_parents(workspace_root: Path, target: Path) -> list[str]:
    """Directories that do not exist yet, outermost first.

    Collected *before* the write so the result can report what was created.
    Without it a new folder appears in the diff with no mention that the tool
    made it, which is exactly the kind of unannounced side effect this project
    tries not to have.
    """
    missing: list[str] = []
    parent = target.parent
    while parent != workspace_root and parent != parent.parent:
        if parent.exists():
            break
        missing.append(relative_to_workspace(workspace_root, parent))
        parent = parent.parent
    missing.reverse()
    return missing


def create(workspace: Path, *, path: str, content: str) -> ToolResult:
    """Write ``content`` to a new file at ``path``, creating parent folders."""
    raw = str(path or "").strip().replace("\\", "/")
    if raw.endswith("/"):
        return ToolResult.failure(
            "create",
            f"{raw!r} names a directory, and create writes files. Git cannot track an "
            f"empty directory, so create the file you actually want inside it — for "
            f"example {raw}README.md — and the directory is created with it.",
        )

    if content is None:
        return ToolResult.failure("create", "content is required.")
    if not str(content).strip():
        return ToolResult.failure(
            "create", "content is empty. Write the file's actual contents."
        )
    if len(content) > MAX_NEW_FILE_CHARS:
        return ToolResult.failure(
            "create",
            f"New files must be under {MAX_NEW_FILE_CHARS:,} characters (this one is "
            f"{len(content):,}). Create the file with its core, then extend it with edit.",
        )
    if "\x00" in content:
        return ToolResult.failure(
            "create", "content contains a null byte; create writes text files only."
        )

    try:
        target = resolve_in_workspace(workspace, raw, must_exist=False)
    except SafetyError as exc:
        return ToolResult.failure("create", str(exc))

    workspace_root = Path(workspace).resolve()
    rel_path = relative_to_workspace(workspace_root, target)

    if target.exists():
        kind = "a directory" if target.is_dir() else "a file"
        return ToolResult.failure(
            "create",
            f"{rel_path} already exists ({kind}). create only writes new files — read it "
            f"and use edit to change what is there.",
            path=rel_path,
            existed=True,
        )

    # New files get LF and a trailing newline: they have no existing convention
    # to preserve, and every other tool in the chain (git, diff, POSIX) assumes
    # a final newline.
    body = str(content).replace("\r\n", "\n").replace("\r", "\n")
    if not body.endswith("\n"):
        body += "\n"

    new_directories = _missing_parents(workspace_root, target)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # "x" rather than "w": if the path appeared between the check above and
        # this write, fail rather than overwrite it.
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(body)
    except FileExistsError:
        return ToolResult.failure(
            "create", f"{rel_path} already exists.", path=rel_path, existed=True
        )
    except OSError as exc:
        return ToolResult.failure(
            "create", f"Could not create {rel_path}: {exc}", path=rel_path
        )

    lines = body.splitlines()
    preview = "\n".join(
        f"{number:>4} | {line}"
        for number, line in enumerate(lines[:PREVIEW_LINES], start=1)
    )
    if len(lines) > PREVIEW_LINES:
        preview += f"\n     | … {len(lines) - PREVIEW_LINES} more line(s)"

    folders = (
        f" New folder(s): {', '.join(new_directories)}." if new_directories else ""
    )
    return ToolResult.success(
        "create",
        f"Created {rel_path} — {len(lines)} line(s).{folders}\n\n{preview}",
        path=rel_path,
        lines=len(lines),
        created_directories=new_directories,
    )
