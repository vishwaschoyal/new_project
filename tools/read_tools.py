"""The four read-only tools: grep, glob, read, bash.

Every result is stamped with a repository-relative path and, where meaningful, a
line number. That stamping is what makes a citation checkable later: an answer
can only cite coordinates that appeared in an observation during this request.

All four are bounded on result count, character count, and wall time. None of
them can modify the repository.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from config import LIMITS
from core.errors import SafetyError
from core.safety import (
    BLOCKED_DIRECTORIES,
    is_binary_file,
    is_secret_path,
    relative_to_workspace,
    resolve_in_workspace,
)
from tools.base import ToolResult, clamp_text

_RG_EXCLUDES: list[str] = []
for _directory in sorted(BLOCKED_DIRECTORIES):
    _RG_EXCLUDES += ["--glob", f"!{_directory}/**"]
for _pattern in (".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "*.ppk"):
    _RG_EXCLUDES += ["--glob", f"!{_pattern}"]


def _run(args: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        shell=False,  # never a shell string: no injection surface
    )


# --------------------------------------------------------------------------
# grep
# --------------------------------------------------------------------------
def grep(
    workspace: Path,
    *,
    pattern: str,
    path: str = "",
    glob: str = "",
    case_sensitive: bool = False,
    max_results: int = LIMITS.max_grep_matches,
) -> ToolResult:
    """Search file contents; return ``file:line:text`` entries."""
    if not (pattern or "").strip():
        return ToolResult.failure("grep", "A search pattern is required.")

    try:
        re.compile(pattern)
    except re.error as exc:
        return ToolResult.failure("grep", f"Invalid regular expression: {exc}")

    # "." and "" both mean the repository root — a natural thing for the model
    # to send, and refusing it wastes a step for no safety benefit.
    search_root = workspace
    target = "."
    scope = (path or "").strip().strip("/")
    if scope and scope not in {".", "./"}:
        try:
            scoped = resolve_in_workspace(workspace, scope)
        except SafetyError as exc:
            return ToolResult.failure("grep", str(exc))
        if scoped.is_dir():
            search_root = scoped
        else:
            # Narrowing a search to one file is a reasonable thing to ask for,
            # and ripgrep takes a file argument happily. Handing that path to
            # subprocess as the working directory instead raises
            # NotADirectoryError, which reaches the model as an exception name
            # it cannot act on — it burns a step and guesses again.
            search_root = scoped.parent
            target = scoped.name

    limit = max(1, min(max_results, LIMITS.max_grep_matches))
    args = [
        "rg", "--line-number", "--no-heading", "--color", "never",
        # ripgrep drops the filename prefix when handed exactly one file, which
        # would leave the parser below reading the line number as the path.
        "--with-filename",
        "--max-count", "10",          # per file, so one file cannot fill the budget
        "--max-columns", "300",
        "--max-filesize", "1M",
        *_RG_EXCLUDES,
    ]
    if not case_sensitive:
        args.append("--ignore-case")
    if glob:
        args += ["--glob", glob]
    args += ["--regexp", pattern, target]

    try:
        proc = _run(args, cwd=search_root, timeout=LIMITS.tool_timeout_seconds)
    except subprocess.TimeoutExpired:
        return ToolResult.failure("grep", f"Search timed out after {LIMITS.tool_timeout_seconds}s.")
    except FileNotFoundError:
        return ToolResult.failure("grep", "ripgrep (rg) is not installed or not on PATH.")
    except OSError as exc:
        # A vanished directory, a permission refusal, a path the OS will not
        # accept as a working directory. The loop can act on a sentence; it
        # cannot act on a traceback class name.
        return ToolResult.failure("grep", f"Search could not run: {exc}")

    # rg exits 1 for "no matches", which is a valid answer, not a failure.
    if proc.returncode not in (0, 1):
        return ToolResult.failure("grep", (proc.stderr or "search failed").strip()[:500])

    lines: list[str] = []
    for raw in proc.stdout.splitlines():
        if not raw.strip():
            continue
        # rg output relative to search_root; restamp against the workspace root.
        parts = raw.split(":", 2)
        if len(parts) < 3:
            continue
        file_part, line_no, text = parts
        absolute = (search_root / file_part).resolve()
        rel = relative_to_workspace(workspace, absolute)
        if is_secret_path(rel):
            continue
        lines.append(f"{rel}:{line_no}:{text.strip()}")
        if len(lines) >= limit:
            break

    if not lines:
        return ToolResult.success(
            "grep", f"No matches for {pattern!r}.", matches=0, pattern=pattern
        )

    body, char_truncated = clamp_text("\n".join(lines), LIMITS.max_tool_output_chars)
    truncated = char_truncated or len(lines) >= limit
    return ToolResult.success(
        "grep",
        body,
        matches=len(lines),
        pattern=pattern,
        truncated=truncated,
        shown=len(lines),
        limit=limit,
    )


# --------------------------------------------------------------------------
# glob
# --------------------------------------------------------------------------
def glob(
    workspace: Path,
    *,
    pattern: str,
    max_results: int = LIMITS.max_glob_paths,
) -> ToolResult:
    """Find files by path pattern; return sorted repository-relative paths."""
    if not (pattern or "").strip():
        return ToolResult.failure("glob", "A path pattern is required.")
    if pattern.startswith("/") or re.match(r"^[A-Za-z]:", pattern) or ".." in pattern:
        return ToolResult.failure("glob", "Pattern must be relative and must not contain '..'.")

    limit = max(1, min(max_results, LIMITS.max_glob_paths))
    matches: list[str] = []

    try:
        for candidate in sorted(workspace.glob(pattern)):
            if not candidate.is_file():
                continue
            rel = relative_to_workspace(workspace, candidate)
            if is_secret_path(rel) or any(p in BLOCKED_DIRECTORIES for p in Path(rel).parts):
                continue
            # Defence in depth: glob can follow a symlink out of the tree.
            resolved = candidate.resolve()
            if workspace.resolve() not in resolved.parents:
                continue
            matches.append(rel)
            if len(matches) >= limit:
                break
    except (OSError, ValueError) as exc:
        return ToolResult.failure("glob", f"Invalid pattern: {exc}")

    if not matches:
        return ToolResult.success("glob", f"No files match {pattern!r}.", count=0)

    return ToolResult.success(
        "glob",
        "\n".join(matches),
        count=len(matches),
        pattern=pattern,
        truncated=len(matches) >= limit,
        shown=len(matches),
        limit=limit,
    )


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------
def read(
    workspace: Path,
    *,
    path: str,
    offset: int = 1,
    limit: int = LIMITS.max_read_lines,
) -> ToolResult:
    """Read a focused line range, returned as ``lineno | text``."""
    try:
        target = resolve_in_workspace(workspace, path)
    except SafetyError as exc:
        return ToolResult.failure("read", str(exc))

    if not target.is_file():
        return ToolResult.failure("read", f"Not a file: {path}")
    if is_binary_file(target):
        return ToolResult.failure("read", f"{path} is a binary file and cannot be read.")

    size = target.stat().st_size
    if size > LIMITS.max_file_bytes:
        return ToolResult.failure(
            "read", f"{path} is too large to read ({size // 1024} KB)."
        )

    offset = max(1, int(offset or 1))
    limit = max(1, min(int(limit or LIMITS.max_read_lines), LIMITS.max_read_lines))

    try:
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            all_lines = handle.readlines()
    except OSError as exc:
        return ToolResult.failure("read", f"Could not read {path}: {exc}")

    total = len(all_lines)
    if offset > total:
        return ToolResult.failure(
            "read", f"{path} has {total} lines; offset {offset} is past the end."
        )

    selected = all_lines[offset - 1 : offset - 1 + limit]
    width = len(str(offset + len(selected) - 1))
    rendered = "\n".join(
        f"{offset + i:>{width}} | {line.rstrip()}" for i, line in enumerate(selected)
    )

    body, char_truncated = clamp_text(rendered, LIMITS.max_tool_output_chars)
    end_line = offset + len(selected) - 1
    header = f"{relative_to_workspace(workspace, target)} lines {offset}-{end_line} of {total}"

    return ToolResult.success(
        "read",
        f"{header}\n{body}",
        path=relative_to_workspace(workspace, target),
        start_line=offset,
        end_line=end_line,
        total_lines=total,
        truncated=char_truncated or end_line < total,
    )


# --------------------------------------------------------------------------
# bash — approved read-only inspection only
# --------------------------------------------------------------------------
# Allow-list, not deny-list. A deny-list of dangerous commands is unwinnable:
# every new binary on the image is a hole. Only these programs run, and for git
# only these subcommands.
_ALLOWED_GIT_SUBCOMMANDS = frozenset(
    {"log", "show", "diff", "status", "branch", "blame", "ls-files", "ls-tree",
     "rev-parse", "describe", "shortlog", "tag", "remote", "config"}
)
_ALLOWED_PROGRAMS = frozenset({"git", "rg", "ls", "wc", "find", "tree"})

# Any of these turn one command into several, or redirect output to disk.
_SHELL_METACHARACTERS = re.compile(r"[;&|><`$\n\r\\]|\$\(|\)\s*\{")

_GIT_WRITE_FLAGS = frozenset(
    {"--set", "--add", "--unset", "--replace-all", "--edit", "-e"}
)


def _reject_command(command: str) -> str | None:
    """Return a refusal reason, or None if the command may run."""
    if not command.strip():
        return "A command is required."
    if _SHELL_METACHARACTERS.search(command):
        return (
            "Shell operators (; & | > < ` $ \\) are not permitted. "
            "Run one simple command."
        )
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Could not parse command: {exc}"
    if not parts:
        return "A command is required."

    program = parts[0]
    if program not in _ALLOWED_PROGRAMS:
        return (
            f"{program!r} is not an approved command. "
            f"Approved: {', '.join(sorted(_ALLOWED_PROGRAMS))}."
        )

    if program == "git":
        subcommand = next((p for p in parts[1:] if not p.startswith("-")), "")
        if subcommand not in _ALLOWED_GIT_SUBCOMMANDS:
            return (
                f"git {subcommand!r} is not a read-only subcommand. "
                f"Approved: {', '.join(sorted(_ALLOWED_GIT_SUBCOMMANDS))}."
            )
        # `git config --set x y` writes; only reads are allowed.
        if subcommand == "config" and any(f in _GIT_WRITE_FLAGS for f in parts):
            return "git config may only be used to read values."

    for part in parts[1:]:
        if part.startswith("-"):
            continue
        if part.startswith("/") or re.match(r"^[A-Za-z]:", part):
            return f"Absolute paths are not permitted: {part!r}"
        if ".." in Path(part).parts:
            return f"Paths must not traverse upward: {part!r}"
        if is_secret_path(part):
            return f"Refusing to reference a secret path: {part!r}"

    return None


def bash(workspace: Path, *, command: str) -> ToolResult:
    """Run one approved read-only inspection command inside the workspace."""
    refusal = _reject_command(command)
    if refusal:
        return ToolResult.failure("bash", refusal, refused=True)

    args = shlex.split(command)
    try:
        proc = _run(args, cwd=workspace, timeout=LIMITS.tool_timeout_seconds)
    except subprocess.TimeoutExpired:
        return ToolResult.failure(
            "bash", f"Command timed out after {LIMITS.tool_timeout_seconds}s."
        )
    except FileNotFoundError:
        return ToolResult.failure("bash", f"{args[0]!r} is not installed on this host.")
    except OSError as exc:
        return ToolResult.failure("bash", f"Command could not run: {exc}")

    stdout, out_truncated = clamp_text(proc.stdout or "", LIMITS.max_tool_output_chars)
    stderr, err_truncated = clamp_text(proc.stderr or "", 2_000)

    if proc.returncode != 0 and not stdout:
        return ToolResult.failure(
            "bash",
            f"Command exited {proc.returncode}: {stderr or 'no output'}",
            exit_code=proc.returncode,
        )

    body = stdout or "(no output)"
    if stderr:
        body += f"\n\n[stderr]\n{stderr}"

    return ToolResult.success(
        "bash",
        body,
        command=command,
        exit_code=proc.returncode,
        truncated=out_truncated or err_truncated,
    )
