"""The ``edit`` tool: one exact, unique string replacement per call.

Determinism is the whole design. A model that can supply a fuzzy target or a
line number will eventually hit the wrong place in a file it half-remembers.
Requiring an exact string that occurs *exactly once* means the edit either lands
where the model intended or does not happen at all:

- **zero matches** → the model's memory of the file is stale; it must re-read.
- **two or more matches** → the target is ambiguous; the model must include
  surrounding context to disambiguate.

Both are refusals, and both return the information needed to retry correctly.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from config import LIMITS
from core.errors import SafetyError
from core.safety import is_binary_file, relative_to_workspace, resolve_in_workspace
from tools.base import ToolResult

MAX_EDIT_STRING_CHARS = 8_000


def _preview_diff(rel_path: str, before: str, after: str, *, context: int = 3) -> str:
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        n=context,
        lineterm="",
    )
    return "\n".join(diff)


def _line_numbers_of(haystack: str, needle: str) -> list[int]:
    """1-based line numbers where each occurrence starts."""
    positions: list[int] = []
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            return positions
        positions.append(haystack.count("\n", 0, index) + 1)
        start = index + 1


def edit(
    workspace: Path,
    *,
    path: str,
    old_string: str,
    new_string: str,
) -> ToolResult:
    """Replace one exact occurrence of ``old_string`` with ``new_string``."""
    if old_string == new_string:
        return ToolResult.failure("edit", "old_string and new_string are identical.")
    if not old_string:
        return ToolResult.failure(
            "edit", "old_string is required. Creating whole files is not supported by edit."
        )
    if len(old_string) > MAX_EDIT_STRING_CHARS or len(new_string) > MAX_EDIT_STRING_CHARS:
        return ToolResult.failure(
            "edit", f"Edit strings must be under {MAX_EDIT_STRING_CHARS} characters."
        )

    try:
        target = resolve_in_workspace(workspace, path)
    except SafetyError as exc:
        return ToolResult.failure("edit", str(exc))

    if not target.is_file():
        return ToolResult.failure("edit", f"Not a file: {path}")
    if is_binary_file(target):
        return ToolResult.failure("edit", f"{path} is a binary file and cannot be edited.")
    if target.stat().st_size > LIMITS.max_file_bytes:
        return ToolResult.failure("edit", f"{path} is too large to edit safely.")

    rel_path = relative_to_workspace(workspace, target)
    original = target.read_text(encoding="utf-8", errors="strict")

    occurrences = original.count(old_string)
    if occurrences == 0:
        return ToolResult.failure(
            "edit",
            f"old_string was not found in {rel_path}. The file may differ from what you "
            f"read — re-read the relevant range and retry with the exact current text.",
            matches=0,
            path=rel_path,
        )
    if occurrences > 1:
        lines = _line_numbers_of(original, old_string)
        return ToolResult.failure(
            "edit",
            f"old_string appears {occurrences} times in {rel_path} (lines "
            f"{', '.join(str(n) for n in lines[:10])}). Include surrounding context so the "
            f"target is unique.",
            matches=occurrences,
            path=rel_path,
        )

    updated = original.replace(old_string, new_string, 1)

    # Preserve the file's existing line endings: rewriting a CRLF file with LF
    # turns a one-line change into a whole-file diff.
    newline = "\r\n" if "\r\n" in original else "\n"
    with target.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(updated.replace("\r\n", "\n"))

    start_line = original.count("\n", 0, original.find(old_string)) + 1
    return ToolResult.success(
        "edit",
        f"Edited {rel_path} at line {start_line}.\n\n"
        f"{_preview_diff(rel_path, original, updated)}",
        path=rel_path,
        line=start_line,
        added_lines=new_string.count("\n") + 1,
        removed_lines=old_string.count("\n") + 1,
    )
