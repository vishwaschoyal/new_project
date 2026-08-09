"""Tool schemas and the single dispatcher.

One registry, one dispatch path. Every tool call the model makes is validated
here — name, arguments, and *capability* — before any tool function runs.

Capabilities are the mechanism that keeps the read path read-only. A read-only
run is not merely a run where the model was told not to edit; ``edit`` is not
present in its tool list and would be refused by the dispatcher even if the
model invented the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from tools import create_tools, edit_tools, read_tools, verify_tools
from tools.base import ToolResult, Timer


class Capability(str, Enum):
    READ = "read"        # grep, glob, read, bash
    EDIT = "edit"        # edit
    VERIFY = "verify"    # run_check (sandboxed)
    DELEGATE = "delegate"  # spawn subagents


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    capability: Capability
    handler: Callable[..., ToolResult]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# --------------------------------------------------------------------------
# Specifications
# --------------------------------------------------------------------------
GREP = ToolSpec(
    name="grep",
    description=(
        "Search file contents across the repository with a regular expression. "
        "Returns 'path:line:text' entries. Use this first to locate code — never "
        "guess a file path."
    ),
    parameters=_obj(
        {
            "pattern": {
                "type": "string",
                "description": "Regular expression, e.g. 'def create_app' or 'class \\w+Store'.",
            },
            "path": {
                "type": "string",
                "description": "Optional repository-relative directory to restrict the search to.",
            },
            "glob": {
                "type": "string",
                "description": "Optional file filter, e.g. '*.py' or '**/routes/*.js'.",
            },
            "case_sensitive": {"type": "boolean", "description": "Default false."},
        },
        ["pattern"],
    ),
    capability=Capability.READ,
    handler=read_tools.grep,
)

GLOB = ToolSpec(
    name="glob",
    description=(
        "Find files by path pattern, e.g. '**/*.py' or 'src/**/test_*.js'. "
        "Use when you need to discover structure rather than content."
    ),
    parameters=_obj(
        {"pattern": {"type": "string", "description": "Repository-relative glob pattern."}},
        ["pattern"],
    ),
    capability=Capability.READ,
    handler=read_tools.glob,
)

READ = ToolSpec(
    name="read",
    description=(
        "Read a focused range of lines from one file, returned with line numbers. "
        "Read the range around a match you found — do not read whole files."
    ),
    parameters=_obj(
        {
            "path": {"type": "string", "description": "Repository-relative file path."},
            "offset": {"type": "integer", "description": "1-based first line. Default 1."},
            "limit": {"type": "integer", "description": "How many lines to read. Default 120, max 250."},
        },
        ["path"],
    ),
    capability=Capability.READ,
    handler=read_tools.read,
)

BASH = ToolSpec(
    name="bash",
    description=(
        "Run one approved read-only inspection command. Permitted: git (log, show, diff, "
        "status, branch, blame, ls-files, ls-tree, rev-parse, describe, shortlog, tag, "
        "remote, config), rg, ls, wc, find, tree. Shell operators are rejected."
    ),
    parameters=_obj(
        {"command": {"type": "string", "description": "A single command, no pipes or redirection."}},
        ["command"],
    ),
    capability=Capability.READ,
    handler=read_tools.bash,
)

EDIT = ToolSpec(
    name="edit",
    description=(
        "Replace one exact, unique string in one file. The old_string must appear exactly "
        "once — include surrounding lines to make it unique. Re-read the file first so the "
        "text matches the current contents exactly."
    ),
    parameters=_obj(
        {
            "path": {"type": "string", "description": "Repository-relative file path."},
            "old_string": {"type": "string", "description": "Exact current text, unique in the file."},
            "new_string": {"type": "string", "description": "Replacement text."},
        },
        ["path", "old_string", "new_string"],
    ),
    capability=Capability.EDIT,
    handler=edit_tools.edit,
)

CREATE = ToolSpec(
    name="create",
    description=(
        "Write a NEW file that does not exist yet. Missing parent folders are created "
        "along with it, so 'pkg/parsers/json.py' makes the folders too — there is no "
        "separate folder tool because git cannot track an empty directory. Refuses if "
        "the path already exists: to change an existing file, read it and use edit."
    ),
    parameters=_obj(
        {
            "path": {
                "type": "string",
                "description": "Repository-relative path for the new file, e.g. 'utils/text.py'.",
            },
            "content": {
                "type": "string",
                "description": "Complete contents of the file, matching the project's style.",
            },
        },
        ["path", "content"],
    ),
    capability=Capability.EDIT,
    handler=create_tools.create,
)


RUN_CHECK = ToolSpec(
    name="run_check",
    description=(
        "Run the repository's tests or build inside an isolated sandbox to verify a change. "
        "Approved runners: pytest, python, npm, npx, yarn, pnpm, node, go, cargo, make, mvn, "
        "gradle, ruff, mypy, eslint, tsc. Prefer the narrowest command that covers your change "
        "(a single test file) before running the whole suite. Read failures and fix the cause; "
        "never weaken a test to make it pass."
    ),
    parameters=_obj(
        {
            "command": {
                "type": "string",
                "description": "One command, e.g. 'pytest tests/test_auth.py -q'. No shell operators.",
            },
            "timeout": {
                "type": "integer",
                "description": "Optional timeout in seconds. Defaults to the configured sandbox limit.",
            },
        },
        ["command"],
    ),
    capability=Capability.VERIFY,
    handler=verify_tools.run_check,
)


_ALL_SPECS: tuple[ToolSpec, ...] = (GREP, GLOB, READ, BASH, EDIT, CREATE, RUN_CHECK)


class ToolRegistry:
    """Holds the tool set for one run and dispatches validated calls."""

    def __init__(
        self,
        workspace: Path,
        *,
        capabilities: set[Capability] | None = None,
        extra_specs: tuple[ToolSpec, ...] = (),
    ):
        self.workspace = Path(workspace)
        self.capabilities = capabilities or {Capability.READ}
        specs = (*_ALL_SPECS, *extra_specs)
        self._specs: dict[str, ToolSpec] = {
            spec.name: spec for spec in specs if spec.capability in self.capabilities
        }

    @property
    def names(self) -> list[str]:
        return sorted(self._specs)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI tool definitions. Order is stable so the cached prefix holds."""
        return [self._specs[name].to_openai_schema() for name in self.names]

    def dispatch(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult.failure(
                name or "unknown",
                f"Unknown or unavailable tool {name!r}. Available: {', '.join(self.names)}.",
                refused=True,
            )

        args = dict(arguments or {})

        # Reject unexpected keys rather than silently ignoring them: a model
        # passing `file` instead of `path` should be told, not quietly given a
        # default.
        allowed = set(spec.parameters["properties"])
        unexpected = set(args) - allowed
        if unexpected:
            return ToolResult.failure(
                name,
                f"Unexpected argument(s): {', '.join(sorted(unexpected))}. "
                f"Expected: {', '.join(sorted(allowed))}.",
                refused=True,
            )

        missing = [key for key in spec.parameters["required"] if not args.get(key) and args.get(key) != 0]
        if missing:
            return ToolResult.failure(
                name, f"Missing required argument(s): {', '.join(missing)}.", refused=True
            )

        with Timer() as timer:
            try:
                result = spec.handler(self.workspace, **args)
            except TypeError as exc:
                return ToolResult.failure(name, f"Invalid arguments: {exc}", refused=True)
            except Exception as exc:  # a tool bug must not end the request
                result = ToolResult.failure(name, f"{type(exc).__name__}: {exc}")

        result.tool = name
        result.metadata["duration_ms"] = timer.ms
        return result


READ_ONLY_CAPABILITIES = {Capability.READ}
EDITING_CAPABILITIES = {Capability.READ, Capability.EDIT, Capability.VERIFY}
