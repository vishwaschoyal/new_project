"""Path confinement, secret filtering, and output redaction.

This module is the single chokepoint every file access passes through — the
file viewer, all four read tools, the edit tool, and the sandbox. Centralising
it means a traversal fix lands everywhere at once rather than in one caller.

Three separate guarantees, deliberately not collapsed into one check:

1. **Confinement** — a resolved path must stay inside its workspace, after
   symlinks are followed. Resolving first and comparing second is what defeats
   both ``../`` traversal and a symlink pointing out of the tree.
2. **Secrecy** — some paths inside the workspace are still off limits (``.env``,
   key material, ``.git`` internals). Confinement alone would happily read them.
3. **Redaction** — anything that reaches the model or the browser is scrubbed of
   credential-shaped strings, because a repository can contain a leaked key in a
   file we had every right to read.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from core.errors import SafetyError

# --------------------------------------------------------------------------
# Secret paths
# --------------------------------------------------------------------------
# Matched against the repository-relative POSIX path, case-insensitively.
SECRET_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keystore",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.ppk",
    "credentials",
    "credentials.*",
    ".npmrc",
    ".pypirc",
    ".netrc",
    "secrets.*",
    "*secrets.yaml",
    "*secrets.yml",
    "*.jks",
    "service-account*.json",
    ".htpasswd",
)

# Directories whose contents are never served or searched. ``.git`` is excluded
# because its object store and config can contain credentials and packed
# history the user did not ask us to expose; Git itself is reached through the
# git CLI, not by reading these files.
BLOCKED_DIRECTORIES: frozenset[str] = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", ".tox", ".mypy_cache",
     ".pytest_cache", "dist", "build", ".next", ".terraform"}
)

# .env.example / .env.sample are templates with blank values — safe and useful.
_SECRET_EXCEPTIONS: frozenset[str] = frozenset(
    {".env.example", ".env.sample", ".env.template", ".env.dist"}
)


def is_secret_path(relative_path: str) -> bool:
    """Whether a repository-relative path must never be read or served."""
    posix = PurePosixPath(str(relative_path).replace("\\", "/"))
    name_lower = posix.name.lower()

    if name_lower in _SECRET_EXCEPTIONS:
        return False

    lower = PurePosixPath(str(posix).lower())
    for pattern in SECRET_FILE_PATTERNS:
        if lower.match(pattern) or PurePosixPath(name_lower).match(pattern):
            return True
    return False


def is_blocked_directory(relative_path: str) -> bool:
    parts = PurePosixPath(str(relative_path).replace("\\", "/")).parts
    return any(part in BLOCKED_DIRECTORIES for part in parts)


# --------------------------------------------------------------------------
# Path confinement
# --------------------------------------------------------------------------
def resolve_in_workspace(
    workspace_root: Path,
    relative_path: str,
    *,
    must_exist: bool = True,
    allow_secret: bool = False,
) -> Path:
    """Resolve ``relative_path`` inside ``workspace_root`` or raise SafetyError.

    Rejects absolute paths, drive letters, UNC paths, ``..`` traversal, symlinks
    that leave the workspace, blocked directories, and secret files.
    """
    raw = str(relative_path or "").strip().replace("\\", "/")
    if not raw or raw in {".", "./"}:
        raise SafetyError("A file path is required.")

    if raw.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", raw):
        raise SafetyError(f"Absolute paths are not allowed: {relative_path!r}")

    if "\x00" in raw:
        raise SafetyError("Path contains a null byte.")

    root = workspace_root.resolve()
    candidate = (root / raw).resolve()

    # Compare *after* resolution so symlink escapes are caught alongside `../`.
    if candidate != root and root not in candidate.parents:
        raise SafetyError(f"Path escapes the workspace: {relative_path!r}")

    rel = candidate.relative_to(root).as_posix()

    if is_blocked_directory(rel):
        raise SafetyError(f"Path is inside a blocked directory: {rel}")

    if not allow_secret and is_secret_path(rel):
        raise SafetyError(f"Refusing to access a secret file: {rel}")

    if must_exist and not candidate.exists():
        raise SafetyError(f"Path does not exist: {rel}")

    return candidate


def relative_to_workspace(workspace_root: Path, path: Path) -> str:
    """Repository-relative POSIX path, for stamping onto every result."""
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.name


# --------------------------------------------------------------------------
# Binary detection
# --------------------------------------------------------------------------
_TEXT_HINTS = b"\t\n\r\f\b"


def is_binary_file(path: Path, *, sniff_bytes: int = 8192) -> bool:
    """Heuristic: a NUL byte, or a high proportion of non-text bytes."""
    try:
        chunk = path.open("rb").read(sniff_bytes)
    except OSError:
        return True
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    nontext = sum(
        1 for b in chunk if b < 32 and bytes([b]) not in _TEXT_HINTS
    )
    return nontext / len(chunk) > 0.30


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------
# Ordered most-specific first; each captures a recognisable prefix so the
# replacement still shows *what kind* of credential was removed.
_REDACTION_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"), "sk-proj-***redacted***"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "sk-***redacted***"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "gh*_***redacted***"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github_pat_***redacted***"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), "xox*-***redacted***"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA***redacted***"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "AIza***redacted***"),
    (
        re.compile(
            r"-----BEGIN[A-Z ]*PRIVATE KEY-----.*?-----END[A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "***redacted private key***",
    ),
    # Generic assignment: KEY=..., "token": "...", secret: '...'
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|secret|password|passwd|token|access[_-]?key)"
            r"[\"']?\s*[:=]\s*)[\"']?([A-Za-z0-9_\-/+.]{16,})[\"']?"
        ),
        r"\1***redacted***",
    ),
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
     "***redacted jwt***"),
)


def redact_secrets(text: str) -> str:
    """Scrub credential-shaped strings from anything leaving the workspace."""
    if not text:
        return text
    for pattern, replacement in _REDACTION_RULES:
        text = pattern.sub(replacement, text)
    return text
