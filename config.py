"""Environment-backed configuration and the single model-client factory.

``ChatOpenAI`` is used here deliberately and exclusively as a *transport*: it
speaks the Responses API, adapts messages, and surfaces native tool calls and
usage metadata. It does not own the agent loop. The application owns the
messages, evidence, tool execution, budget, and stopping conditions.

No application module may construct an OpenAI SDK client directly. That rule is
enforced by ``tests/test_architecture.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# env helpers
# --------------------------------------------------------------------------
def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or "").strip() or default


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name) or default)
    except ValueError:
        return default


def _bool(name: str, default: bool = False) -> bool:
    raw = _str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# The rebuild has exactly one model switch. Keeping it here makes a model change
# a one-line .env operation instead of a provider name spread through the loop.
DEFAULT_READ_LOOP_MODEL = "gpt-5.4-mini"
READ_LOOP_MODEL = _str("READ_LOOP_MODEL", DEFAULT_READ_LOOP_MODEL)

# Bumping this key deliberately invalidates the provider-side prompt cache.
# Change it whenever the system prompt or tool schemas change, otherwise the
# provider may serve a prefix that no longer matches what we send.
READ_LOOP_PROMPT_CACHE_KEY = "read-loop-system-tools-v6"
# Workers run a different system prompt, so they are a different cacheable
# prefix. The key is a routing hint: sharing one across two prefixes makes the
# two evict each other on the machine they land on. Giving workers their own
# means the four of them — who *are* prefix-identical — route together, and
# they stop competing with the main agent's much longer-lived prefix.
SUBAGENT_PROMPT_CACHE_KEY = "subagent-system-tools-v6"
READ_LOOP_PROMPT_CACHE_RETENTION = "24h"
READ_LOOP_MAX_OUTPUT_TOKENS = 3_000

OPENAI_API_KEY = _str("OPENAI_API_KEY")
GITHUB_TOKEN = _str("GITHUB_PERSONAL_ACCESS_TOKEN")


def create_read_loop_model(
    *,
    cache_key: str = READ_LOOP_PROMPT_CACHE_KEY,
    streaming: bool = True,
    max_output_tokens: int = READ_LOOP_MAX_OUTPUT_TOKENS,
    **client_kwargs: Any,
) -> ChatOpenAI:
    """Build the model client used by the hand-written loop and every subagent.

    LangChain's own response cache is disabled on purpose: repeated requests
    must actually reach the provider so that provider-side prompt-cache hits
    stay measurable through native cached-token usage metadata.
    """
    model_kwargs = dict(client_kwargs.pop("model_kwargs", {}) or {})
    model_kwargs["prompt_cache_key"] = cache_key
    model_kwargs["prompt_cache_retention"] = READ_LOOP_PROMPT_CACHE_RETENTION
    # Batched tool calls are the single largest cost lever in the loop.
    #
    # A step re-sends every message before it, so the bill is roughly
    # steps x average-context — quadratic in step count. One call per step
    # turns "read these three files" into three full context re-sends.
    #
    # This was previously False to keep the observation order deterministic,
    # but the loop does not need the provider's help for that: the
    # orchestrator iterates `message.tool_calls` and dispatches them one at a
    # time, in the order the model listed them. Enabling batching changes how
    # many calls arrive per message, not the order they run in, so the trail
    # stays exactly as reproducible and auditable as it was.
    model_kwargs["parallel_tool_calls"] = True

    client_kwargs.setdefault("cache", False)

    return ChatOpenAI(
        model=READ_LOOP_MODEL,
        model_kwargs=model_kwargs,
        stream_usage=True,
        streaming=streaming,
        use_responses_api=True,
        max_completion_tokens=max_output_tokens,
        **client_kwargs,
    )


# --------------------------------------------------------------------------
# Limits — every bound the product enforces, in one place
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Limits:
    """Hard bounds. Nothing in the system is allowed to be unbounded."""

    # Agent loop
    max_agent_steps: int = field(default_factory=lambda: _int("MAX_AGENT_STEPS", 24))
    request_token_budget: int = field(
        default_factory=lambda: _int("REQUEST_TOKEN_BUDGET", 120_000)
    )
    # Tokens held back so a budget-exhausted run can still write its answer.
    final_synthesis_reserve: int = 8_000

    # Tools
    max_grep_matches: int = 60
    max_glob_paths: int = 100
    max_read_lines: int = 250
    # What an omitted `limit` actually gets. The tool schema has always
    # advertised 120, but the handler defaulted to `max_read_lines` — so a
    # model that trusted the description and left `limit` off was billed for
    # 250 lines. That is roughly double the tokens per read, and the extra
    # lines widen every subsequent range enough to trip the redundant-read
    # refusal, which costs a step on top of the tokens.
    default_read_lines: int = 120
    max_tool_output_chars: int = 12_000
    tool_timeout_seconds: int = 20

    # Files
    max_file_bytes: int = 2_000_000
    max_viewable_file_bytes: int = 500_000

    # Repository
    clone_timeout_seconds: int = 180
    max_repo_bytes: int = 500 * 1024 * 1024

    # Conversation history
    max_history_messages: int = 20
    max_history_chars: int = 40_000

    # Subagents (Phase 7)
    max_subagents: int = field(default_factory=lambda: _int("MAX_SUBAGENTS", 4))
    subagent_token_budget: int = 40_000
    subagent_max_steps: int = 12

    # Sandbox (Phase 6)
    sandbox_timeout_seconds: int = field(
        default_factory=lambda: _int("SANDBOX_TIMEOUT_SECONDS", 300)
    )
    sandbox_memory_mb: int = 1024
    sandbox_cpus: float = 2.0
    sandbox_max_output_chars: int = 20_000

    # HTTP / streaming
    http_timeout_seconds: int = 15
    stream_timeout_seconds: int = 600


LIMITS = Limits()


# --------------------------------------------------------------------------
# Workspace location
# --------------------------------------------------------------------------
# Cloud-sync clients fight git over `.git` internals. They open handles on files
# the moment they appear, and a clone creates hundreds of small files fast, so
# git's attempt to lock `.git/config` fails with "could not lock config file:
# No such file or directory" — intermittently, which makes it look like a bug in
# this application rather than in the folder it was told to write to.
SYNC_CLIENT_MARKERS: tuple[str, ...] = ("onedrive", "dropbox", "google drive", "icloud")


def synced_folder(path: Path) -> str | None:
    """Name of the sync client whose folder ``path`` sits inside, if any."""
    lowered = str(path).lower()
    for marker in SYNC_CLIENT_MARKERS:
        if marker in lowered:
            return marker.title().replace(" ", "")
    return None


def _default_workspace_root() -> Path:
    """Where cloned repositories live.

    Deliberately *not* inside the project directory. On Windows a project often
    sits under OneDrive, which breaks clones as described above — and cloned
    repositories are large, disposable caches, so uploading them to someone's
    cloud storage quota is a second bug on top of the first.

    Set WORKSPACE_ROOT to override; a relative value is resolved against the
    project directory.
    """
    override = _str("WORKSPACE_ROOT")
    if override:
        candidate = Path(override)
        return candidate.resolve() if candidate.is_absolute() else (BASE_DIR / candidate).resolve()

    if os.name == "nt":
        base = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
    else:
        base = os.getenv("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return Path(base, "ai-coding-workspace", "workspaces").resolve()


# --------------------------------------------------------------------------
# Application settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    secret_key: str = field(
        default_factory=lambda: _str("FLASK_SECRET_KEY", "dev-insecure-key")
    )
    log_level: str = field(default_factory=lambda: _str("LOG_LEVEL", "INFO").upper())
    workspace_root: Path = field(default_factory=_default_workspace_root)

    conversation_store: str = field(
        default_factory=lambda: _str("CONVERSATION_STORE", "sqlite").lower()
    )
    database_url: str = field(
        default_factory=lambda: _str("DATABASE_URL", "sqlite:///instance/app.db")
    )

    sandbox_mode: str = field(default_factory=lambda: _str("SANDBOX_MODE", "auto").lower())
    sandbox_image: str = field(
        default_factory=lambda: _str("SANDBOX_IMAGE", "python:3.12-slim")
    )

    auth_enabled: bool = field(default_factory=lambda: _bool("AUTH_ENABLED", False))
    rate_limit_per_minute: int = field(
        default_factory=lambda: _int("RATE_LIMIT_PER_MINUTE", 30)
    )
    daily_cost_limit_usd: float = field(
        default_factory=lambda: _float("DAILY_COST_LIMIT_USD", 10.0)
    )

    @property
    def model_configured(self) -> bool:
        return bool(OPENAI_API_KEY)


SETTINGS = Settings()
