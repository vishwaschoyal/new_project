"""Architecture constraints enforced as tests.

The model-client rule is a project invariant, not a style preference: the
application owns the loop, and `ChatOpenAI` is the only sanctioned transport. A
direct OpenAI SDK client anywhere in application code would route around the
budget accounting, the usage metadata extraction, and the prompt-cache
configuration. So it is checked mechanically rather than by review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

APPLICATION_PACKAGES = ("agents", "routes", "services", "tools", "core")
APPLICATION_MODULES = ("app.py", "app_state.py", "config.py")

FORBIDDEN_CLIENTS = {"OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"}


def application_files() -> list[Path]:
    files = [ROOT / name for name in APPLICATION_MODULES]
    for package in APPLICATION_PACKAGES:
        files.extend(sorted((ROOT / package).rglob("*.py")))
    return [path for path in files if path.exists()]


def test_application_files_are_discovered():
    """Guard the guard: a broken glob would make every check below vacuous."""
    files = application_files()
    assert len(files) > 15
    assert any(path.name == "orchestrator.py" for path in files)


@pytest.mark.parametrize("path", application_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_direct_openai_sdk_import(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("openai"), (
                    f"{path.relative_to(ROOT)} imports {alias.name!r}. "
                    "Application code must use langchain_openai.ChatOpenAI."
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "openai" or module.startswith("openai."):
                pytest.fail(
                    f"{path.relative_to(ROOT)} imports from {module!r}. "
                    "Application code must use langchain_openai.ChatOpenAI."
                )


@pytest.mark.parametrize("path", application_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_openai_client_construction(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else ""
        )
        assert name not in FORBIDDEN_CLIENTS, (
            f"{path.relative_to(ROOT)} constructs {name}(). "
            "Use config.create_read_loop_model() instead."
        )


def test_model_client_is_created_in_exactly_one_place():
    """One factory means prompt-cache settings and budgets cannot drift."""
    constructing = [
        path for path in application_files()
        if "ChatOpenAI(" in path.read_text(encoding="utf-8")
    ]
    assert [p.name for p in constructing] == ["config.py"]


def test_prompt_cache_is_configured():
    from config import create_read_loop_model

    import inspect
    source = inspect.getsource(create_read_loop_model)
    assert "prompt_cache_key" in source
    assert "stream_usage=True" in source


def test_tools_never_import_the_orchestrator():
    """The dependency runs one way: orchestrator -> tools. A cycle here would
    mean tools could reach into loop state."""
    for path in sorted((ROOT / "tools").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "from agents" not in source, f"{path.name} imports from agents"
        assert "import agents" not in source, f"{path.name} imports agents"


def test_no_secrets_committed():
    """The real .env must never become a tracked file."""
    import subprocess

    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, shell=False
    )
    tracked = set(result.stdout.split())
    assert ".env" not in tracked
    assert not any(name.endswith((".pem", ".key")) for name in tracked)
