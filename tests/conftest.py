"""Shared fixtures.

Everything here is hermetic: no network, no model calls, no Docker, and no
writes outside pytest's tmp_path. The suite must be runnable offline and cost
nothing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app_state  # noqa: E402
from services.storage.memory_store import InMemoryConversationStore  # noqa: E402
from services.workspace_service import Workspace, WorkspaceService  # noqa: E402
from services.github_service import RepoRef  # noqa: E402


SAMPLE_FILES: dict[str, str] = {
    "app.py": (
        "import os\n"
        "from config import load_settings\n"
        "from handlers import handle_request\n"
        "\n"
        "\n"
        "def create_app():\n"
        "    settings = load_settings()\n"
        "    return Application(settings)\n"
        "\n"
        "\n"
        "class Application:\n"
        "    def __init__(self, settings):\n"
        "        self.settings = settings\n"
        "\n"
        "    def dispatch(self, request):\n"
        "        return handle_request(request, self.settings)\n"
    ),
    "config.py": (
        "import os\n"
        "\n"
        "\n"
        "def load_settings():\n"
        "    secret = os.environ.get('SECRET_KEY')\n"
        "    if not secret:\n"
        "        raise RuntimeError('SECRET_KEY is required')\n"
        "    return {'secret': secret}\n"
    ),
    "handlers.py": (
        "def handle_request(request, settings):\n"
        "    validated = validate(request)\n"
        "    return respond(validated, settings)\n"
        "\n"
        "\n"
        "def validate(request):\n"
        "    if not request.get('path'):\n"
        "        raise ValueError('path is required')\n"
        "    return request\n"
        "\n"
        "\n"
        "def respond(request, settings):\n"
        "    return {'status': 200, 'path': request['path']}\n"
    ),
    "README.md": "# Sample\n\nA fixture repository used by the test suite.\n",
    "src/utils/helpers.py": "def slugify(value):\n    return value.lower().replace(' ', '-')\n",
    ".env": "OPENAI_API_KEY=sk-should-never-be-read-1234567890abcdef\n",
    "secrets.yaml": "password: hunter2hunter2hunter2\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A small real Git repository on disk."""
    root = tmp_path / "sample-repo"
    root.mkdir()

    for relative, content in SAMPLE_FILES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    (root / "assets").mkdir()
    (root / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00binary")

    env = {"GIT_CONFIG_NOSYSTEM": "1", "PATH": __import__("os").environ.get("PATH", "")}
    run = lambda *args: subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, env=env, shell=False
    )
    run("init", "-q", "-b", "main")
    run("config", "user.email", "test@example.com")
    run("config", "user.name", "Test")
    run("add", "-A")
    run("commit", "-q", "-m", "initial commit")
    return root


@pytest.fixture
def store() -> InMemoryConversationStore:
    return InMemoryConversationStore()


@pytest.fixture
def workspace_service(tmp_path: Path) -> WorkspaceService:
    return WorkspaceService(tmp_path / "workspaces")


@pytest.fixture
def loaded_workspace(repo: Path, workspace_service: WorkspaceService) -> Workspace:
    """Register the fixture repo as a thread's workspace without cloning."""
    workspace = Workspace(
        thread_id="thread1",
        repo=RepoRef("acme", "sample-repo"),
        path=repo,
        default_branch="main",
        branch="main",
        head_sha="abc123def456",
    )
    workspace_service._workspaces["thread1"] = workspace
    return workspace


@pytest.fixture
def client(store, workspace_service, monkeypatch):
    """Flask test client wired to hermetic singletons."""
    from app import create_app
    from services import quota_service

    monkeypatch.setattr(app_state, "conversations", store)
    monkeypatch.setattr(app_state, "workspaces", workspace_service)
    quota_service.reset_for_tests()

    application = create_app(TESTING=True)
    with application.test_client() as test_client:
        yield test_client
