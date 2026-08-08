"""Multi-user isolation and private-repository access.

Two regressions live here.

**Authorisation.** A thread ID is client-supplied. Only the task routes checked
who owned a thread, so with AUTH_ENABLED=true one user could pass another's
thread ID to the repository, chat, or history routes and read their cloned
source — private source included.

**Private repositories.** `/users/{username}/repos` returns public repositories
only, even when authenticated, so the token owner could not see their own
private work. `/user/repos` is the endpoint that includes it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import app_state
from config import SETTINGS
from core.errors import ForbiddenError
from services import github_service, quota_service
from services.github_service import RepoRef
from services.storage.base import Message
from services.workspace_service import Workspace


@pytest.fixture
def auth_on(monkeypatch):
    """Turn on authentication for the modules that read the setting."""
    enabled = dataclasses.replace(SETTINGS, auth_enabled=True)
    monkeypatch.setattr(quota_service, "SETTINGS", enabled)
    quota_service.reset_for_tests()
    return enabled


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


ALICE = "alice-key-01234567890123456789"
BOB = "bob-key-01234567890123456789"


class TestSingleUserMode:
    """The default. Everyone is 'local' — convenient, and not multi-tenant."""

    def test_everyone_shares_one_identity(self, client):
        assert quota_service.LOCAL_USER_ID == "local"
        with client.application.test_request_context():
            assert quota_service.current_user_id() == "local"

    def test_no_key_is_required(self, client, loaded_workspace):
        assert client.get("/api/repo/status?thread_id=thread1").status_code == 200


class TestWorkspaceOwnership:
    def test_owner_may_use_their_workspace(self, workspace_service, repo: Path):
        workspace_service._workspaces["t1"] = Workspace(
            thread_id="t1", repo=RepoRef("acme", "r"), path=repo,
            default_branch="main", branch="main", head_sha="abc", owner_id="alice",
        )
        assert workspace_service.require("t1", user_id="alice").owner_id == "alice"

    def test_another_user_is_refused(self, workspace_service, repo: Path):
        workspace_service._workspaces["t1"] = Workspace(
            thread_id="t1", repo=RepoRef("acme", "r"), path=repo,
            default_branch="main", branch="main", head_sha="abc", owner_id="alice",
        )
        with pytest.raises(ForbiddenError, match="another user"):
            workspace_service.require("t1", user_id="bob")

    def test_unclaimed_workspaces_stay_open(self, workspace_service, repo: Path):
        """Single-user mode records no owner; that must not lock anyone out."""
        workspace_service._workspaces["t1"] = Workspace(
            thread_id="t1", repo=RepoRef("acme", "r"), path=repo,
            default_branch="main", branch="main", head_sha="abc",
        )
        assert workspace_service.require("t1", user_id="anyone") is not None


class TestCrossUserAccess:
    """With auth on, Bob must not reach Alice's thread through any route."""

    @pytest.fixture
    def alice_thread(self, client, store, workspace_service, repo: Path, auth_on):
        with client.application.test_request_context(headers=_headers(ALICE)):
            alice_id = quota_service.current_user_id()

        store.append(Message(thread_id="shared-id", role="user", content="alice's question",
                             metadata={"user_id": alice_id}))
        workspace_service._workspaces["shared-id"] = Workspace(
            thread_id="shared-id", repo=RepoRef("alice", "private-repo"), path=repo,
            default_branch="main", branch="main", head_sha="abc", owner_id=alice_id,
        )
        return alice_id

    def test_repository_file_is_refused(self, client, alice_thread):
        """The sharpest one: this would return Alice's private source."""
        response = client.get(
            "/api/repo/file?thread_id=shared-id&path=config.py", headers=_headers(BOB)
        )
        assert response.status_code == 403

    def test_repository_tree_is_refused(self, client, alice_thread):
        response = client.get("/api/repo/tree?thread_id=shared-id", headers=_headers(BOB))
        assert response.status_code == 403

    def test_history_is_refused(self, client, alice_thread):
        response = client.get("/api/history?thread_id=shared-id", headers=_headers(BOB))
        assert response.status_code == 403

    def test_deleting_the_thread_is_refused(self, client, alice_thread):
        response = client.delete("/api/history?thread_id=shared-id", headers=_headers(BOB))
        assert response.status_code == 403

    def test_running_the_agent_is_refused(self, client, alice_thread):
        response = client.post(
            "/api/chat",
            json={"thread_id": "shared-id", "message": "what is in this repo?"},
            headers=_headers(BOB),
        )
        assert response.status_code == 403

    def test_starting_a_task_is_refused(self, client, alice_thread):
        response = client.post(
            "/api/task/start",
            json={"thread_id": "shared-id", "description": "change something"},
            headers=_headers(BOB),
        )
        assert response.status_code == 403

    def test_the_owner_still_has_access(self, client, alice_thread):
        response = client.get(
            "/api/repo/file?thread_id=shared-id&path=config.py", headers=_headers(ALICE)
        )
        assert response.status_code == 200

    def test_a_missing_key_is_rejected(self, client, alice_thread):
        response = client.get("/api/repo/tree?thread_id=shared-id")
        assert response.status_code == 401

    def test_user_ids_are_hashes_not_keys(self, client, auth_on):
        """The ID reaches logs, usage records, and directory names."""
        with client.application.test_request_context(headers=_headers(ALICE)):
            user_id = quota_service.current_user_id()
        assert user_id.startswith("u_")
        assert ALICE not in user_id


class TestPrivateRepositories:
    @pytest.fixture(autouse=True)
    def clear_cache(self, monkeypatch):
        monkeypatch.setattr(github_service, "_authenticated_login", None)

    def test_own_account_uses_the_endpoint_that_includes_private(self, monkeypatch):
        calls: list[tuple[str, dict]] = []

        def fake_get(path, **params):
            calls.append((path, params))
            if path == "/user":
                return {"login": "vishwaschoyal"}
            return [{"name": "secret", "full_name": "vishwaschoyal/secret", "private": True}]

        monkeypatch.setattr(github_service, "_get", fake_get)
        monkeypatch.setattr(github_service, "GITHUB_TOKEN", "ghp_token", raising=False)

        repos = github_service.fetch_repositories("vishwaschoyal")

        assert any(path == "/user/repos" for path, _ in calls)
        assert not any(path.startswith("/users/") for path, _ in calls)
        assert repos[0]["private"] is True

    def test_other_accounts_use_the_public_endpoint(self, monkeypatch):
        """`/user/repos` only ever describes the token owner."""
        calls: list[str] = []

        def fake_get(path, **params):
            calls.append(path)
            if path == "/user":
                return {"login": "vishwaschoyal"}
            return []

        monkeypatch.setattr(github_service, "_get", fake_get)
        monkeypatch.setattr(github_service, "GITHUB_TOKEN", "ghp_token", raising=False)

        github_service.fetch_repositories("torvalds")
        assert "/users/torvalds/repos" in calls
        assert "/user/repos" not in calls

    def test_without_a_token_the_public_endpoint_is_used(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(github_service, "GITHUB_TOKEN", "", raising=False)
        monkeypatch.setattr(
            github_service, "_get", lambda path, **p: (calls.append(path), [])[1]
        )
        github_service.fetch_repositories("anyone")
        assert calls == ["/users/anyone/repos"]

    def test_private_flag_reaches_the_browser(self, monkeypatch):
        monkeypatch.setattr(github_service, "GITHUB_TOKEN", "", raising=False)
        monkeypatch.setattr(
            github_service, "_get",
            lambda path, **p: [{"name": "r", "full_name": "o/r", "private": True}],
        )
        assert github_service.fetch_repositories("o")[0]["private"] is True

    def test_ui_marks_private_repositories(self):
        app_js = (Path(__file__).resolve().parents[1] / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        assert "repo-private" in app_js
        assert "repo.private" in app_js
