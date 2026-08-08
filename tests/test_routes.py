"""Flask route tests: validation, lifecycle, history, SSE, and the approval gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app_state
from services.storage.base import Message
from tests.fake_model import FakeModel, install, tool_call, turn


class TestRepoRoutes:
    def test_status_reports_nothing_loaded(self, client):
        response = client.get("/api/repo/status?thread_id=t1")
        assert response.status_code == 200
        assert response.get_json()["loaded"] is False

    def test_status_requires_a_thread_id(self, client):
        assert client.get("/api/repo/status").status_code == 400

    @pytest.mark.parametrize(
        "url",
        ["not-a-url", "https://gitlab.com/owner/repo", "git@github.com:o/r.git",
         "https://github.com/", "https://evil.test/github.com/o/r"],
    )
    def test_rejects_bad_repository_urls(self, client, url):
        response = client.post("/api/repo/load", json={"thread_id": "t1", "url": url})
        assert response.status_code == 400

    def test_tree_lists_safe_entries_only(self, client, loaded_workspace):
        response = client.get("/api/repo/tree?thread_id=thread1")
        names = [entry["name"] for entry in response.get_json()["entries"]]
        assert "app.py" in names
        assert ".env" not in names
        assert ".git" not in names

    def test_file_viewer_returns_content(self, client, loaded_workspace):
        response = client.get("/api/repo/file?thread_id=thread1&path=config.py")
        assert response.status_code == 200
        assert "load_settings" in response.get_json()["content"]

    def test_file_viewer_refuses_secrets(self, client, loaded_workspace):
        response = client.get("/api/repo/file?thread_id=thread1&path=.env")
        assert response.status_code == 403

    def test_file_viewer_refuses_traversal(self, client, loaded_workspace):
        response = client.get("/api/repo/file?thread_id=thread1&path=../../../etc/passwd")
        assert response.status_code == 403

    def test_file_viewer_refuses_binaries(self, client, loaded_workspace):
        response = client.get("/api/repo/file?thread_id=thread1&path=assets/logo.png")
        assert response.status_code == 400

    def test_branches_are_listed(self, client, loaded_workspace):
        data = client.get("/api/repo/branches?thread_id=thread1").get_json()
        assert data["current"] == "main"

    def test_rejects_unknown_branch(self, client, loaded_workspace):
        response = client.post(
            "/api/repo/branch", json={"thread_id": "thread1", "branch": "no-such-branch"}
        )
        assert response.status_code == 400

    @pytest.mark.parametrize("branch", ["--upload-pack=evil", "a;rm -rf /", "../escape"])
    def test_rejects_malicious_branch_names(self, client, loaded_workspace, branch):
        response = client.post(
            "/api/repo/branch", json={"thread_id": "thread1", "branch": branch}
        )
        assert response.status_code == 400


class TestThreadIsolation:
    def test_one_thread_cannot_reach_another_workspace(self, client, loaded_workspace):
        """thread1 has a repo; thread2 must not be able to read through it."""
        assert client.get("/api/repo/file?thread_id=thread1&path=app.py").status_code == 200
        response = client.get("/api/repo/file?thread_id=thread2&path=app.py")
        assert response.status_code == 404

    def test_workspace_directories_are_hashed_per_thread(self, workspace_service):
        first = workspace_service._thread_dir("thread-a")
        second = workspace_service._thread_dir("thread-b")
        assert first != second
        assert first.name.startswith("ws_")
        assert "thread-a" not in first.name   # never client-controlled text

    @pytest.mark.parametrize("thread_id", ["../escape", "a/b", "x" * 100, "", "a;b"])
    def test_rejects_malformed_thread_ids(self, workspace_service, thread_id):
        from core.errors import ValidationError

        with pytest.raises(ValidationError):
            workspace_service._thread_dir(thread_id)

    def test_unload_refuses_paths_outside_the_root(self, workspace_service, tmp_path):
        from core.errors import SafetyError

        outside = tmp_path / "important-data"
        outside.mkdir()
        (outside / "file.txt").write_text("do not delete", encoding="utf-8")

        with pytest.raises(SafetyError):
            workspace_service._remove_tree(outside)
        assert (outside / "file.txt").exists()

    def test_unload_refuses_the_workspace_root(self, workspace_service):
        from core.errors import SafetyError

        with pytest.raises(SafetyError):
            workspace_service._remove_tree(workspace_service.root)


class TestHistoryRoutes:
    def test_returns_stored_turns(self, client, store):
        store.append(Message(thread_id="t1", role="user", content="hello"))
        store.append(Message(thread_id="t1", role="assistant", content="hi"))
        messages = client.get("/api/history?thread_id=t1").get_json()["messages"]
        assert len(messages) == 2

    def test_requires_a_thread_id(self, client):
        assert client.get("/api/history").status_code == 400

    def test_unknown_thread_is_empty_not_an_error(self, client):
        response = client.get("/api/history?thread_id=nope")
        assert response.status_code == 200
        assert response.get_json()["messages"] == []

    def test_lists_threads(self, client, store):
        store.append(Message(thread_id="t1", role="user", content="first"))
        threads = client.get("/api/history/threads").get_json()["threads"]
        assert threads[0]["thread_id"] == "t1"

    def test_delete_removes_history(self, client, store):
        store.append(Message(thread_id="t1", role="user", content="x"))
        assert client.delete("/api/history?thread_id=t1").get_json()["deleted"] is True
        assert store.history("t1") == []


class TestChatRoute:
    def test_requires_a_message(self, client):
        assert client.post("/api/chat", json={"thread_id": "t1"}).status_code == 400

    def test_rejects_an_oversized_message(self, client):
        response = client.post("/api/chat", json={"thread_id": "t1", "message": "x" * 20_000})
        assert response.status_code == 400

    def test_rejects_an_unknown_mode(self, client, loaded_workspace):
        response = client.post(
            "/api/chat", json={"thread_id": "thread1", "message": "hi", "mode": "delete"}
        )
        assert response.status_code == 400

    def test_requires_a_loaded_repository(self, client):
        response = client.post("/api/chat", json={"thread_id": "t1", "message": "hello"})
        assert response.status_code == 404

    def test_code_mode_requires_a_task_branch(self, client, loaded_workspace):
        """Editing without an approved task branch is refused before any spend."""
        response = client.post(
            "/api/chat",
            json={"thread_id": "thread1", "message": "fix the bug", "mode": "code"},
        )
        assert response.status_code == 403

    def test_streams_a_complete_run(self, client, loaded_workspace, monkeypatch):
        install(monkeypatch, FakeModel([
            turn(calls=[tool_call("read", path="app.py", offset=1, limit=20)]),
            turn(text="The entry point is app.py:6."),
        ]))

        response = client.post(
            "/api/chat", json={"thread_id": "thread1", "message": "Where is the entry point?"}
        )
        assert response.status_code == 200
        assert response.mimetype == "text/event-stream"

        body = response.get_data(as_text=True)
        assert "event: start" in body
        assert "event: tool_end" in body
        assert "event: done" in body
        assert "event: end" in body

    def test_persists_bounded_turns_only(self, client, loaded_workspace, store, monkeypatch):
        """System prompts, raw tool output, and evidence must not be stored."""
        install(monkeypatch, FakeModel([
            turn(calls=[tool_call("read", path="app.py", offset=1, limit=20)]),
            turn(text="Entry point is app.py:6."),
        ]))
        # Persistence happens on the agent thread just before the stream's
        # sentinel, so the body must be drained before asserting on the store.
        response = client.post("/api/chat", json={"thread_id": "thread1", "message": "Where is it?"})
        response.get_data(as_text=True)

        history = store.history("thread1")
        assert [m.role for m in history] == ["user", "assistant"]
        joined = " ".join(m.content for m in history)
        assert "repository investigator" not in joined   # no system prompt
        assert "1 | import os" not in joined             # no raw tool output

    def test_records_usage_for_billing(self, client, loaded_workspace, store, monkeypatch):
        install(monkeypatch, FakeModel([turn(text="Answer.", input_tokens=500, output_tokens=50)]))
        response = client.post("/api/chat", json={"thread_id": "thread1", "message": "Question?"})
        response.get_data(as_text=True)

        records = store.usage_since(user_id="local", since=0)
        assert len(records) == 1
        assert records[0].input_tokens > 0

    def test_config_endpoint_reports_the_environment(self, client):
        data = client.get("/api/chat/config").get_json()
        assert "model" in data and "sandbox_backend" in data
        assert data["max_steps"] > 0


class TestHealthRoutes:
    def test_liveness_is_always_ok(self, client):
        assert client.get("/healthz").get_json()["status"] == "ok"

    def test_readiness_reports_each_dependency(self, client):
        data = client.get("/readyz").get_json()
        assert "checks" in data
        assert data["checks"]["git"] is True
        assert "sandbox" in data["checks"]


class TestErrorHandling:
    def test_unknown_route_returns_json(self, client):
        response = client.get("/api/nope")
        assert response.status_code == 404
        assert response.get_json()["code"] == "not_found"

    def test_index_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"AI Coding Workspace" in response.data

    def test_responses_carry_a_request_id(self, client):
        assert client.get("/healthz").headers.get("X-Request-ID")
