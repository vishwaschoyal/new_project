"""Where clones live, and surviving a filesystem that fights back.

Regression: the default workspace root sat next to the project, which on this
machine put it inside OneDrive. A sync client opens handles on files as git
creates them, so clones failed intermittently with "could not lock config file:
No such file or directory" — a failure that looks like an application bug and
is not one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from config import synced_folder
from services.workspace_service import (
    WorkspaceService,
    _clone_error_message,
    _looks_transient,
)


class TestSyncedFolderDetection:
    @pytest.mark.parametrize(
        "path,expected",
        [
            (r"C:\Users\me\OneDrive\Desktop\project\workspaces", "Onedrive"),
            (r"C:\Users\me\Dropbox\code\workspaces", "Dropbox"),
            ("/Users/me/Google Drive/work/workspaces", "GoogleDrive"),
            ("/Users/me/Library/Mobile Documents/iCloud~folder", "Icloud"),
        ],
    )
    def test_detects_sync_clients(self, path, expected):
        assert synced_folder(Path(path)) == expected

    @pytest.mark.parametrize(
        "path",
        [r"C:\Users\me\AppData\Local\ai-coding-workspace\workspaces",
         "/home/me/.local/share/ai-coding-workspace/workspaces",
         "/var/lib/acw/workspaces"],
    )
    def test_ordinary_paths_are_clean(self, path):
        assert synced_folder(Path(path)) is None

    def test_default_root_is_not_beside_the_project(self):
        """The project directory is exactly where a sync client tends to be."""
        from config import BASE_DIR, _default_workspace_root

        root = _default_workspace_root()
        assert BASE_DIR not in root.parents
        assert synced_folder(root) is None

    def test_explicit_override_is_honoured(self, monkeypatch, tmp_path):
        import config

        monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "custom"))
        assert config._default_workspace_root() == (tmp_path / "custom").resolve()

    def test_relative_override_resolves_against_the_project(self, monkeypatch):
        import config

        monkeypatch.setenv("WORKSPACE_ROOT", "my-clones")
        assert config._default_workspace_root() == (config.BASE_DIR / "my-clones").resolve()


class TestTransientErrorClassification:
    @pytest.mark.parametrize(
        "error",
        [
            "error: could not lock config file .git/config: No such file or directory",
            "fatal: could not create work tree dir: Permission denied",
            "error: unable to create file src/x.py: Access is denied",
        ],
    )
    def test_filesystem_contention_is_retryable(self, error):
        assert _looks_transient(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            "fatal: Authentication failed for 'https://github.com/o/r.git/'",
            "fatal: repository not found",
            "fatal: could not resolve host: github.com",
        ],
    )
    def test_auth_and_network_failures_are_not_retried(self, error):
        """Retrying these wastes the user's time and never succeeds."""
        assert _looks_transient(error) is False

    def test_empty_error_is_not_transient(self):
        assert _looks_transient("") is False

    def test_message_names_the_sync_client_and_the_fix(self):
        message = _clone_error_message(
            "error: could not lock config file .git/config: No such file or directory",
            Path(r"C:\Users\me\OneDrive\project\workspaces"),
        )
        assert "Onedrive" in message
        assert "WORKSPACE_ROOT" in message

    def test_message_stays_plain_outside_a_synced_folder(self):
        message = _clone_error_message(
            "fatal: repository not found", Path(r"C:\data\workspaces")
        )
        assert "WORKSPACE_ROOT" not in message


class TestSafeRemoval:
    def test_rename_before_delete_frees_the_name_immediately(self, tmp_path):
        """Windows keeps a deleted directory name reserved while a handle is
        open. Renaming first means the path is reusable at once."""
        service = WorkspaceService(tmp_path / "ws")
        target = service.root / "ws_abc123"
        (target / "repo").mkdir(parents=True)
        (target / "repo" / "file.txt").write_text("x", encoding="utf-8")

        service._remove_tree(target)

        assert not target.exists()
        # The name is immediately reusable, which is the whole point.
        target.mkdir()
        (target / "recreated.txt").write_text("y", encoding="utf-8")
        assert (target / "recreated.txt").exists()

    def test_leaves_a_directory_rather_than_failing_when_delete_is_blocked(
        self, tmp_path, monkeypatch
    ):
        """If the delete cannot finish, the clone must still proceed."""
        service = WorkspaceService(tmp_path / "ws")
        target = service.root / "ws_blocked"
        target.mkdir(parents=True)

        def refuse(*_args, **kwargs):
            onerror = kwargs.get("onerror")
            if onerror:
                onerror(lambda _p: None, str(target), None)

        monkeypatch.setattr(shutil, "rmtree", refuse)

        service._remove_tree(target)          # must not raise
        assert not target.exists()            # renamed out of the way regardless

    def test_still_refuses_paths_outside_the_root(self, tmp_path):
        """The rename must not weaken the deletion guard."""
        from core.errors import SafetyError

        service = WorkspaceService(tmp_path / "ws")
        outside = tmp_path / "precious"
        outside.mkdir()

        with pytest.raises(SafetyError):
            service._remove_tree(outside)
        assert outside.exists()
