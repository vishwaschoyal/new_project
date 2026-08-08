"""Security tests: traversal, symlinks, secret paths, redaction, binaries."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from core.errors import SafetyError
from core.safety import (
    is_binary_file,
    is_blocked_directory,
    is_secret_path,
    redact_secrets,
    relative_to_workspace,
    resolve_in_workspace,
)


class TestPathConfinement:
    @pytest.mark.parametrize(
        "attempt",
        [
            "../outside.txt",
            "../../etc/passwd",
            "src/../../escape.py",
            "src/./../../escape.py",
            "/etc/passwd",
            "C:/Windows/System32/drivers/etc/hosts",
            "\\\\server\\share\\file.txt",
            "//server/share/file.txt",
        ],
    )
    def test_rejects_escapes(self, repo: Path, attempt: str):
        with pytest.raises(SafetyError):
            resolve_in_workspace(repo, attempt)

    def test_rejects_empty_and_null(self, repo: Path):
        for attempt in ("", "   ", ".", "file\x00.py"):
            with pytest.raises(SafetyError):
                resolve_in_workspace(repo, attempt)

    def test_accepts_paths_inside(self, repo: Path):
        resolved = resolve_in_workspace(repo, "src/utils/helpers.py")
        assert resolved.is_file()
        assert relative_to_workspace(repo, resolved) == "src/utils/helpers.py"

    def test_normalises_backslashes(self, repo: Path):
        resolved = resolve_in_workspace(repo, "src\\utils\\helpers.py")
        assert resolved.name == "helpers.py"

    @pytest.mark.skipif(
        sys.platform == "win32", reason="symlink creation needs elevation on Windows"
    )
    def test_rejects_symlink_escape(self, repo: Path, tmp_path: Path):
        """A symlink is resolved *before* confinement is checked, so a link
        pointing outside the workspace is caught rather than followed."""
        outside = tmp_path / "outside-secret.txt"
        outside.write_text("sensitive", encoding="utf-8")
        (repo / "link.txt").symlink_to(outside)

        with pytest.raises(SafetyError, match="escapes the workspace"):
            resolve_in_workspace(repo, "link.txt")

    def test_rejects_blocked_directories(self, repo: Path):
        with pytest.raises(SafetyError, match="blocked directory"):
            resolve_in_workspace(repo, ".git/config")


class TestSecretPaths:
    @pytest.mark.parametrize(
        "path",
        [".env", ".env.production", "config/.env.local", "keys/server.pem",
         "id_rsa", "app.key", "secrets.yaml", "deploy/credentials",
         ".npmrc", ".netrc", "service-account-prod.json"],
    )
    def test_flags_secret_paths(self, path: str):
        assert is_secret_path(path) is True

    @pytest.mark.parametrize(
        "path", [".env.example", ".env.sample", "app.py", "README.md", "src/keys.py"]
    )
    def test_allows_ordinary_paths(self, path: str):
        assert is_secret_path(path) is False

    def test_secret_file_is_refused_even_though_it_exists(self, repo: Path):
        assert (repo / ".env").is_file()
        with pytest.raises(SafetyError, match="secret file"):
            resolve_in_workspace(repo, ".env")

    def test_blocked_directory_detection(self):
        assert is_blocked_directory("node_modules/pkg/index.js")
        assert is_blocked_directory(".git/objects/ab/cdef")
        assert not is_blocked_directory("src/app.py")


class TestRedaction:
    @pytest.mark.parametrize(
        "secret",
        [
            "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
            "sk-abcdefghijklmnopqrstuvwxyz1234",
            "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
            "github_pat_11ABCDEFG0abcdefghijklmnop",
            "AKIAIOSFODNN7EXAMPLE",
            "xoxb-1234567890-abcdefghij",
        ],
    )
    def test_removes_credentials(self, secret: str):
        redacted = redact_secrets(f"the key is {secret} ok")
        assert secret not in redacted
        assert "redacted" in redacted

    def test_removes_assignments(self):
        redacted = redact_secrets('api_key = "abcdefghij1234567890"')
        assert "abcdefghij1234567890" not in redacted

    def test_removes_private_key_blocks(self):
        block = "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----"
        assert "MIIabc" not in redact_secrets(block)

    def test_leaves_ordinary_text_alone(self):
        text = "def handle_request(request):\n    return respond(request)"
        assert redact_secrets(text) == text

    def test_handles_empty(self):
        assert redact_secrets("") == ""


class TestBinaryDetection:
    def test_detects_binary(self, repo: Path):
        assert is_binary_file(repo / "assets" / "logo.png") is True

    def test_accepts_text(self, repo: Path):
        assert is_binary_file(repo / "app.py") is False

    def test_empty_file_is_text(self, tmp_path: Path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        assert is_binary_file(empty) is False
