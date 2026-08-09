"""Tool tests: bounds, stamping, determinism, and the bash allow-list."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import create_tools, edit_tools, read_tools
from tools.registry import (
    Capability,
    EDITING_CAPABILITIES,
    READ_ONLY_CAPABILITIES,
    ToolRegistry,
)


class TestGrep:
    def test_returns_stamped_matches(self, repo: Path):
        result = read_tools.grep(repo, pattern="def handle_request")
        assert result.ok
        assert "handlers.py:1:" in result.content

    def test_reports_no_matches_as_success(self, repo: Path):
        """No matches is an answer, not a failure — the model must be able to
        conclude 'this does not exist here'."""
        result = read_tools.grep(repo, pattern="def nonexistent_symbol_xyz")
        assert result.ok
        assert result.metadata["matches"] == 0

    def test_rejects_invalid_regex(self, repo: Path):
        result = read_tools.grep(repo, pattern="def (unclosed")
        assert not result.ok
        assert "regular expression" in result.content

    def test_requires_a_pattern(self, repo: Path):
        assert not read_tools.grep(repo, pattern="  ").ok

    def test_never_returns_secret_files(self, repo: Path):
        result = read_tools.grep(repo, pattern="OPENAI_API_KEY|password")
        assert ".env" not in result.content
        assert "secrets.yaml" not in result.content

    def test_bounds_result_count(self, repo: Path):
        for index in range(80):
            (repo / f"gen_{index}.py").write_text("MARKER_TOKEN = 1\n", encoding="utf-8")
        result = read_tools.grep(repo, pattern="MARKER_TOKEN", max_results=10)
        assert result.ok
        assert len(result.content.splitlines()) <= 10
        assert result.metadata["truncated"] is True

    def test_restricts_to_a_subdirectory(self, repo: Path):
        result = read_tools.grep(repo, pattern="def", path="src")
        assert result.ok
        assert "handlers.py" not in result.content

    def test_rejects_traversal_in_path(self, repo: Path):
        result = read_tools.grep(repo, pattern="def", path="../..")
        assert not result.ok

    def test_scoping_to_a_single_file_works(self, repo: Path):
        """Regression: a file as the scope became the subprocess working
        directory, and the model got `NotADirectoryError: [WinError 267]` — a
        message it cannot act on, so it burned a step and guessed again."""
        result = read_tools.grep(repo, pattern="def", path="handlers.py")
        assert result.ok
        assert "handlers.py:1:" in result.content
        assert "app.py" not in result.content

    def test_a_single_file_scope_is_still_path_stamped(self, repo: Path):
        """ripgrep omits the filename when given one file; without forcing it
        back on, the line number is parsed as the path."""
        result = read_tools.grep(repo, pattern="def validate", path="handlers.py")
        assert result.ok
        assert result.content.startswith("handlers.py:6:")

    @pytest.mark.parametrize("scope", [".", "./", "", "  ", "/"])
    def test_treats_dot_as_the_repository_root(self, repo: Path, scope: str):
        """Models naturally send path='.' for 'search everywhere'. Refusing it
        burns a step and teaches nothing."""
        result = read_tools.grep(repo, pattern="def handle_request", path=scope)
        assert result.ok
        assert "handlers.py:1:" in result.content


class TestGlob:
    def test_finds_files(self, repo: Path):
        result = read_tools.glob(repo, pattern="*.py")
        assert result.ok
        assert "app.py" in result.content

    def test_supports_recursive_patterns(self, repo: Path):
        result = read_tools.glob(repo, pattern="**/*.py")
        assert "src/utils/helpers.py" in result.content

    def test_excludes_secrets(self, repo: Path):
        result = read_tools.glob(repo, pattern="*")
        assert ".env" not in result.content
        assert "secrets.yaml" not in result.content

    @pytest.mark.parametrize("pattern", ["../*", "/etc/*", "C:/*"])
    def test_rejects_escapes(self, repo: Path, pattern: str):
        assert not read_tools.glob(repo, pattern=pattern).ok

    def test_reports_no_matches(self, repo: Path):
        result = read_tools.glob(repo, pattern="*.nonexistent")
        assert result.ok and result.metadata["count"] == 0


class TestRead:
    def test_returns_numbered_lines(self, repo: Path):
        result = read_tools.read(repo, path="config.py", offset=1, limit=5)
        assert result.ok
        assert "1 | import os" in result.content
        assert result.metadata["start_line"] == 1

    def test_reads_a_focused_range(self, repo: Path):
        result = read_tools.read(repo, path="handlers.py", offset=6, limit=3)
        assert result.metadata["start_line"] == 6
        assert result.metadata["end_line"] == 8
        assert "def validate" in result.content

    def test_caps_the_line_limit(self, repo: Path):
        from config import LIMITS

        big = repo / "big.py"
        big.write_text("\n".join(f"line {i}" for i in range(2000)), encoding="utf-8")
        result = read_tools.read(repo, path="big.py", offset=1, limit=9999)
        body = result.content.split("\n", 1)[1]
        assert len(body.splitlines()) <= LIMITS.max_read_lines

    def test_rejects_offset_past_end(self, repo: Path):
        result = read_tools.read(repo, path="config.py", offset=9999)
        assert not result.ok
        assert "past the end" in result.content

    def test_refuses_secret_files(self, repo: Path):
        result = read_tools.read(repo, path=".env")
        assert not result.ok
        assert "secret" in result.content.lower()

    def test_refuses_binaries(self, repo: Path):
        result = read_tools.read(repo, path="assets/logo.png")
        assert not result.ok
        assert "binary" in result.content.lower()

    def test_refuses_traversal(self, repo: Path):
        assert not read_tools.read(repo, path="../../../etc/passwd").ok

    def test_reports_truncation_when_more_remains(self, repo: Path):
        result = read_tools.read(repo, path="handlers.py", offset=1, limit=2)
        assert result.metadata["truncated"] is True


class TestBash:
    def test_allows_approved_git_reads(self, repo: Path):
        result = read_tools.bash(repo, command="git status")
        assert result.ok

    def test_allows_ls(self, repo: Path):
        result = read_tools.bash(repo, command="ls")
        assert result.ok
        assert "app.py" in result.content

    @pytest.mark.parametrize(
        "command",
        ["rm -rf /", "curl http://evil.test", "python -c 'print(1)'", "chmod 777 .",
         "sudo ls", "wget http://x", "nc -l 4444"],
    )
    def test_refuses_unapproved_programs(self, repo: Path, command: str):
        result = read_tools.bash(repo, command=command)
        assert not result.ok
        assert result.metadata.get("refused") is True

    @pytest.mark.parametrize(
        "command",
        ["ls; rm -rf .", "ls && curl evil.test", "ls | sh", "ls > out.txt",
         "ls `whoami`", "ls $(whoami)", "ls & ls"],
    )
    def test_refuses_shell_operators(self, repo: Path, command: str):
        """Any operator that turns one command into several, or writes to disk."""
        result = read_tools.bash(repo, command=command)
        assert not result.ok
        assert "Shell operators" in result.content

    @pytest.mark.parametrize(
        "command", ["git push origin main", "git commit -m x", "git checkout -b new",
                    "git reset --hard", "git clean -fd"],
    )
    def test_refuses_git_write_subcommands(self, repo: Path, command: str):
        assert not read_tools.bash(repo, command=command).ok

    def test_refuses_git_config_writes(self, repo: Path):
        assert not read_tools.bash(repo, command="git config --add user.name x").ok
        assert read_tools.bash(repo, command="git config user.name").ok is not None

    def test_refuses_absolute_and_traversal_paths(self, repo: Path):
        assert not read_tools.bash(repo, command="ls /etc").ok
        assert not read_tools.bash(repo, command="ls ../..").ok

    def test_refuses_secret_path_arguments(self, repo: Path):
        assert not read_tools.bash(repo, command="ls .env").ok


class TestEdit:
    def test_replaces_a_unique_string(self, repo: Path):
        result = edit_tools.edit(
            repo, path="config.py",
            old_string="raise RuntimeError('SECRET_KEY is required')",
            new_string="raise RuntimeError('SECRET_KEY must be configured')",
        )
        assert result.ok
        assert "must be configured" in (repo / "config.py").read_text(encoding="utf-8")

    def test_refuses_when_target_is_missing(self, repo: Path):
        result = edit_tools.edit(
            repo, path="config.py", old_string="text that is not there", new_string="x"
        )
        assert not result.ok
        assert result.metadata["matches"] == 0
        assert "re-read" in result.content.lower()

    def test_refuses_ambiguous_targets(self, repo: Path):
        """Two matches means the model does not actually know where it is
        editing, so nothing is written."""
        target = repo / "dup.py"
        target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
        result = edit_tools.edit(repo, path="dup.py", old_string="value = 1", new_string="value = 2")
        assert not result.ok
        assert result.metadata["matches"] == 2
        assert target.read_text(encoding="utf-8") == "value = 1\nvalue = 1\n"

    def test_refuses_identical_strings(self, repo: Path):
        assert not edit_tools.edit(repo, path="config.py", old_string="x", new_string="x").ok

    def test_refuses_secret_files(self, repo: Path):
        assert not edit_tools.edit(
            repo, path=".env", old_string="OPENAI_API_KEY", new_string="X"
        ).ok

    def test_refuses_traversal(self, repo: Path):
        assert not edit_tools.edit(
            repo, path="../outside.py", old_string="a", new_string="b"
        ).ok

    def test_returns_a_diff(self, repo: Path):
        result = edit_tools.edit(
            repo, path="README.md", old_string="# Sample", new_string="# Sample Project"
        )
        assert "+# Sample Project" in result.content
        assert "-# Sample" in result.content


class TestCreate:
    """`create` writes new files; `edit` changes existing ones.

    The pair has to stay disjoint. If `create` could overwrite, a whole file
    could be replaced by a call that quoted none of its current contents — the
    exact failure `edit`'s exact-match rule exists to make impossible.
    """

    def test_writes_a_new_file(self, repo: Path):
        result = create_tools.create(
            repo, path="utils/text.py", content="def shout(value):\n    return value.upper()\n"
        )
        assert result.ok
        assert (repo / "utils" / "text.py").read_text(encoding="utf-8").startswith("def shout")
        assert result.metadata["path"] == "utils/text.py"
        assert result.metadata["lines"] == 2

    def test_creates_missing_parent_folders(self, repo: Path):
        """The answer to 'make me a folder': a folder arrives with its file,
        because git cannot track an empty directory."""
        result = create_tools.create(
            repo, path="pkg/parsers/json_parser.py", content="VALUE = 1\n"
        )
        assert result.ok
        assert (repo / "pkg" / "parsers").is_dir()
        assert result.metadata["created_directories"] == ["pkg", "pkg/parsers"]

    def test_reports_only_folders_it_actually_made(self, repo: Path):
        result = create_tools.create(repo, path="src/utils/new.py", content="X = 1\n")
        assert result.ok
        # src/ and src/utils/ already exist in the fixture repository.
        assert result.metadata["created_directories"] == []

    def test_refuses_an_existing_file(self, repo: Path):
        before = (repo / "app.py").read_text(encoding="utf-8")
        result = create_tools.create(repo, path="app.py", content="wiped\n")
        assert not result.ok
        assert "already exists" in result.content
        assert "edit" in result.content
        assert (repo / "app.py").read_text(encoding="utf-8") == before

    def test_refuses_an_existing_directory(self, repo: Path):
        result = create_tools.create(repo, path="src", content="not a file\n")
        assert not result.ok
        assert "already exists" in result.content

    def test_points_a_directory_request_at_a_file(self, repo: Path):
        result = create_tools.create(repo, path="docs/", content="x\n")
        assert not result.ok
        assert "README.md" in result.content
        assert not (repo / "docs").exists()

    @pytest.mark.parametrize("path", ["../escape.py", "/etc/passwd", "C:/temp/x.py"])
    def test_refuses_paths_outside_the_workspace(self, repo: Path, path: str):
        assert not create_tools.create(repo, path=path, content="x\n").ok

    @pytest.mark.parametrize("path", ["keys/private.pem", "src/credentials", "id_rsa"])
    def test_refuses_secret_paths(self, repo: Path, path: str):
        result = create_tools.create(repo, path=path, content="x\n")
        assert not result.ok
        assert not (repo / path).exists()

    def test_never_overwrites_an_existing_secret_file(self, repo: Path):
        before = (repo / ".env").read_bytes()
        assert not create_tools.create(repo, path=".env", content="OPENAI_API_KEY=x\n").ok
        assert (repo / ".env").read_bytes() == before

    def test_refuses_blocked_directories(self, repo: Path):
        assert not create_tools.create(repo, path=".git/hooks/pre-commit", content="x\n").ok

    def test_refuses_empty_content(self, repo: Path):
        assert not create_tools.create(repo, path="blank.py", content="   \n").ok

    def test_refuses_oversized_content(self, repo: Path):
        oversized = "x" * (create_tools.MAX_NEW_FILE_CHARS + 1)
        result = create_tools.create(repo, path="big.py", content=oversized)
        assert not result.ok
        assert not (repo / "big.py").exists()

    def test_refuses_binary_content(self, repo: Path):
        assert not create_tools.create(repo, path="bin.dat", content="a\x00b").ok

    def test_normalises_line_endings_and_final_newline(self, repo: Path):
        """A new file has no existing convention to preserve, so it gets the
        one every other tool in the chain assumes."""
        create_tools.create(repo, path="crlf.py", content="a = 1\r\nb = 2")
        written = (repo / "crlf.py").read_bytes()
        assert b"\r" not in written
        assert written.endswith(b"\n")

    def test_previews_the_content_it_wrote(self, repo: Path):
        result = create_tools.create(repo, path="p.py", content="alpha = 1\nbeta = 2\n")
        assert "alpha = 1" in result.content
        assert "beta = 2" in result.content

    def test_redacts_secrets_from_the_preview(self, repo: Path):
        """The preview echoes the content back to the model and the trail, so
        it goes through the same redaction as any other tool output."""
        result = create_tools.create(
            repo, path="cfg.py", content='TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz0123"\n'
        )
        assert result.ok
        assert "ghp_abcdefghijklmnopqrstuvwxyz0123" not in result.content


class TestRegistry:
    def test_read_only_registry_excludes_edit(self, repo: Path):
        registry = ToolRegistry(repo, capabilities=READ_ONLY_CAPABILITIES)
        assert registry.names == ["bash", "glob", "grep", "read"]
        assert "edit" not in registry.names

    def test_edit_is_refused_without_the_capability(self, repo: Path):
        """Defence in depth: even if the model invents the call, dispatch says no."""
        registry = ToolRegistry(repo, capabilities=READ_ONLY_CAPABILITIES)
        result = registry.dispatch("edit", {"path": "app.py", "old_string": "a", "new_string": "b"})
        assert not result.ok
        assert result.metadata["refused"] is True

    def test_create_is_refused_without_the_capability(self, repo: Path):
        """A read-only run must not be able to write a file into the workspace."""
        registry = ToolRegistry(repo, capabilities=READ_ONLY_CAPABILITIES)
        result = registry.dispatch("create", {"path": "planted.py", "content": "x = 1\n"})
        assert not result.ok
        assert result.metadata["refused"] is True
        assert not (repo / "planted.py").exists()

    def test_editing_registry_includes_edit_and_run_check(self, repo: Path):
        registry = ToolRegistry(repo, capabilities=EDITING_CAPABILITIES)
        assert "edit" in registry.names
        assert "create" in registry.names
        assert "run_check" in registry.names

    def test_rejects_unknown_tools(self, repo: Path):
        result = ToolRegistry(repo).dispatch("delete_everything", {})
        assert not result.ok

    def test_rejects_unexpected_arguments(self, repo: Path):
        result = ToolRegistry(repo).dispatch("read", {"file": "app.py"})
        assert not result.ok
        assert "Unexpected argument" in result.content

    def test_rejects_missing_required_arguments(self, repo: Path):
        result = ToolRegistry(repo).dispatch("grep", {})
        assert not result.ok
        assert "Missing required" in result.content

    def test_records_duration(self, repo: Path):
        result = ToolRegistry(repo).dispatch("glob", {"pattern": "*.py"})
        assert "duration_ms" in result.metadata

    def test_schema_order_is_stable(self, repo: Path):
        """A reordered tool list invalidates the provider prompt cache."""
        first = [s["function"]["name"] for s in ToolRegistry(repo).schemas()]
        second = [s["function"]["name"] for s in ToolRegistry(repo).schemas()]
        assert first == second == sorted(first)

    def test_a_crashing_tool_does_not_end_the_request(self, repo: Path, monkeypatch):
        def explode(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(read_tools, "glob", explode)
        registry = ToolRegistry(repo)
        registry._specs["glob"] = registry._specs["glob"].__class__(
            **{**registry._specs["glob"].__dict__, "handler": explode}
        )
        result = registry.dispatch("glob", {"pattern": "*"})
        assert not result.ok
        assert "RuntimeError" in result.content
