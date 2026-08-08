"""GitHub URL validation and public discovery over the REST API.

URL parsing is strict and allow-list based. The permissive alternative — accept
anything that looks like a URL and let ``git clone`` decide — would let a user
hand us ``file:///`` or an SSH remote and clone something off the host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests

from config import GITHUB_TOKEN, LIMITS
from core.errors import NotFoundError, ValidationError

API_ROOT = "https://api.github.com"

_ALLOWED_HOSTS = {"github.com", "www.github.com"}
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._\-]{1,100}$")
_RESERVED_OWNERS = {
    "settings", "notifications", "explore", "marketplace", "pulls", "issues",
    "sponsors", "collections", "topics", "trending", "features", "about",
    "pricing", "login", "join", "new", "orgs", "apps", "codespaces",
}


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}.git"

    @property
    def html_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"


def _validate_segment(value: str, kind: str) -> str:
    segment = (value or "").strip()
    if segment.endswith(".git"):
        segment = segment[: -len(".git")]
    if not _SEGMENT_RE.match(segment) or segment in {".", ".."}:
        raise ValidationError(f"Invalid GitHub {kind}: {value!r}")
    return segment


def parse_profile_url(url: str) -> str:
    """Accept ``https://github.com/<user>`` or a bare username; return the username."""
    raw = (url or "").strip()
    if not raw:
        raise ValidationError("A GitHub profile URL or username is required.")

    if "://" in raw or raw.startswith("github.com"):
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        if parsed.scheme not in {"http", "https"}:
            raise ValidationError("Only http(s) GitHub URLs are supported.")
        if parsed.netloc.lower() not in _ALLOWED_HOSTS:
            raise ValidationError("Only github.com profiles are supported.")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) != 1:
            raise ValidationError("Expected a profile URL like https://github.com/<username>.")
        raw = parts[0]

    username = _validate_segment(raw, "username")
    if username.lower() in _RESERVED_OWNERS:
        raise ValidationError(f"{username!r} is a reserved GitHub path, not a profile.")
    return username


def parse_repo_url(url: str) -> RepoRef:
    """Accept a github.com repository URL or ``owner/name``; return a RepoRef."""
    raw = (url or "").strip()
    if not raw:
        raise ValidationError("A GitHub repository URL is required.")

    if raw.startswith("git@"):
        raise ValidationError("SSH remotes are not supported. Use the https:// URL.")

    if "://" in raw or raw.startswith("github.com"):
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        if parsed.scheme not in {"http", "https"}:
            raise ValidationError("Only http(s) GitHub URLs are supported.")
        if parsed.netloc.lower() not in _ALLOWED_HOSTS:
            raise ValidationError("Only github.com repositories are supported.")
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) < 2:
            raise ValidationError("Expected a URL like https://github.com/<owner>/<repo>.")
        owner, name = parts[0], parts[1]
    elif raw.count("/") == 1:
        owner, name = raw.split("/")
    else:
        raise ValidationError("Expected a repository URL or 'owner/name'.")

    return RepoRef(_validate_segment(owner, "owner"), _validate_segment(name, "repository"))


# --------------------------------------------------------------------------
# API access
# --------------------------------------------------------------------------
def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "ai-coding-workspace",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _get(path: str, **params: Any) -> Any:
    try:
        response = requests.get(
            f"{API_ROOT}{path}",
            headers=_headers(),
            params=params or None,
            timeout=LIMITS.http_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise NotFoundError(f"GitHub request failed: {exc}") from exc

    if response.status_code == 404:
        raise NotFoundError(f"GitHub returned 404 for {path}")
    if response.status_code == 403 and "rate limit" in response.text.lower():
        raise NotFoundError(
            "GitHub API rate limit reached. Set GITHUB_PERSONAL_ACCESS_TOKEN to raise it."
        )
    if not response.ok:
        raise NotFoundError(f"GitHub returned {response.status_code} for {path}")

    return response.json()


def fetch_profile(username: str) -> dict[str, Any]:
    data = _get(f"/users/{username}")
    return {
        "login": data.get("login"),
        "name": data.get("name"),
        "avatar_url": data.get("avatar_url"),
        "html_url": data.get("html_url"),
        "bio": data.get("bio"),
        "public_repos": data.get("public_repos", 0),
    }


_authenticated_login: str | None | object = None
_UNRESOLVED = object()


def authenticated_login() -> str | None:
    """Login of the token's owner, or None when no usable token is set.

    Cached for the process: it cannot change without a restart, and this is
    consulted on every repository listing.
    """
    global _authenticated_login
    if _authenticated_login is not _UNRESOLVED and _authenticated_login is not None:
        return _authenticated_login  # type: ignore[return-value]
    if not GITHUB_TOKEN:
        return None
    try:
        _authenticated_login = _get("/user").get("login")
    except NotFoundError:
        _authenticated_login = None
    return _authenticated_login  # type: ignore[return-value]


def fetch_repositories(username: str, *, limit: int = 100) -> list[dict[str, Any]]:
    """List a user's repositories.

    `/users/{username}/repos` returns **public repositories only**, even when
    authenticated — which is why a token owner could not see their own private
    work. `/user/repos` is the endpoint that includes private repositories, and
    it only exists for the authenticated account, so it is used exactly when the
    requested user *is* the token owner.
    """
    owner = authenticated_login()
    if owner and owner.lower() == username.lower():
        data = _get(
            "/user/repos",
            per_page=min(limit, 100),
            sort="updated",
            affiliation="owner,collaborator,organization_member",
        )
    else:
        data = _get(f"/users/{username}/repos", per_page=min(limit, 100), sort="updated")
    repos = []
    for item in data if isinstance(data, list) else []:
        repos.append(
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "html_url": item.get("html_url"),
                "clone_url": item.get("clone_url"),
                "description": item.get("description"),
                "language": item.get("language"),
                "stars": item.get("stargazers_count", 0),
                "default_branch": item.get("default_branch", "main"),
                "private": item.get("private", False),
                "size_kb": item.get("size", 0),
                "updated_at": item.get("updated_at"),
            }
        )
    return repos


def fetch_repository(ref: RepoRef) -> dict[str, Any]:
    data = _get(f"/repos/{ref.owner}/{ref.name}")
    return {
        "full_name": data.get("full_name"),
        "default_branch": data.get("default_branch", "main"),
        "size_kb": data.get("size", 0),
        "private": data.get("private", False),
        "html_url": data.get("html_url"),
        "clone_url": data.get("clone_url"),
        "description": data.get("description"),
    }


def create_pull_request(
    ref: RepoRef,
    *,
    title: str,
    body: str,
    head: str,
    base: str,
) -> dict[str, Any]:
    """Open a pull request. Only ever called after explicit user approval."""
    if not GITHUB_TOKEN:
        raise ValidationError(
            "GITHUB_PERSONAL_ACCESS_TOKEN is required to open a pull request."
        )
    try:
        response = requests.post(
            f"{API_ROOT}/repos/{ref.owner}/{ref.name}/pulls",
            headers=_headers(),
            json={"title": title, "body": body, "head": head, "base": base},
            timeout=LIMITS.http_timeout_seconds,
        )
    except requests.RequestException as exc:
        raise NotFoundError(f"GitHub request failed: {exc}") from exc

    if response.status_code not in (200, 201):
        raise ValidationError(
            f"GitHub refused the pull request ({response.status_code}): {response.text[:300]}"
        )

    data = response.json()
    return {"number": data.get("number"), "html_url": data.get("html_url")}
