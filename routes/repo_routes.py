"""Repository lifecycle and inspection endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import app_state
from core.errors import ValidationError
from services import github_service

repo_bp = Blueprint("repo", __name__, url_prefix="/api/repo")


def _json() -> dict:
    return request.get_json(silent=True) or {}


def _thread_id(payload: dict | None = None) -> str:
    source = payload if payload is not None else _json()
    thread_id = source.get("thread_id") or request.args.get("thread_id") or ""
    if not thread_id:
        raise ValidationError("thread_id is required.")
    return thread_id


@repo_bp.post("/profile")
def load_profile():
    """Resolve a GitHub profile and list its public repositories."""
    payload = _json()
    username = github_service.parse_profile_url(payload.get("url", ""))
    return jsonify(
        {
            "profile": github_service.fetch_profile(username),
            "repositories": github_service.fetch_repositories(username),
        }
    )


@repo_bp.post("/load")
def load_repository():
    """Clone a repository into this thread's workspace."""
    payload = _json()
    thread_id = _thread_id(payload)
    url = payload.get("url") or payload.get("repo_url") or ""
    workspace = app_state.workspaces.load(thread_id, url)
    return jsonify({"workspace": workspace.to_dict()})


@repo_bp.get("/status")
def repository_status():
    return jsonify(app_state.workspaces.status(_thread_id()))


@repo_bp.get("/branches")
def repository_branches():
    return jsonify(app_state.workspaces.branches(_thread_id()))


@repo_bp.post("/branch")
def switch_branch():
    """Human-requested branch switch. The agent never calls this."""
    payload = _json()
    workspace = app_state.workspaces.switch_branch(
        _thread_id(payload), payload.get("branch", "")
    )
    return jsonify({"workspace": workspace.to_dict()})


@repo_bp.get("/tree")
def repository_tree():
    return jsonify(
        app_state.workspaces.tree(_thread_id(), request.args.get("path", ""))
    )


@repo_bp.get("/file")
def repository_file():
    path = request.args.get("path", "")
    if not path:
        raise ValidationError("path is required.")
    return jsonify(app_state.workspaces.read_file(_thread_id(), path))


@repo_bp.post("/unload")
def unload_repository():
    payload = _json()
    removed = app_state.workspaces.unload(_thread_id(payload))
    return jsonify({"unloaded": removed})
