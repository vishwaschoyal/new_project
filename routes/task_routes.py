"""Coding-task endpoints: start, review the diff, approve, discard.

Publication lives behind ``POST /api/task/approve`` and nowhere else. The agent
has no route to it — approval is a request the browser makes after a human has
looked at the diff returned by ``/api/task/review``.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import app_state
from core.errors import ValidationError
from services import quota_service
from services.publish_service import VALID_ACTIONS, publisher

task_bp = Blueprint("task", __name__, url_prefix="/api/task")


def _payload() -> dict:
    return request.get_json(silent=True) or {}


def _thread_id(payload: dict | None = None) -> str:
    source = payload if payload is not None else _payload()
    thread_id = source.get("thread_id") or request.args.get("thread_id") or ""
    if not thread_id:
        raise ValidationError("thread_id is required.")
    quota_service.require_thread_access(quota_service.current_user_id(), thread_id)
    return thread_id


@task_bp.post("/start")
def start_task():
    """Create the task branch. Required before the agent may edit anything."""
    payload = _payload()
    thread_id = _thread_id(payload)
    workspace = app_state.workspaces.require(thread_id)
    task = publisher.start_task(workspace, payload.get("description", ""))
    return jsonify({"task": task.to_dict()})


@task_bp.get("/status")
def task_status():
    task = publisher.get(_thread_id())
    return jsonify({"active": task is not None, "task": task.to_dict() if task else None})


@task_bp.get("/review")
def review_task():
    """The diff and check results the user approves or rejects."""
    thread_id = _thread_id()
    workspace = app_state.workspaces.require(thread_id)
    return jsonify(publisher.review(workspace))


@task_bp.post("/approve")
def approve_task():
    """The only path to a commit, a push, or a pull request."""
    payload = _payload()
    thread_id = _thread_id(payload)

    action = (payload.get("action") or "").strip().lower()
    if action not in VALID_ACTIONS:
        raise ValidationError(f"action must be one of {sorted(VALID_ACTIONS)}.")

    # Approval is a decision about a specific diff. Requiring the client to
    # confirm explicitly keeps a stray POST from publishing.
    if not payload.get("confirmed"):
        raise ValidationError("Set 'confirmed': true to approve this action.")

    workspace = app_state.workspaces.require(thread_id)
    outcome = publisher.approve(
        workspace,
        action=action,
        commit_message=payload.get("commit_message", ""),
        pr_title=payload.get("pr_title", ""),
        pr_body=payload.get("pr_body", ""),
        approved_by=quota_service.current_user_id(),
    )
    return jsonify(outcome)


@task_bp.post("/discard")
def discard_task():
    thread_id = _thread_id()
    workspace = app_state.workspaces.require(thread_id)
    return jsonify(publisher.discard(workspace))


@task_bp.get("/usage")
def usage():
    return jsonify(quota_service.usage_summary(quota_service.current_user_id()))
