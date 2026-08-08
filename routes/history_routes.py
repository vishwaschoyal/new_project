"""Bounded conversation history endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import app_state
from config import LIMITS, SETTINGS
from core.errors import ValidationError
from services import quota_service

history_bp = Blueprint("history", __name__, url_prefix="/api/history")


@history_bp.get("")
def get_history():
    thread_id = request.args.get("thread_id", "")
    if not thread_id:
        raise ValidationError("thread_id is required.")
    quota_service.require_thread_access(quota_service.current_user_id(), thread_id)

    messages = app_state.conversations.history(
        thread_id, limit=LIMITS.max_history_messages
    )
    return jsonify(
        {
            "thread_id": thread_id,
            "messages": [m.to_dict() for m in messages],
        }
    )


@history_bp.get("/threads")
def list_threads():
    user_id = quota_service.current_user_id()
    return jsonify(
        {
            "threads": app_state.conversations.threads(
                user_id=user_id if SETTINGS.auth_enabled else None, limit=50
            )
        }
    )


@history_bp.delete("")
def delete_thread():
    thread_id = request.args.get("thread_id", "")
    if not thread_id:
        raise ValidationError("thread_id is required.")
    quota_service.require_thread_access(quota_service.current_user_id(), thread_id)

    deleted = app_state.conversations.delete_thread(thread_id)
    # Deleting the conversation also releases the cloned repository: keeping a
    # workspace for a thread the user has discarded is a storage leak.
    app_state.workspaces.unload(thread_id)
    return jsonify({"deleted": deleted})
