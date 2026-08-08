"""Bounded conversation history endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

import app_state
from config import LIMITS
from core.errors import ValidationError

history_bp = Blueprint("history", __name__, url_prefix="/api/history")


@history_bp.get("")
def get_history():
    thread_id = request.args.get("thread_id", "")
    if not thread_id:
        raise ValidationError("thread_id is required.")

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
    return jsonify({"threads": app_state.conversations.threads(limit=50)})


@history_bp.delete("")
def delete_thread():
    thread_id = request.args.get("thread_id", "")
    if not thread_id:
        raise ValidationError("thread_id is required.")

    deleted = app_state.conversations.delete_thread(thread_id)
    # Deleting the conversation also releases the cloned repository: keeping a
    # workspace for a thread the user has discarded is a storage leak.
    app_state.workspaces.unload(thread_id)
    return jsonify({"deleted": deleted})
