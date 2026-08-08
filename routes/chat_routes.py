"""The chat endpoint: validate, run the agent, stream the run over SSE.

The agent runs on a worker thread and pushes events into a queue; the response
generator drains that queue. That indirection exists because the agent is
synchronous and blocking — running it inside the generator would buffer the
whole investigation and deliver it as one lump after the run finished, which is
exactly the experience streaming is meant to avoid.

The queue is bounded. If a client disconnects mid-run the generator stops
draining, and an unbounded queue would then hold the entire run in memory.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid

from flask import Blueprint, Response, jsonify, request, stream_with_context

import app_state
from agents.orchestrator import AgentEvent, Orchestrator, messages_from_history
from config import LIMITS, SETTINGS
from core.errors import AppError, ForbiddenError, QuotaExceededError, ValidationError
from services import quota_service
from services.publish_service import publisher
from services.storage import Message, UsageRecord
from tools.registry import EDITING_CAPABILITIES, READ_ONLY_CAPABILITIES

log = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__, url_prefix="/api/chat")

MAX_QUESTION_CHARS = 8_000
EVENT_QUEUE_MAXSIZE = 500
_SENTINEL = object()


def _sse(event: AgentEvent) -> str:
    return f"event: {event.type}\ndata: {json.dumps(event.data, default=str)}\n\n"


@chat_bp.post("")
def chat():
    payload = request.get_json(silent=True) or {}

    question = (payload.get("message") or payload.get("question") or "").strip()
    if not question:
        raise ValidationError("A message is required.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValidationError(f"Message is too long (limit {MAX_QUESTION_CHARS} characters).")

    thread_id = (payload.get("thread_id") or "").strip() or uuid.uuid4().hex[:16]
    mode = (payload.get("mode") or "ask").strip().lower()
    if mode not in {"ask", "code"}:
        raise ValidationError("mode must be 'ask' or 'code'.")

    user_id = quota_service.current_user_id()
    quota_service.require_thread_access(user_id, thread_id)
    quota_service.enforce(user_id)

    workspace = app_state.workspaces.require(thread_id, user_id=user_id)

    # A coding run must be on a task branch. Checked here, before any model
    # spend, and again inside the edit path.
    task = None
    if mode == "code":
        task = publisher.ensure_editable(workspace)

    history = messages_from_history(
        app_state.conversations.history(thread_id, limit=LIMITS.max_history_messages)
    )

    events: queue.Queue = queue.Queue(maxsize=EVENT_QUEUE_MAXSIZE)
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]

    def emit(event: AgentEvent) -> None:
        try:
            events.put(event, timeout=5)
        except queue.Full:
            # The client stopped reading. Dropping progress events is correct;
            # the run continues and its result is still persisted.
            log.warning("event queue full; dropping event", extra={"type": event.type})

    orchestrator = Orchestrator(
        workspace=workspace.path,
        question=question,
        repo_full_name=workspace.repo.full_name,
        branch=workspace.branch,
        history=history,
        capabilities=EDITING_CAPABILITIES if mode == "code" else READ_ONLY_CAPABILITIES,
        emit=emit,
        allow_delegation=True,
        name="main",
    )

    def run_agent() -> None:
        try:
            result = orchestrator.run()
            _persist(thread_id, user_id, question, result, mode=mode)
            if mode == "code":
                _record_task_progress(thread_id, result)
        except AppError as exc:
            emit(AgentEvent("error", {"message": exc.message, "code": exc.code}))
        except Exception as exc:
            log.exception("agent run failed", extra={"thread_id": thread_id})
            emit(AgentEvent("error", {"message": f"The run failed: {exc}"}))
        finally:
            events.put(_SENTINEL)

    worker = threading.Thread(target=run_agent, name=f"agent-{thread_id}", daemon=True)
    worker.start()

    @stream_with_context
    def generate():
        yield _sse(AgentEvent("start", {"thread_id": thread_id, "mode": mode, "request_id": request_id}))
        deadline = time.time() + LIMITS.stream_timeout_seconds

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                yield _sse(AgentEvent("error", {"message": "The run exceeded the stream timeout."}))
                break
            try:
                event = events.get(timeout=min(15.0, remaining))
            except queue.Empty:
                # Keep-alive comment: proxies close an idle connection, and a
                # long read between model calls looks idle.
                yield ": keep-alive\n\n"
                continue

            if event is _SENTINEL:
                break
            yield _sse(event)

        yield _sse(AgentEvent("end", {"thread_id": thread_id}))

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # nginx buffers SSE into uselessness otherwise
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


def _persist(thread_id: str, user_id: str, question: str, result, *, mode: str) -> None:
    """Store the bounded turns and the usage record — never the raw evidence."""
    store = app_state.conversations
    store.append(Message(thread_id=thread_id, role="user", content=question,
                         metadata={"user_id": user_id, "mode": mode}))
    store.append(
        Message(
            thread_id=thread_id,
            role="assistant",
            content=result.answer,
            metadata={
                "user_id": user_id,
                "citations": result.citations,
                "usage": result.usage,
                "termination_reason": result.termination_reason,
                "steps_used": result.steps_used,
            },
        )
    )

    usage = result.usage
    store.record_usage(
        UsageRecord(
            thread_id=thread_id,
            user_id=user_id,
            model=usage.get("model", ""),
            input_tokens=usage.get("input_tokens", 0),
            cached_input_tokens=usage.get("cached_input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            cost_usd=usage.get("cost_usd", 0.0),
        )
    )


def _record_task_progress(thread_id: str, result) -> None:
    """Fold the run's edits and checks into the task audit trail."""
    for call in result.tool_calls:
        if call["name"] == "edit" and call["ok"]:
            publisher.record_edit(thread_id, str(call["arguments"].get("path", "")))
        elif call["name"] == "run_check":
            publisher.record_check(
                thread_id,
                {
                    "command": call["arguments"].get("command", ""),
                    "ok": call["ok"],
                    "backend": "sandbox",
                },
            )


@chat_bp.get("/config")
def chat_config():
    """What the browser needs to render the session correctly."""
    from config import READ_LOOP_MODEL
    from services.sandbox_service import active_backend, sandbox_available

    return jsonify(
        {
            "model": READ_LOOP_MODEL,
            "model_configured": SETTINGS.model_configured,
            "max_steps": LIMITS.max_agent_steps,
            "token_budget": LIMITS.request_token_budget,
            "sandbox_backend": active_backend(),
            "sandbox_isolated": sandbox_available(),
            "auth_enabled": SETTINGS.auth_enabled,
        }
    )
