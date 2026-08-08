"""Flask application factory.

The factory registers blueprints and one error handler. Route modules stay thin:
validation and orchestration live in services and agents, so the same logic is
reachable from tests and scripts without a request context.
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template

from config import SETTINGS
from core.errors import AppError
from core.logging_setup import configure_logging, install_request_logging

log = logging.getLogger(__name__)


def create_app(**overrides) -> Flask:
    configure_logging(SETTINGS.log_level)

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=SETTINGS.secret_key,
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,  # no request body needs more than this
        TEMPLATES_AUTO_RELOAD=True,
    )
    app.config.update(overrides)

    install_request_logging(app)

    from routes.chat_routes import chat_bp
    from routes.health_routes import health_bp
    from routes.history_routes import history_bp
    from routes.repo_routes import repo_bp
    from routes.task_routes import task_bp

    app.register_blueprint(repo_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(task_bp)
    app.register_blueprint(health_bp)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.errorhandler(AppError)
    def handle_app_error(exc: AppError):
        # Expected failures: the message is written for the user to read.
        log.info("app error", extra={"code": exc.code, "detail": exc.message})
        return jsonify(exc.to_dict()), exc.status_code

    @app.errorhandler(404)
    def handle_404(_exc):
        return jsonify({"error": "Not found", "code": "not_found"}), 404

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        # Unexpected failures: log the trace, return no internals to the client.
        log.exception("unhandled error", extra={"error_type": type(exc).__name__})
        return jsonify({"error": "Internal server error", "code": "internal_error"}), 500

    return app


app = create_app()


def _startup_banner(host: str, port: int) -> str:
    """What the operator needs to know before the first request.

    Flask's own "Running on ..." line is suppressed because werkzeug is pinned
    to WARNING in configure_logging, so the URL is printed here. The rest is the
    configuration that silently changes behaviour — the model being billed, and
    whether repository code actually runs isolated.
    """
    from config import READ_LOOP_MODEL, SETTINGS
    from services.sandbox_service import active_backend, sandbox_available

    url = f"http://{'127.0.0.1' if host in {'0.0.0.0', ''} else host}:{port}"
    backend = active_backend()
    isolated = backend == "docker" and sandbox_available()

    lines = [
        "",
        "  AI Coding Workspace",
        f"  ---> {url}",
        "",
        f"  model    {READ_LOOP_MODEL}" + ("" if SETTINGS.model_configured else "   (no OPENAI_API_KEY!)"),
        f"  store    {SETTINGS.conversation_store}",
        f"  sandbox  {backend}" + ("  (isolated)" if isolated else "  (NOT isolated)"),
        "",
    ]

    # ASCII only below: the Windows console is cp1252 and turns an em-dash into
    # a replacement character.
    if not SETTINGS.model_configured:
        lines.append("  ! OPENAI_API_KEY is not set - every run will fail.")
    if not isolated:
        lines.append("  ! Repository checks run without container isolation.")
        lines.append("    Start Docker Desktop for a real sandbox. Ask mode is unaffected;")
        lines.append("    it never executes repository code.")
    if SETTINGS.auth_enabled is False:
        lines.append("  ! AUTH_ENABLED=false - every visitor shares one identity and quota.")
        lines.append("    Fine locally, not for anything reachable from the internet.")

    lines += ["", "  Ctrl+C to stop.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import os

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))

    print(_startup_banner(host, port), flush=True)
    app.run(host=host, port=port, debug=False, threaded=True)
