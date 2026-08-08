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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
