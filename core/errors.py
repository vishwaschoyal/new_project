"""Application error types.

Every error carries an HTTP status and a machine-readable code so routes can
translate failures into JSON without each blueprint inventing its own shape.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for expected, user-visible failures."""

    status_code = 400
    code = "app_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        payload = {"error": self.message, "code": self.code}
        if self.details:
            payload["details"] = self.details
        return payload


class ValidationError(AppError):
    status_code = 400
    code = "validation_error"


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    code = "forbidden"


class QuotaExceededError(AppError):
    status_code = 429
    code = "quota_exceeded"


class SafetyError(AppError):
    """A request tried to cross a safety boundary.

    Raised for workspace escapes, secret paths, unapproved executables, unsafe
    shell syntax, and any attempt to publish without approval. These are never
    retried automatically.
    """

    status_code = 403
    code = "safety_violation"


class WorkspaceError(AppError):
    status_code = 400
    code = "workspace_error"


class SandboxError(AppError):
    status_code = 500
    code = "sandbox_error"


class ProviderError(AppError):
    """The model provider failed. The loop finalises on best-effort evidence."""

    status_code = 502
    code = "provider_error"


class BudgetExceededError(AppError):
    status_code = 429
    code = "budget_exceeded"
