"""Process-wide singletons.

Kept in one module so tests can swap an implementation (an in-memory store, a
temporary workspace root) without importing the whole application.
"""

from __future__ import annotations

from services.storage import ConversationStore, create_store
from services.workspace_service import WorkspaceService

workspaces = WorkspaceService()
conversations: ConversationStore = create_store()


def reset_for_tests(*, store: ConversationStore | None = None, workspace_root=None) -> None:
    """Replace the singletons. Used only by the test suite."""
    global workspaces, conversations
    if store is not None:
        conversations = store
    if workspace_root is not None:
        workspaces = WorkspaceService(workspace_root)
