"""Durable SQL conversation storage (Phase 8).

Implements the same :class:`ConversationStore` contract as the in-memory store,
so promoting a deployment to durable storage is a ``CONVERSATION_STORE=sqlite``
configuration change rather than a code change.

SQLite is used directly through the standard library. Connections are
thread-local because Flask serves requests — and streams agent runs — on
multiple threads, and a single SQLite connection is not safe to share across
them. WAL mode lets readers proceed during writes.

For Postgres, keep this class as the reference implementation and swap the
``_connect``/parameter style; the SQL is intentionally plain.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from config import BASE_DIR, LIMITS
from services.storage.base import ConversationStore, Message, UsageRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    thread_id    TEXT NOT NULL,
    user_id      TEXT NOT NULL DEFAULT '',
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id, created_at);

CREATE TABLE IF NOT EXISTS usage_records (
    id                  TEXT PRIMARY KEY,
    thread_id           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL,
    cached_input_tokens INTEGER NOT NULL,
    output_tokens       INTEGER NOT NULL,
    reasoning_tokens    INTEGER NOT NULL,
    cost_usd            REAL NOT NULL,
    created_at          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user_time ON usage_records(user_id, created_at);
"""


def resolve_sqlite_path(database_url: str) -> Path:
    """Turn ``sqlite:///relative/path.db`` into an absolute Path."""
    prefix = "sqlite:///"
    raw = database_url[len(prefix):] if database_url.startswith(prefix) else database_url
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


class SqlConversationStore(ConversationStore):
    def __init__(
        self,
        database_url: str,
        *,
        max_messages: int = LIMITS.max_history_messages,
        max_chars: int = LIMITS.max_history_chars,
    ):
        self._path = resolve_sqlite_path(database_url)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_messages = max_messages
        self._max_chars = max_chars
        self._local = threading.local()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    # -- history ---------------------------------------------------------
    def append(self, message: Message) -> Message:
        import json

        conn = self._connect()
        with conn:
            conn.execute(
                "INSERT INTO messages (id, thread_id, user_id, role, content, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message.id,
                    message.thread_id,
                    str(message.metadata.get("user_id", "")),
                    message.role,
                    message.content,
                    json.dumps(message.metadata, default=str),
                    message.created_at,
                ),
            )
            self._trim_thread(conn, message.thread_id)
        return message

    def _trim_thread(self, conn: sqlite3.Connection, thread_id: str) -> None:
        """Enforce the same count and character caps as the in-memory store."""
        rows = conn.execute(
            "SELECT id, LENGTH(content) AS n FROM messages"
            " WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ).fetchall()

        doomed: list[str] = []
        surviving = list(rows)
        while len(surviving) > self._max_messages:
            doomed.append(surviving.pop(0)["id"])
        total = sum(r["n"] for r in surviving)
        while total > self._max_chars and len(surviving) > 1:
            row = surviving.pop(0)
            total -= row["n"]
            doomed.append(row["id"])

        if doomed:
            conn.executemany(
                "DELETE FROM messages WHERE id = ?", [(i,) for i in doomed]
            )

    def history(self, thread_id: str, *, limit: int | None = None) -> list[Message]:
        import json

        sql = (
            "SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at ASC"
        )
        rows = self._connect().execute(sql, (thread_id,)).fetchall()
        if limit:
            rows = rows[-limit:]
        return [
            Message(
                id=r["id"],
                thread_id=r["thread_id"],
                role=r["role"],  # type: ignore[arg-type]
                content=r["content"],
                metadata=json.loads(r["metadata"] or "{}"),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def threads(self, *, user_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        params: list[Any] = []
        where = ""
        if user_id is not None:
            where = "WHERE user_id = ?"
            params.append(user_id)
        rows = self._connect().execute(
            f"""
            SELECT thread_id,
                   COUNT(*)        AS message_count,
                   MAX(created_at) AS updated_at
            FROM messages {where}
            GROUP BY thread_id
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

        summaries = []
        for row in rows:
            title_row = self._connect().execute(
                "SELECT content FROM messages WHERE thread_id = ? AND role = 'user'"
                " ORDER BY created_at ASC LIMIT 1",
                (row["thread_id"],),
            ).fetchone()
            summaries.append(
                {
                    "thread_id": row["thread_id"],
                    "title": (title_row["content"][:80] if title_row else ""),
                    "message_count": row["message_count"],
                    "updated_at": row["updated_at"],
                }
            )
        return summaries

    def delete_thread(self, thread_id: str) -> bool:
        conn = self._connect()
        with conn:
            cur = conn.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        return cur.rowcount > 0

    # -- usage -----------------------------------------------------------
    def record_usage(self, record: UsageRecord) -> UsageRecord:
        conn = self._connect()
        with conn:
            conn.execute(
                "INSERT INTO usage_records (id, thread_id, user_id, model, input_tokens,"
                " cached_input_tokens, output_tokens, reasoning_tokens, cost_usd, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.thread_id,
                    record.user_id,
                    record.model,
                    record.input_tokens,
                    record.cached_input_tokens,
                    record.output_tokens,
                    record.reasoning_tokens,
                    record.cost_usd,
                    record.created_at,
                ),
            )
        return record

    def usage_since(self, *, user_id: str, since: float) -> list[UsageRecord]:
        rows = self._connect().execute(
            "SELECT * FROM usage_records WHERE user_id = ? AND created_at >= ?"
            " ORDER BY created_at ASC",
            (user_id, since),
        ).fetchall()
        return [
            UsageRecord(
                id=r["id"],
                thread_id=r["thread_id"],
                user_id=r["user_id"],
                model=r["model"],
                input_tokens=r["input_tokens"],
                cached_input_tokens=r["cached_input_tokens"],
                output_tokens=r["output_tokens"],
                reasoning_tokens=r["reasoning_tokens"],
                cost_usd=r["cost_usd"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


def utcnow() -> float:
    return time.time()
