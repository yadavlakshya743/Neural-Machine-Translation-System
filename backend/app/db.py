"""SQLite persistence for users and translation history (repository pattern).

Uses Python's built-in sqlite3. Every query that touches history is scoped by user_id, so
a user can only ever read their own records (PRD US-008).
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from backend.app.core.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS translations (
    id                TEXT PRIMARY KEY,
    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_lang       TEXT NOT NULL,
    target_lang       TEXT NOT NULL,
    source_text       TEXT NOT NULL,
    translated_text   TEXT NOT NULL,
    decoding_strategy TEXT NOT NULL,
    beam_size         INTEGER,
    model_version     TEXT NOT NULL,
    latency_ms        REAL NOT NULL,
    tokens_generated  INTEGER NOT NULL,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_translations_user ON translations(user_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = settings.database_path
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# ------------------------------------------------------------------ users
def create_user(username: str, password_hash: str) -> Dict[str, Any]:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, _now()),
        )
        return get_user_by_id(cur.lastrowid, conn)


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[Dict[str, Any]]:
    if conn is None:
        with connect() as c:
            return get_user_by_id(user_id, c)
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


# ------------------------------------------------------------ translations
def insert_translation(record: Dict[str, Any]) -> None:
    cols = ("id", "user_id", "source_lang", "target_lang", "source_text", "translated_text",
            "decoding_strategy", "beam_size", "model_version", "latency_ms", "tokens_generated",
            "created_at")
    record = {**record, "created_at": record.get("created_at") or _now()}
    with connect() as conn:
        conn.execute(
            f"INSERT INTO translations ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
            tuple(record[c] for c in cols),
        )


def list_translations(user_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM translations WHERE user_id = ? ORDER BY created_at DESC, rowid DESC "
            "LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def count_translations(user_id: int) -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM translations WHERE user_id = ?",
                            (user_id,)).fetchone()[0]


def get_translation(user_id: int, translation_id: str) -> Optional[Dict[str, Any]]:
    """Returns the record only if it belongs to user_id."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM translations WHERE id = ? AND user_id = ?",
                           (translation_id, user_id)).fetchone()
        return dict(row) if row else None
