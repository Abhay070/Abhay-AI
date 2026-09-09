"""
Persistence: conversations, messages, and long-term memory.

SQLite, because this is a personal AI. It is a single file you can back up by
copying it, it needs no server, and it will handle years of conversation without
noticing. Reach for Postgres when there are concurrent users, not before.

Memory design note. The spec asks for memory that is selective, controllable,
and deletable. Three properties follow from that and are enforced here:

  1. Memories are keyed facts, not conversation transcripts. "prefers Python
     over JavaScript" is memory; "asked about dicts on Tuesday" is not.
  2. Every memory records where it came from, so you can audit it.
  3. Everything is editable and deletable through the API and the UI. Memory you
     cannot inspect is memory you cannot trust.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    mode        TEXT NOT NULL DEFAULT 'standard',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    pinned      INTEGER NOT NULL DEFAULT 0,
    archived    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL,
    meta            TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'fact',
    source_id   TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_cat ON memories(category);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    filename        TEXT NOT NULL,
    path            TEXT NOT NULL,
    mimetype        TEXT NOT NULL DEFAULT '',
    size            INTEGER NOT NULL DEFAULT 0,
    excerpt         TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL
);
"""

# Memory categories. Kept small on purpose — a taxonomy nobody can remember is
# a taxonomy nobody uses correctly.
CATEGORIES = ("fact", "preference", "project", "goal", "decision", "skill")


def _id() -> str:
    return uuid.uuid4().hex[:16]


def now() -> float:
    return time.time()


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        # WAL lets the UI read while a write is in flight, which matters as soon
        # as streaming and autosave overlap.
        db.execute("PRAGMA journal_mode = WAL")
        try:
            yield db
            db.commit()
        finally:
            db.close()

    # -- conversations -----------------------------------------------------

    def create_conversation(self, title: str = "New conversation",
                            mode: str = "standard") -> str:
        cid, t = _id(), now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations (id,title,mode,created_at,updated_at) "
                "VALUES (?,?,?,?,?)", (cid, title, mode, t, t))
        return cid

    def list_conversations(self, limit: int = 100, archived: bool = False) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT c.*, (SELECT COUNT(*) FROM messages m
                                WHERE m.conversation_id = c.id) AS message_count
                   FROM conversations c
                   WHERE c.archived = ?
                   ORDER BY c.pinned DESC, c.updated_at DESC
                   LIMIT ?""", (int(archived), limit)).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, cid: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id = ?", (cid,)).fetchone()
        return dict(row) if row else None

    def update_conversation(self, cid: str, **fields) -> None:
        allowed = {"title", "mode", "pinned", "archived"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        sets["updated_at"] = now()
        clause = ", ".join(f"{k} = ?" for k in sets)
        with self.connect() as db:
            db.execute(f"UPDATE conversations SET {clause} WHERE id = ?",
                       (*sets.values(), cid))

    def delete_conversation(self, cid: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
            db.execute("DELETE FROM conversations WHERE id = ?", (cid,))

    def search_conversations(self, query: str, limit: int = 40) -> list[dict]:
        """Search titles and message bodies. LIKE is enough at personal scale;
        swap in FTS5 if this ever gets slow."""
        like = f"%{query}%"
        with self.connect() as db:
            rows = db.execute(
                """SELECT DISTINCT c.*,
                          (SELECT COUNT(*) FROM messages m
                           WHERE m.conversation_id = c.id) AS message_count
                   FROM conversations c
                   LEFT JOIN messages m ON m.conversation_id = c.id
                   WHERE c.title LIKE ? OR m.content LIKE ?
                   ORDER BY c.updated_at DESC LIMIT ?""", (like, like, limit)).fetchall()
        return [dict(r) for r in rows]

    # -- messages ----------------------------------------------------------

    def add_message(self, cid: str, role: str, content: str,
                    meta: dict[str, Any] | None = None) -> str:
        mid, t = _id(), now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO messages (id,conversation_id,role,content,created_at,meta) "
                "VALUES (?,?,?,?,?,?)",
                (mid, cid, role, content, t, json.dumps(meta or {})))
            db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (t, cid))
        return mid

    def get_messages(self, cid: str, limit: int = 500) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? "
                "ORDER BY created_at ASC LIMIT ?", (cid, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["meta"] = json.loads(d["meta"])
            except (json.JSONDecodeError, TypeError):
                d["meta"] = {}
            out.append(d)
        return out

    def delete_message(self, mid: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM messages WHERE id = ?", (mid,))

    def truncate_after(self, cid: str, mid: str) -> None:
        """Drop everything after a message. Used when the user edits a turn and
        regenerates from there — the old branch is gone, cleanly."""
        with self.connect() as db:
            row = db.execute("SELECT created_at FROM messages WHERE id = ?", (mid,)).fetchone()
            if row:
                db.execute(
                    "DELETE FROM messages WHERE conversation_id = ? AND created_at > ?",
                    (cid, row["created_at"]))

    # -- memory ------------------------------------------------------------

    def add_memory(self, content: str, category: str = "fact",
                   source_id: str | None = None) -> str | None:
        """Store a durable fact. Returns None if it duplicates one already held.

        The dedupe is deliberately crude — exact match after normalization. A
        smarter version would embed and compare, but crude-and-predictable beats
        clever-and-surprising for something the user has to audit by hand."""
        content = content.strip()
        if not content:
            return None
        if category not in CATEGORIES:
            category = "fact"
        norm = content.lower().rstrip(".")
        with self.connect() as db:
            existing = db.execute(
                "SELECT id FROM memories WHERE LOWER(RTRIM(content,'.')) = ?",
                (norm,)).fetchone()
            if existing:
                db.execute("UPDATE memories SET updated_at = ? WHERE id = ?",
                           (now(), existing["id"]))
                return None
            rid, t = _id(), now()
            db.execute(
                "INSERT INTO memories (id,content,category,source_id,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?)", (rid, content, category, source_id, t, t))
        return rid

    def list_memories(self, category: str | None = None, limit: int = 200) -> list[dict]:
        with self.connect() as db:
            if category:
                rows = db.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY updated_at DESC "
                    "LIMIT ?", (category, limit)).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                    (limit,)).fetchall()
        return [dict(r) for r in rows]

    def search_memories(self, query: str, limit: int = 20) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM memories WHERE content LIKE ? ORDER BY updated_at DESC "
                "LIMIT ?", (f"%{query}%", limit)).fetchall()
        return [dict(r) for r in rows]

    def update_memory(self, rid: str, content: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                       (content, now(), rid))

    def delete_memory(self, rid: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM memories WHERE id = ?", (rid,))

    def clear_memories(self) -> int:
        with self.connect() as db:
            n = db.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
            db.execute("DELETE FROM memories")
        return n

    def touch_memories(self, ids: list[str]) -> None:
        """Count a memory as used. Lets the UI show what is actually earning its
        place in the context window, versus what is just sitting there."""
        if not ids:
            return
        with self.connect() as db:
            db.executemany("UPDATE memories SET hits = hits + 1 WHERE id = ?",
                           [(i,) for i in ids])

    # -- files -------------------------------------------------------------

    def add_file(self, filename: str, path: str, mimetype: str, size: int,
                 excerpt: str, conversation_id: str | None = None) -> str:
        fid = _id()
        with self.connect() as db:
            db.execute(
                "INSERT INTO files (id,conversation_id,filename,path,mimetype,size,"
                "excerpt,created_at) VALUES (?,?,?,?,?,?,?,?)",
                (fid, conversation_id, filename, path, mimetype, size, excerpt, now()))
        return fid

    def get_file(self, fid: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM files WHERE id = ?", (fid,)).fetchone()
        return dict(row) if row else None

    # -- settings ----------------------------------------------------------

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO settings (key,value) VALUES (?,?) "
                       "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                       (key, json.dumps(value)))

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def stats(self) -> dict:
        with self.connect() as db:
            q = lambda s: db.execute(s).fetchone()[0]  # noqa: E731
            return {
                "conversations": q("SELECT COUNT(*) FROM conversations"),
                "messages": q("SELECT COUNT(*) FROM messages"),
                "memories": q("SELECT COUNT(*) FROM memories"),
                "files": q("SELECT COUNT(*) FROM files"),
                "db_size_kb": round(self.path.stat().st_size / 1024, 1)
                if self.path.exists() else 0,
            }
