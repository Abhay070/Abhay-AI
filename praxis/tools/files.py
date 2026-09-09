"""Reading files the user uploaded.

Uploads are parsed to text at upload time (see server.py) and stored with an id.
This tool lets the model pull the full text back when the excerpt in context is
not enough."""

from __future__ import annotations

from pathlib import Path

from . import Tool, ToolResult, register

_store = None
MAX_CHARS = 20000


def bind(store) -> None:
    global _store
    _store = store


def read_file(file_id: str = "", filename: str = "") -> ToolResult:
    if _store is None:
        return ToolResult(False, "File storage is not available.")

    record = _store.get_file(file_id) if file_id else None
    if record is None and filename:
        with _store.connect() as db:
            row = db.execute("SELECT * FROM files WHERE filename = ? "
                             "ORDER BY created_at DESC LIMIT 1", (filename,)).fetchone()
        record = dict(row) if row else None
    if record is None:
        return ToolResult(False, f"No uploaded file matching "
                                 f"{file_id or filename!r}.")

    path = Path(record["path"])
    if not path.exists():
        return ToolResult(False, f"File '{record['filename']}' is registered but missing "
                                 "from disk.")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ToolResult(False, f"Could not read '{record['filename']}': {e}")

    truncated = len(text) > MAX_CHARS
    body = text[:MAX_CHARS]
    note = f"\n\n[truncated at {MAX_CHARS} of {len(text)} characters]" if truncated else ""
    return ToolResult(True, f"# {record['filename']}\n\n{body}{note}",
                      {"filename": record["filename"], "truncated": truncated})


register(Tool(
    name="read_file",
    description="Read the full text of a file the user uploaded. Use when the excerpt "
                "shown in the conversation is not enough to answer properly",
    args={"file_id": "the file's id", "filename": "or its name"},
    run=read_file,
    icon="▤",
))
