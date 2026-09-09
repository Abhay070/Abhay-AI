"""
Memory tools: what Praxis chooses to keep, and what it can look up.

The model decides what is worth remembering. That is the correct design — a
heuristic that stores everything produces a memory nobody can trust, and one
that stores nothing produces an assistant that asks your name every morning.

The guardrails are in the prompt (below) and in the store's dedupe, not in a
filter that tries to second-guess the model.
"""

from __future__ import annotations

from . import Tool, ToolResult, register

# Injected by the server at startup so tools do not import the app.
_store = None


def bind(store) -> None:
    global _store
    _store = store


def remember(content: str, category: str = "fact", context: dict | None = None) -> ToolResult:
    if _store is None:
        return ToolResult(False, "Memory is not available.")
    content = str(content).strip()
    if len(content) < 3:
        return ToolResult(False, "Too short to be a useful memory.")
    if len(content) > 500:
        return ToolResult(False, "Too long. Store the durable fact, not the paragraph.")

    rid = _store.add_memory(content, category, (context or {}).get("conversation_id"))
    if rid is None:
        return ToolResult(True, f"Already remembered: {content}")
    return ToolResult(True, f"Remembered ({category}): {content}",
                      {"memory_id": rid, "content": content, "category": category})


def recall(query: str = "") -> ToolResult:
    if _store is None:
        return ToolResult(False, "Memory is not available.")
    rows = _store.search_memories(query) if query else _store.list_memories(limit=40)
    if not rows:
        return ToolResult(True, f"Nothing remembered matching '{query}'." if query
                                else "No memories stored yet.")
    _store.touch_memories([r["id"] for r in rows])
    listing = "\n".join(f"- [{r['category']}] {r['content']}" for r in rows[:25])
    return ToolResult(True, f"{len(rows)} memories:\n{listing}", {"count": len(rows)})


def forget(query: str) -> ToolResult:
    """Deletes only on an unambiguous single match. Bulk deletion belongs in the
    UI, where the user can see exactly what is about to disappear."""
    if _store is None:
        return ToolResult(False, "Memory is not available.")
    rows = _store.search_memories(query)
    if not rows:
        return ToolResult(False, f"Nothing matching '{query}'.")
    if len(rows) > 1:
        listing = "\n".join(f"- {r['content']}" for r in rows[:8])
        return ToolResult(False,
                          f"'{query}' matches {len(rows)} memories. Be more specific, or "
                          f"delete them in the Memory panel:\n{listing}")
    _store.delete_memory(rows[0]["id"])
    return ToolResult(True, f"Forgotten: {rows[0]['content']}")


register(Tool(
    name="remember",
    description="Store a durable fact about the user for future conversations. Use for "
                "preferences, ongoing projects, goals, decisions, and skills — NOT for "
                "conversational detail or anything they'd expect you to forget",
    args={"content": "the fact, one sentence",
          "category": "fact | preference | project | goal | decision | skill"},
    run=remember,
    icon="◆",
    requires="enable_memory",
))

register(Tool(
    name="recall",
    description="Search what you remember about the user. Relevant memories are already "
                "in your context — use this only to look for something specific that "
                "isn't",
    args={"query": "search text, or empty for everything"},
    run=recall,
    icon="◇",
    requires="enable_memory",
))

register(Tool(
    name="forget",
    description="Delete a stored memory. Only acts when exactly one memory matches",
    args={"query": "text identifying the memory to delete"},
    run=forget,
    icon="◌",
    requires="enable_memory",
))
