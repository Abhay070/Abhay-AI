"""
Praxis — API server.

Run:
    pip install -r requirements.txt
    python server.py          →  http://localhost:8000

Endpoints
    GET  /                      landing page
    GET  /chat                  the app
    GET  /api/bootstrap         everything the UI needs on load
    GET  /api/health            provider status
    POST /api/chat              stream a reply (SSE)
    GET/POST/PATCH/DELETE  /api/conversations[/id]
    GET  /api/conversations/:id/export     markdown export
    GET/POST/PATCH/DELETE  /api/memories[/id]
    POST /api/upload            attach a file
    GET  /api/prompt            inspect the exact system prompt being sent
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from praxis import (__version__, council as council_mod, modes as modes_mod,
                    providers, tools as toolkit)
from praxis.agent import Agent
from praxis.config import BRAND, settings
from praxis.store import CATEGORIES, Store
from praxis.tools import files as files_tool, memory as memory_tool

settings.ensure_dirs()
store = Store(settings.db_path)
agent = Agent(store, settings)

# Tools that need storage get it here, so tool modules never import the app.
memory_tool.bind(store)
files_tool.bind(store)

WEB = Path(__file__).parent / "web"

app = FastAPI(title=BRAND.name, version=__version__)
app.mount("/static", StaticFiles(directory=WEB / "static"), name="static")


# -- models -----------------------------------------------------------------

class Attachment(BaseModel):
    """A file the user attached, by reference.

    By reference deliberately. Pasting a PDF's text into the message means the
    user's own transcript is mostly PDF — which is what happened the first time
    someone attached one. The id travels; the text is folded into the prompt at
    compose time and stays out of the visible turn."""

    id: str
    filename: str = ""
    characters: int = 0


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    mode: str = settings.default_mode
    provider: str | None = None
    strategy: str | None = None
    regenerate_from: str | None = None
    attachments: list[Attachment] = []


class ConversationPatch(BaseModel):
    title: str | None = None
    mode: str | None = None
    pinned: bool | None = None
    archived: bool | None = None


class MemoryIn(BaseModel):
    content: str
    category: str = "fact"


# -- pages ------------------------------------------------------------------

@app.get("/")
async def landing() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/chat")
async def chat_page() -> FileResponse:
    return FileResponse(WEB / "chat.html")


# -- meta -------------------------------------------------------------------

@app.get("/api/bootstrap")
async def bootstrap() -> dict:
    """One round trip for everything the UI needs at startup."""
    provider, notes = await providers.resolve(settings.provider, settings.fallback_chain)
    ok, detail = await provider.health()
    active = toolkit.available(settings)
    return {
        "brand": {"name": BRAND.name, "tagline": BRAND.tagline,
                  "owner": BRAND.owner, "version": __version__},
        "provider": {"key": provider.name, "label": provider.label,
                     "ok": ok, "detail": detail, "notes": notes,
                     "requested": settings.provider},
        "providers": providers.catalogue(),
        "strategies": [
            {"key": "single",  "label": "Single",  "icon": "◈",
             "blurb": "One model. Fastest and cheapest."},
            {"key": "council", "label": "Council", "icon": "⚖",
             "blurb": "Several models answer; the best one wins."},
            {"key": "race",    "label": "Race",    "icon": "⚡",
             "blurb": "All at once; first usable answer wins."},
            {"key": "cascade", "label": "Cascade", "icon": "↗",
             "blurb": "Cheap model first, escalate only if weak."},
        ],
        "council": {
            "members": [council_mod.describe(m) for m in
                        council_mod.build_members(settings.council_members)],
            "judge": settings.council_judge or "heuristic",
            "default_strategy": settings.default_strategy,
        },
        "modes": modes_mod.catalogue(),
        "tools": [{"name": t.name, "icon": t.icon, "dangerous": t.dangerous,
                   "description": t.description} for t in active],
        "features": {
            "tools": settings.enable_tools, "memory": settings.enable_memory,
            "web": settings.enable_web, "code": settings.enable_code_exec,
        },
        "memory_categories": list(CATEGORIES),
        "conversations": store.list_conversations(limit=60),
        "stats": store.stats(),
        "user_name": settings.user_name,
    }


@app.get("/api/health")
async def health() -> dict:
    provider, notes = await providers.resolve(settings.provider, settings.fallback_chain)
    ok, detail = await provider.health()
    return {"provider": provider.name, "label": provider.label,
            "ok": ok, "detail": detail, "notes": notes}


@app.get("/api/council")
async def council_status() -> dict:
    """Which council members can actually serve right now, and why not."""
    members = council_mod.build_members(settings.council_members)
    live, dropped = await council_mod.healthy_members(members)
    return {
        "configured": [council_mod.describe(m) for m in members],
        "live": [council_mod.describe(m) for m in live],
        "dropped": dropped,
        "judge": settings.council_judge or "heuristic",
        "default_strategy": settings.default_strategy,
        "usable": len(live) >= 1,
    }


@app.get("/api/prompt")
async def inspect_prompt(mode: str = "standard") -> dict:
    """Show the exact system prompt for a mode.

    This is your own AI — you should be able to see precisely what it was told.
    An assistant whose instructions you cannot read is one you are trusting on
    faith."""
    messages, memories, active = agent.compose([], mode, settings.user_name)
    system = messages[0]["content"]
    return {
        "mode": mode,
        "system_prompt": system,
        "characters": len(system),
        "estimated_tokens": len(system) // 4,
        "memories_included": len(memories),
        "tools_included": [t.name for t in active],
    }


# -- conversations ----------------------------------------------------------

@app.get("/api/conversations")
async def list_conversations(q: str = Query(""), archived: bool = False) -> list[dict]:
    if q:
        return store.search_conversations(q)
    return store.list_conversations(archived=archived)


@app.post("/api/conversations")
async def create_conversation(mode: str = Body("standard", embed=True)) -> dict:
    cid = store.create_conversation(mode=mode)
    return store.get_conversation(cid)


@app.get("/api/conversations/{cid}")
async def get_conversation(cid: str) -> dict:
    conversation = store.get_conversation(cid)
    if not conversation:
        raise HTTPException(404, "No such conversation")
    conversation["messages"] = store.get_messages(cid)
    return conversation


@app.patch("/api/conversations/{cid}")
async def patch_conversation(cid: str, patch: ConversationPatch) -> dict:
    if not store.get_conversation(cid):
        raise HTTPException(404, "No such conversation")
    fields = {k: (int(v) if isinstance(v, bool) else v)
              for k, v in patch.model_dump(exclude_none=True).items()}
    store.update_conversation(cid, **fields)
    return store.get_conversation(cid)


@app.delete("/api/conversations/{cid}")
async def delete_conversation(cid: str) -> dict:
    store.delete_conversation(cid)
    return {"deleted": cid}


@app.get("/api/conversations/{cid}/export")
async def export_conversation(cid: str) -> PlainTextResponse:
    conversation = store.get_conversation(cid)
    if not conversation:
        raise HTTPException(404, "No such conversation")
    lines = [f"# {conversation['title']}", "",
             f"*Exported from {BRAND.name} · mode: {conversation['mode']}*", "", "---", ""]
    for m in store.get_messages(cid):
        who = {"user": BRAND.owner, "assistant": BRAND.name}.get(m["role"], m["role"])
        lines += [f"### {who}", "", m["content"], ""]
    return PlainTextResponse(
        "\n".join(lines), media_type="text/markdown",
        headers={"Content-Disposition":
                 f'attachment; filename="{conversation["title"][:40]}.md"'})


# -- memory -----------------------------------------------------------------

@app.get("/api/memories")
async def list_memories(category: str = Query(""), q: str = Query(""),
                        include_superseded: bool = Query(True)) -> list[dict]:
    # The panel shows superseded memories by default — marked, not hidden — so
    # the user can see what was retired and undo it. The prompt is what filters
    # them out; the audit view should not.
    if q:
        return store.search_memories(q, limit=200,
                                     include_superseded=include_superseded)
    return store.list_memories(category or None,
                               include_superseded=include_superseded)


@app.post("/api/memories/{rid}/restore")
async def restore_memory(rid: str) -> dict:
    store.restore_memory(rid)
    return {"restored": rid}


@app.post("/api/memories/{rid}/confirm")
async def confirm_memory(rid: str) -> dict:
    store.confirm_memory(rid)
    return {"confirmed": rid}


@app.post("/api/memories")
async def create_memory(item: MemoryIn) -> dict:
    rid = store.add_memory(item.content, item.category,
                           origin="user", confidence=1.0, confirmed=True)
    if rid is None:
        raise HTTPException(409, "Already remembered")
    return {"id": rid, "content": item.content, "category": item.category}


@app.patch("/api/memories/{rid}")
async def patch_memory(rid: str, item: MemoryIn) -> dict:
    store.update_memory(rid, item.content)
    return {"id": rid, "content": item.content}


@app.delete("/api/memories/{rid}")
async def delete_memory(rid: str) -> dict:
    store.delete_memory(rid)
    return {"deleted": rid}


@app.delete("/api/memories")
async def clear_memories() -> dict:
    return {"deleted": store.clear_memories()}


# -- files ------------------------------------------------------------------

TEXTUAL = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".tsv", ".html",
           ".css", ".yml", ".yaml", ".toml", ".ini", ".sh", ".sql", ".xml",
           ".rs", ".go", ".java", ".c", ".h", ".cpp", ".rb", ".php", ".log"}
MAX_UPLOAD = 10 * 1024 * 1024


@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 conversation_id: str = Query("")) -> dict:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD // 1024 // 1024} MB)")

    name = Path(file.filename or "upload").name
    suffix = Path(name).suffix.lower()
    dest = settings.upload_dir / f"{abs(hash(name + str(len(raw)))):x}{suffix}"

    if suffix in TEXTUAL or not suffix:
        text = raw.decode("utf-8", errors="replace")
    elif suffix == ".pdf":
        text = _extract_pdf(raw)
    else:
        raise HTTPException(
            415, f"Cannot read '{suffix}' files as text. Supported: "
                 f"{', '.join(sorted(TEXTUAL))} and .pdf")

    dest.write_text(text, encoding="utf-8")
    excerpt = text[:1500]
    fid = store.add_file(name, str(dest), file.content_type or "", len(raw),
                         excerpt, conversation_id or None)
    return {"id": fid, "filename": name, "size": len(raw),
            "characters": len(text), "excerpt": excerpt,
            "truncated": len(text) > len(excerpt)}


def _extract_pdf(raw: bytes) -> str:
    # BaseException, not Exception, and that is deliberate. pypdf imports
    # `cryptography`, whose Rust bindings raise pyo3's PanicException on a
    # broken install — and PanicException subclasses BaseException specifically
    # so it dodges ordinary handlers. Catching only Exception here returns a
    # bare 500 with a Rust stack trace instead of a usable message. Scoped to
    # this one import so nothing else is swallowed.
    try:
        import io

        from pypdf import PdfReader
    except BaseException as e:
        raise HTTPException(
            415, f"PDF support is unavailable ({type(e).__name__}). Install it "
                 "with: pip install pypdf — every other supported file type "
                 "still works.")
    try:
        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        raise HTTPException(422, f"Could not read that PDF: {type(e).__name__}. "
                                 "It may be encrypted, corrupt, or scanned images "
                                 "with no text layer.")
    if not text.strip():
        raise HTTPException(
            422, "That PDF has no extractable text — it is probably scanned "
                 "images. OCR it first, then upload.")
    return text


# -- chat -------------------------------------------------------------------

@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    cid = req.conversation_id
    title_from = req.message.strip() or (
        req.attachments[0].filename if req.attachments else "")
    if not cid or not store.get_conversation(cid):
        cid = store.create_conversation(agent.derive_title(title_from), req.mode)
    else:
        conversation = store.get_conversation(cid)
        if conversation["title"] == "New conversation":
            store.update_conversation(cid, title=agent.derive_title(title_from))
        store.update_conversation(cid, mode=req.mode)

    # Regenerating: drop the old branch before adding the new turn.
    if req.regenerate_from:
        store.truncate_after(cid, req.regenerate_from)
        store.delete_message(req.regenerate_from)

    if req.message.strip() or req.attachments:
        meta = ({"attachments": [a.model_dump() for a in req.attachments]}
                if req.attachments else None)
        store.add_message(cid, "user", req.message, meta)

    history = store.get_messages(cid)
    provider, notes = await providers.resolve(
        req.provider or settings.provider,
        () if req.provider else settings.fallback_chain)

    async def event_stream():
        # The conversation id must reach the client before anything else, or a
        # brand-new chat cannot be updated in place.
        yield f"data: {json.dumps({'type': 'conversation', 'data': {'id': cid}})}\n\n"
        try:
            async for event in agent.run(provider, cid, history, req.mode,
                                         settings.user_name, notes,
                                         strategy=req.strategy):
                yield f"data: {json.dumps({'type': event.type, 'data': event.data})}\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            payload = {"type": "error", "data": {"message": f"{type(e).__name__}: {e}"}}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"})


if __name__ == "__main__":
    import uvicorn

    print(f"\n  {BRAND.name} v{__version__} — {BRAND.tagline}")
    print(f"  provider: {settings.provider}"
          + (f" (fallback: {', '.join(settings.fallback_chain)})"
             if settings.fallback_chain else ""))
    print(f"  tools:    {len(toolkit.available(settings))} enabled")
    print(f"  db:       {settings.db_path}")
    print(f"\n  →  http://{settings.host}:{settings.port}\n")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")
