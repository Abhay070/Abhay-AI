"""
The chat server.

Three endpoints and about 120 lines. This is genuinely most of what a chat
product is on the server side:

    GET  /             the chat UI
    GET  /api/health   which backend is configured, and is it working
    POST /api/chat     stream a reply

Run it:

    pip install -r requirements-app.txt
    cp app/.env.example .env      # then edit it
    python app/server.py

Then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from providers import get_provider

HERE = Path(__file__).parent

# Load .env by hand so the project has no dependency on python-dotenv.
for env_file in (HERE.parent / ".env", HERE / ".env"):
    if env_file.exists():
        for raw in env_file.read_text().splitlines():
            raw = raw.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

# This is your AI's personality. It is a plain string, and changing it changes
# who your assistant is more than almost anything else you can do. Every product
# you have used -- Claude, ChatGPT, Gemini -- has a much longer version of this.
DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are Abhay-AI, a helpful and direct assistant. "
    "You give concrete, specific answers and admit when you do not know something. "
    "You never pad your responses with filler.",
)

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

app = FastAPI(title="Abhay-AI")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    system: str | None = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict:
    """Tells the UI which backend is live, so a missing key is a clear message
    rather than a mysterious failure on first send."""
    provider = get_provider()
    ok, detail = await provider.health()
    return {"provider": provider.name, "ok": ok, "detail": detail}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    Stream a reply using Server-Sent Events.

    Two things worth noticing, because they are the whole trick behind every
    chat assistant:

    1. The model is stateless. It has no memory between requests. The illusion
       of a conversation comes from us resending the entire history every single
       time. That is it. That is the whole mechanism.

    2. Because history grows without bound and context windows do not, we trim
       to the most recent MAX_HISTORY turns. Production systems do something
       smarter here -- summarizing older turns, or retrieving only the relevant
       ones -- but they are solving this same problem.
    """
    provider = get_provider()

    history = [m.model_dump() for m in req.messages][-MAX_HISTORY:]
    messages = [{"role": "system", "content": req.system or DEFAULT_SYSTEM_PROMPT}] + history

    async def event_stream():
        try:
            async for chunk in provider.stream(messages):
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'{type(e).__name__}: {e}'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    provider = get_provider()
    print(f"\n  Abhay-AI  |  provider: {provider.name}")
    print("  http://localhost:8000\n")
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
