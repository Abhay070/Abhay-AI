# app — the chat product

A real AI assistant: streaming responses, conversation memory, your own
personality, and a swappable brain.

```bash
pip install -r ../requirements-app.txt
cp .env.example ../.env      # then edit it
python server.py             # http://localhost:8000
```

## Choosing a brain

Set `PROVIDER` in `.env`. Every option is free.

| Provider | Setup | Notes |
|---|---|---|
| `ollama` | [ollama.com](https://ollama.com), then `ollama pull llama3.2` | **Start here.** Free forever, offline, private. |
| `groq` | free key at [console.groq.com](https://console.groq.com) | Hosted, very fast, generous free tier, no card |
| `gemini` | free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier |
| `scratch` | train stage 1 first | Your own model. Will produce nonsense — that is the point. |

The status dot in the header tells you whether your backend is actually reachable,
so a missing key is a clear message rather than a mystery failure on first send.

## Files

| File | What it is |
|---|---|
| `server.py` | Three endpoints, ~120 lines. Most of what a chat backend is. |
| `providers.py` | One interface, four backends. Swap models without touching the app. |
| `static/index.html` | The whole UI — no build step, no framework, no dependencies |

## The two ideas worth taking away

**Models are stateless.** Yours has no memory between requests. The entire
illusion of a conversation is `server.py` resending the full history every time.
Once you have seen that, chat interfaces stop being mysterious.

**Personality is a string.** `SYSTEM_PROMPT` in your `.env`. It is the highest
leverage line in the project — changing it changes who your assistant is more
than any code you could write.

## Where to take it next

- **Persistence** — conversations vanish on refresh. Add SQLite.
- **Tool use** — let the model call functions. This is what makes it an agent
  rather than a chat box, and it is mostly plumbing.
- **RAG** — feed it your own documents at query time.
- **Streaming markdown** — the renderer here is deliberately minimal and escapes
  everything first.
