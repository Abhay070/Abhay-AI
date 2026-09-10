"""
List the models your configured backend actually offers, right now.

    python scripts/list_models.py              # whichever PROVIDER is in .env
    python scripts/list_models.py groq
    python scripts/list_models.py ollama

Why this exists: hosted providers retire and rename models constantly. A
model name that worked last month returns

    404 — the model X does not exist or you do not have access to it

which reads like a broken key but is not. Guessing the new name from
documentation is unreliable; asking your own account is not. This prints
exactly what YOUR key can reach, and the line to paste into .env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx                                              # noqa: E402

from praxis.config import settings                        # noqa: E402

GREEN, YELLOW, RED, DIM, OFF = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"

# Which env var holds the key, and where the catalogue lives.
BACKENDS = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1/models",
             "GROQ_MODEL", "console.groq.com/keys"),
    "openai": ("OPENAI_API_KEY", None, "OPENAI_MODEL", None),
    "gemini": ("GEMINI_API_KEY", None, "GEMINI_MODEL",
               "aistudio.google.com/apikey"),
    "ollama": (None, None, "OLLAMA_MODEL", "ollama.com"),
}


def openai_style(url: str, key: str) -> list[str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = httpx.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return sorted(m["id"] for m in response.json().get("data", []))


def gemini_models(key: str) -> list[str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    response = httpx.get(url, timeout=20)
    response.raise_for_status()
    out = []
    for m in response.json().get("models", []):
        # Only models that can actually answer a chat turn.
        if "generateContent" in m.get("supportedGenerationMethods", []):
            out.append(m["name"].removeprefix("models/"))
    return sorted(out)


def ollama_models(host: str) -> list[str]:
    response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=10)
    response.raise_for_status()
    return sorted(m["name"] for m in response.json().get("models", []))


def main() -> int:
    backend = (sys.argv[1] if len(sys.argv) > 1 else settings.provider).lower()
    if backend not in BACKENDS:
        print(f"{RED}Unknown backend {backend!r}.{OFF} "
              f"Try one of: {', '.join(BACKENDS)}")
        return 1

    key_var, url, model_var, where = BACKENDS[backend]
    key = os.getenv(key_var, "") if key_var else ""

    if key_var and not key:
        print(f"{RED}{key_var} is not set.{OFF}")
        if where:
            print(f"Get a free key at {where}, then put it in .env")
        return 1

    print(f"Models your {backend} account can reach right now:\n")
    try:
        if backend == "groq":
            models = openai_style(url, key)
        elif backend == "openai":
            base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            models = openai_style(f"{base.rstrip('/')}/models", key)
        elif backend == "gemini":
            models = gemini_models(key)
        else:
            models = ollama_models(os.getenv("OLLAMA_HOST",
                                             "http://localhost:11434"))
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        if code in (401, 403):
            print(f"{RED}Your key was rejected ({code}).{OFF} "
                  f"Check {key_var} in .env — no quotes, no spaces.")
        else:
            print(f"{RED}HTTP {code}{OFF} from the provider.")
        return 1
    except httpx.HTTPError as e:
        print(f"{RED}Could not reach {backend} ({type(e).__name__}).{OFF}")
        if backend == "ollama":
            print("Is the Ollama app running? Look for it in your system tray.")
        return 1

    if not models:
        print(f"{YELLOW}The account is reachable but lists no models.{OFF}")
        if backend == "ollama":
            print("Pull one first:  ollama pull llama3.2")
        return 1

    current = os.getenv(model_var, "")
    # Chat-capable models first; the rest (speech, embeddings) are noise here.
    chat, other = [], []
    for m in models:
        lowered = m.lower()
        # Orpheus is text-to-speech and Groq lists it beside chat models with
        # nothing in the name to say so. Without it here, the ranking below
        # cheerfully recommended a speech model as a chat backend.
        if any(w in lowered for w in ("whisper", "tts", "embed", "guard",
                                      "moderation", "vision-preview",
                                      "orpheus", "speech", "audio", "rerank")):
            other.append(m)
        else:
            chat.append(m)

    for m in chat:
        mark = f"  {GREEN}← currently set{OFF}" if m == current else ""
        print(f"  {m}{mark}")
    if other:
        print(f"\n{DIM}  not for chat: {', '.join(other)}{OFF}")

    if current and current not in models:
        print(f"\n{YELLOW}Your .env asks for {current!r}, which is not in that "
              f"list.{OFF}\nThat is the cause of a 404 'model does not exist'.")

    if chat:
        # Prefer a large instruct-style model as the suggestion.
        # Rank by parameter count where the name states one, then by
        # instruction-tuned markers. Falling back to name LENGTH — as this
        # once did — is meaningless and picked a speech model.
        def size(name: str) -> int:
            import re
            match = re.search(r"(\d+)\s*b\b", name.lower())
            return int(match.group(1)) if match else 0

        def general(name: str) -> bool:
            """Penalise models built for one language or one narrow task."""
            lowered = name.lower()
            return not any(t in lowered for t in
                           ("arabic", "saudi", "allam", "coder", "math",
                            "mini", "small"))

        pick = max(chat, key=lambda m: (
            general(m),
            size(m),
            any(t in m.lower() for t in ("instruct", "versatile", "chat", "oss")),
        ))
        print(f"\n{GREEN}To use one, put this in your .env:{OFF}")
        print(f"  {model_var}={pick}")

        # A thinking model streams its scratchpad in a separate field and
        # charges you for it. On a free tier that is the difference between an
        # answer and an HTTP 429 — and if the thinking runs long enough, the
        # answer field never arrives at all and the reply comes back blank.
        from praxis.providers import REASONING_MODELS
        if REASONING_MODELS.search(pick):
            print(f"  REASONING_EFFORT=low")
            print(f"\n{YELLOW}{pick} thinks before it answers.{OFF} Those thinking "
                  f"tokens\ncount against a free tier's per-minute allowance and "
                  f"can eat the whole\noutput budget. REASONING_EFFORT=low keeps "
                  f"it brief. Raise it to\nmedium or high once you are on a paid "
                  f"tier.")
        print(f"\n{DIM}Then restart:  python server.py{OFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
