"""
Model backends.

One interface, seven implementations. The app never learns which one is in use,
so swapping the brain is a config change rather than a refactor. That separation
is the most valuable structural decision in an LLM product: models turn over
every few months and you do not want that churn reaching your application code.

    demo       scripted responses. No key, no network, no model. Exists so the
               product runs the moment you clone it, and so the tool loop can be
               tested deterministically.
    ollama     open models on your own machine. Free, offline, private.
    groq       hosted open models, free tier, very fast.
    gemini     Google AI Studio, free tier.
    openai     any OpenAI-compatible endpoint: LM Studio, llama.cpp, vLLM,
               OpenRouter, Together. One provider covers a dozen services.
    anthropic  Claude. Paid, and the strongest option if you want one.
    scratch    the model you trained yourself in scratch/. Will produce nonsense.
               Included because it is yours.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

import httpx

Message = dict[str, str]

ROOT = Path(__file__).resolve().parent.parent


class ProviderError(RuntimeError):
    """Raised when a backend cannot serve a request. Triggers the fallback chain."""


class Provider(ABC):
    name = "provider"
    label = "Provider"
    free = False

    @abstractmethod
    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # makes this an async generator for type checkers

    async def health(self) -> tuple[bool, str]:
        return True, "ok"

    @staticmethod
    def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
        """Most APIs want the system prompt separate from the turns."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        return system, rest


# ---------------------------------------------------------------------------

class DemoProvider(Provider):
    """
    A scripted backend so the product works with zero setup.

    It is not a language model and never pretends to be — it pattern-matches the
    last user message and streams a canned reply. Two reasons it earns its place:
    you can explore the entire interface before choosing a model, and the tool
    loop has a deterministic backend to be tested against, which a real model can
    never provide.
    """

    name = "demo"
    label = "Demo (no setup)"
    free = True

    SCRIPTS: list[tuple[tuple[str, ...], str]] = [
        (("hello", "hi ", "hey", "who are you", "what are you"),
         "I'm {brand} — {owner}'s assistant. Right now I'm running the **demo "
         "backend**, which means I'm a script, not a model: I pattern-match your "
         "message and read back a prepared answer.\n\nEverything around me is "
         "real — the streaming, the ten modes, memory, the tool loop, "
         "conversation history. Only the intelligence is missing.\n\n"
         "**Next move:** install [Ollama](https://ollama.com), run "
         "`ollama pull llama3.2`, set `PROVIDER=ollama` in `.env`, and restart. "
         "Free, offline, and actually smart."),
        (("calculate", "what is ", "how much", "*", "+", "solve"),
         "Watch — I'll use a tool rather than guessing at arithmetic:\n\n"
         "<tool_call>\n{{\"name\": \"calculate\", \"args\": {{\"expression\": "
         "\"1920*1080\"}}}}\n</tool_call>"),
        (("time", "date", "today", "what day"),
         "<tool_call>\n{{\"name\": \"current_time\", \"args\": {{}}}}\n</tool_call>"),
        (("remember", "my name is", "i prefer", "i like"),
         "<tool_call>\n{{\"name\": \"remember\", \"args\": {{\"content\": "
         "\"Tried the demo backend and tested memory\", \"category\": "
         "\"fact\"}}}}\n</tool_call>"),
        (("idea", "startup", "business", "should i build"),
         "In **Founder mode** I'd push back on this properly: who specifically "
         "needs it, what do they do today instead, and what is the cheapest "
         "experiment that could kill the idea this week?\n\nBut I'm the demo "
         "script — I can show you the shape of the answer, not the answer. "
         "Connect a real model and ask again."),
    ]

    DEFAULT = (
        "The **demo backend** is a script, so I can't actually answer that.\n\n"
        "What is real in front of you: streaming, ten operating modes, "
        "persistent memory, {n} tools, conversation history, file uploads.\n\n"
        "**Start with:** `ollama pull llama3.2`, then set `PROVIDER=ollama` "
        "in `.env`. Free, offline, no account. Or use Groq's free tier for "
        "something much stronger."
    )

    async def health(self) -> tuple[bool, str]:
        return True, "demo backend — scripted replies, no model attached"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        from .config import BRAND
        from . import tools as toolkit

        last = next((m["content"] for m in reversed(messages)
                     if m["role"] == "user"), "")

        # A tool result just came back. Conclude with it rather than calling the
        # same tool again — a real model reads the result and moves on, and the
        # demo has to behave the same way or the loop never terminates.
        if "<tool_result" in last:
            body = last.split(">", 1)[-1].split("</tool_result>")[0].strip()
            body = body.replace("[OK]", "").replace("[ERROR]", "").strip()
            reply = (f"The tool came back with: **{body.splitlines()[0][:160]}**\n\n"
                     f"That is a real computed result, not a predicted one — which is "
                     f"the entire point of the tool layer. Connect a real model and it "
                     f"will use these same tools to answer properly.")
            for word in reply.split(" "):
                yield word + " "
                await asyncio.sleep(0.012)
            return

        lowered = last.lower()

        # Any question carrying two or more numbers is an arithmetic question,
        # whatever words surround it. Build the tool call from the user's own
        # numbers rather than a canned expression — a demo that only fires on
        # the magic word "calculate" demonstrates nothing about the tool layer.
        pair = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*(?:x|×|\*|by)\s*(\d[\d,]*(?:\.\d+)?)",
                         lowered)
        if pair:
            operands = [pair.group(1), pair.group(2)]
        else:
            found = [n.replace(",", "") for n in
                     re.findall(r"\d[\d,]*(?:\.\d+)?", lowered)]
            # Largest two, so "4K" in the prose loses to the real dimensions.
            operands = sorted(found, key=lambda n: float(n or 0), reverse=True)[:2]

        if len(operands) >= 2 and not any(w in lowered for w in
                                          ("remember", "my name is")):
            expression = "{}*{}".format(*[o.replace(",", "") for o in operands])
            reply = ("Arithmetic goes to a tool — I'd only be predicting plausible "
                     "digits otherwise:\n\n"
                     f'<tool_call>\n{{"name": "calculate", "args": '
                     f'{{"expression": "{expression}"}}}}\n</tool_call>')
            for word in reply.split(" "):
                yield word + " "
                await asyncio.sleep(0.014)
            return

        reply = self.DEFAULT.format(n=len(toolkit.REGISTRY))
        for triggers, script in self.SCRIPTS:
            if any(t in lowered for t in triggers):
                reply = script.format(brand=BRAND.name, owner=BRAND.owner)
                break

        # Stream word by word so the UI's typing behaviour is exercised honestly.
        for word in reply.split(" "):
            yield word + " "
            await asyncio.sleep(random.uniform(0.012, 0.035))


class OllamaProvider(Provider):
    """Open models on your own hardware. The best free option by a distance."""

    name = "ollama"
    label = "Ollama (local)"
    free = True

    def __init__(self) -> None:
        self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
        self.host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

    async def health(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.host}/api/tags")
                r.raise_for_status()
                names = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return False, f"Ollama unreachable at {self.host}. Is it running?"
        if not names:
            return False, "Ollama is running but has no models. Try: ollama pull llama3.2"
        if not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
            return False, f"Model '{self.model}' not pulled. Have: {', '.join(names[:5])}"
        return True, f"{self.model} · local"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "options": {"temperature": 0.7}}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{self.host}/api/chat",
                                         json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            return
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama: {type(e).__name__}") from e


class OpenAICompatibleProvider(Provider):
    """
    Any endpoint that speaks the OpenAI chat-completions protocol.

    That is most of them: LM Studio, llama.cpp's server, vLLM, OpenRouter,
    Together, DeepSeek, Fireworks, and Groq. One implementation, a dozen
    services, because they all copied the same wire format.
    """

    name = "openai"
    label = "OpenAI-compatible"

    def __init__(self, base_url: str = "", api_key: str = "",
                 model: str = "", label: str = "") -> None:
        self.base = (base_url or os.getenv("OPENAI_BASE_URL",
                                           "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if label:
            self.label = label

    async def health(self) -> tuple[bool, str]:
        if not self.api_key and "localhost" not in self.base and "127.0.0.1" not in self.base:
            return False, f"No API key set for {self.base}"
        return True, f"{self.model} · {self.base}"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "messages": messages, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{self.base}/chat/completions",
                                         json=payload, headers=headers) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode()[:200]
                        raise ProviderError(f"HTTP {r.status_code}: {body}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        body = line[6:].strip()
                        if body == "[DONE]":
                            return
                        try:
                            delta = json.loads(body)["choices"][0].get("delta", {})
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                        if delta.get("content"):
                            yield delta["content"]
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.label}: {type(e).__name__}") from e


class GroqProvider(OpenAICompatibleProvider):
    """Hosted open models, free tier, and the fastest inference available."""

    name = "groq"
    label = "Groq (free tier)"
    free = True

    def __init__(self) -> None:
        super().__init__(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY", ""),
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        )

    async def health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GROQ_API_KEY not set. Free key: console.groq.com"
        return True, f"{self.model} · Groq"


class GeminiProvider(Provider):
    """Google AI Studio's free tier."""

    name = "gemini"
    label = "Gemini (free tier)"
    free = True

    def __init__(self) -> None:
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    async def health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GEMINI_API_KEY not set. Free key: aistudio.google.com/apikey"
        return True, f"{self.model} · Google AI Studio"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        if not self.api_key:
            raise ProviderError("GEMINI_API_KEY is not set")
        system, turns = self.split_system(messages)
        payload: dict = {
            "contents": [
                {"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]}
                for m in turns
            ]
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:streamGenerateContent?alt=sse&key={self.api_key}")
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode()[:200]
                        raise ProviderError(f"HTTP {r.status_code}: {body}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        for cand in data.get("candidates", []):
                            for part in cand.get("content", {}).get("parts", []):
                                if part.get("text"):
                                    yield part["text"]
        except httpx.HTTPError as e:
            raise ProviderError(f"Gemini: {type(e).__name__}") from e


class AnthropicProvider(Provider):
    """
    Claude, via the official SDK.

    Paid, unlike everything else here, and worth knowing about: if you want the
    strongest reasoning available behind your own product, this is it. Adaptive
    thinking is on, and server-side refusal fallbacks are enabled so a declined
    request routes to another model instead of failing the turn.
    """

    name = "anthropic"
    label = "Claude"

    def __init__(self) -> None:
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))

    async def health(self) -> tuple[bool, str]:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "The 'anthropic' package is not installed. pip install anthropic"
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY not set. console.anthropic.com"
        return True, f"{self.model} · Anthropic"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:
            raise ProviderError("pip install anthropic") from e

        system, turns = self.split_system(messages)
        client = AsyncAnthropic(api_key=self.api_key or None)
        try:
            async with client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or None,
                messages=turns,
                thinking={"type": "adaptive"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise ProviderError(f"Anthropic: {type(e).__name__}: {e}") from e


class ScratchProvider(Provider):
    """
    The model trained in scratch/, wired in as a first-class backend.

    It will produce nonsense. That is not a bug and it is worth seeing: the same
    interface, the same modes, the same tools, the same streaming — and 800k
    parameters instead of 70 billion. It shows precisely how much of "an AI
    assistant" is scaffolding and how much is the model.
    """

    name = "scratch"
    label = "Your own model"
    free = True

    def __init__(self) -> None:
        self.dir = ROOT / "scratch" / "out"
        self._model = None
        self._tok = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import sys
        scratch = str(self.dir.parent)
        if scratch not in sys.path:
            sys.path.insert(0, scratch)
        import torch
        from model import GPT, GPTConfig
        from tokenizer import CharTokenizer

        ckpt_path = self.dir / "model.pt"
        if not ckpt_path.exists():
            raise ProviderError(
                "No trained model. Run: python scratch/prepare_data.py && "
                "python scratch/train.py")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        m = GPT(GPTConfig(**ckpt["config"]))
        m.load_state_dict(ckpt["model"])
        m.eval()
        self._model, self._tok, self._torch = m, CharTokenizer.load(
            self.dir / "tokenizer.json"), torch

    async def health(self) -> tuple[bool, str]:
        try:
            self._load()
        except Exception as e:
            return False, str(e)
        return True, f"{self._model.num_params():,} params · trained by you"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        self._load()
        torch = self._torch
        import torch.nn.functional as F

        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}"
                               for m in messages if m["role"] != "system")
        ids = self._tok.encode((transcript + "\nASSISTANT:")[-1000:])
        idx = torch.tensor([ids or self._tok.encode("\n")], dtype=torch.long)

        for _ in range(400):
            cond = idx[:, -self._model.config.block_size:]
            with torch.no_grad():
                logits, _ = self._model(cond)
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, min(40, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, nxt), dim=1)
            yield self._tok.decode([nxt.item()])
            await asyncio.sleep(0)  # let the event loop breathe


PROVIDERS: dict[str, type[Provider]] = {
    "demo": DemoProvider,
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "openai": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
    "scratch": ScratchProvider,
}


def get_provider(name: str | None = None) -> Provider:
    key = (name or os.getenv("PROVIDER", "demo")).lower()
    cls = PROVIDERS.get(key)
    if cls is None:
        raise ProviderError(f"Unknown provider '{key}'. "
                            f"Options: {', '.join(PROVIDERS)}")
    return cls()


def from_spec(spec: str) -> Provider:
    """Build a provider from a "backend:model" string, e.g. "groq:llama-3.3-70b".

    The council needs several differently-configured providers alive at once,
    which get_provider() cannot express because it reads one model per backend
    out of the environment."""
    spec = spec.strip()
    if not spec:
        raise ProviderError("Empty provider spec")
    backend, _, model = spec.partition(":")
    provider = get_provider(backend.strip())
    if model.strip():
        # Every backend keeps its model on `.model`; scratch and demo have none.
        if hasattr(provider, "model"):
            provider.model = model.strip()
        else:
            raise ProviderError(f"'{backend}' does not take a model name")
    return provider


def describe(provider: Provider) -> str:
    """A stable label for one configured provider, for display and dedupe."""
    model = getattr(provider, "model", "")
    return f"{provider.name}:{model}" if model else provider.name


def catalogue() -> list[dict]:
    out = []
    for key, cls in PROVIDERS.items():
        out.append({"key": key, "label": cls.label, "free": cls.free})
    return out


async def resolve(primary: str, chain: tuple[str, ...] = ()) -> tuple[Provider, list[str]]:
    """Return the first healthy provider, and a note of what was skipped.

    A fallback chain matters more here than it looks: local models go down when
    you close the laptop lid, free tiers hit their quota, and an assistant that
    dies rather than degrading is a worse product than one that says "Ollama is
    down, I'm on Groq"."""
    notes: list[str] = []
    for name in (primary, *chain):
        try:
            provider = get_provider(name)
        except ProviderError as e:
            notes.append(str(e))
            continue
        ok, detail = await provider.health()
        if ok:
            return provider, notes
        notes.append(f"{name}: {detail}")
    return get_provider(primary), notes
