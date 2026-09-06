"""
Pluggable model backends.

The app does not care where tokens come from. Every backend implements one
method -- stream a reply given a conversation -- so you can swap the brain
without touching the server or the UI. That separation is the single most
valuable architectural decision in an AI app: models change every few months,
and you do not want that churn reaching into your product code.

All four options below are free:

  scratch  your own model from scratch/, running locally. Free forever, offline,
           and not remotely smart. Included because it is yours.
  ollama   open-weights models (Llama, Qwen, Mistral, Gemma) on your own machine.
           Free, offline, private, genuinely useful. Install from ollama.com.
  groq     hosted open models, generous free tier, very fast. Needs a free key.
  gemini   Google's free API tier. Needs a free key.

Set PROVIDER in your .env to pick one.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

import httpx

Message = dict[str, str]  # {"role": "user"|"assistant"|"system", "content": str}


class Provider(ABC):
    """One method. Everything else is an implementation detail."""

    name: str = "provider"

    @abstractmethod
    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Yield chunks of the reply as they are produced."""
        raise NotImplementedError
        yield ""  # pragma: no cover - makes this an async generator for type checkers

    async def health(self) -> tuple[bool, str]:
        """Is this backend actually usable right now? Used by /api/health."""
        return True, "ok"


class ScratchProvider(Provider):
    """
    Your own model, trained in scratch/.

    It has no concept of roles or instructions -- it is a character-level
    continuation engine. So we flatten the conversation into a transcript and let
    it continue. The output will be nonsense-shaped-like-your-training-data, and
    that is the honest, correct behaviour for a model this size. Keeping it behind
    the same interface as the others is the point: you can see exactly how much
    of "an AI assistant" is the model and how much is the scaffolding around it.
    """

    name = "scratch"

    def __init__(self, out_dir: str | None = None):
        self.out_dir = Path(out_dir or Path(__file__).resolve().parent.parent / "scratch" / "out")
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import sys

        scratch_dir = self.out_dir.parent
        if str(scratch_dir) not in sys.path:
            sys.path.insert(0, str(scratch_dir))

        import torch
        from model import GPT, GPTConfig
        from tokenizer import CharTokenizer

        ckpt_path = self.out_dir / "model.pt"
        if not ckpt_path.exists():
            raise RuntimeError(
                f"No trained model at {ckpt_path}. Train one first:\n"
                f"  python scratch/prepare_data.py && python scratch/train.py"
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = GPT(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        model.eval()
        self._model = model
        self._tokenizer = CharTokenizer.load(self.out_dir / "tokenizer.json")
        self._torch = torch

    async def health(self) -> tuple[bool, str]:
        try:
            self._load()
            return True, f"local model, {self._model.num_params():,} params"
        except Exception as e:
            return False, str(e)

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        self._load()
        torch = self._torch
        import torch.nn.functional as F

        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
        )
        transcript += "\nASSISTANT:"

        ids = self._tokenizer.encode(transcript[-1000:]) or self._tokenizer.encode("\n")
        idx = torch.tensor([ids], dtype=torch.long)

        for _ in range(300):
            idx_cond = idx[:, -self._model.config.block_size:]
            with torch.no_grad():
                logits, _ = self._model(idx_cond)
            logits = logits[:, -1, :] / 0.8
            v, _ = torch.topk(logits, min(40, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            yield self._tokenizer.decode([next_token.item()])


class OllamaProvider(Provider):
    """
    Open-weights models running on your own hardware. The best free option.

        1. install from https://ollama.com
        2. ollama pull llama3.2        (2 GB, runs on 8 GB RAM)
        3. set PROVIDER=ollama in .env

    No key, no quota, no network, nothing leaves your machine.
    """

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    async def health(self) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.host}/api/tags")
                r.raise_for_status()
                names = [m["name"] for m in r.json().get("models", [])]
            if not names:
                return False, "Ollama is running but has no models. Try: ollama pull llama3.2"
            if not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
                return False, f"Model '{self.model}' not pulled. Available: {', '.join(names)}"
            return True, f"{self.model} via Ollama"
        except Exception:
            return False, f"Ollama not reachable at {self.host}. Is it running?"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        payload = {"model": self.model, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{self.host}/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break


class GroqProvider(Provider):
    """
    Hosted open models on a free tier, and extremely fast.
    Get a key at https://console.groq.com -- no card required.
    """

    name = "groq"
    BASE = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    async def health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GROQ_API_KEY not set. Free key at https://console.groq.com"
        return True, f"{self.model} via Groq"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        payload = {"model": self.model, "messages": messages, "stream": True}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.BASE, json=payload, headers=headers) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    delta = json.loads(body)["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]


class GeminiProvider(Provider):
    """
    Google's free API tier. Get a key at https://aistudio.google.com/apikey
    """

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    async def health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "GEMINI_API_KEY not set. Free key at https://aistudio.google.com/apikey"
        return True, f"{self.model} via Google AI Studio"

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        # Gemini keeps the system prompt separate and calls the assistant "model".
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        payload: dict = {"contents": contents}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        )
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = json.loads(line[6:])
                    for cand in data.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            if part.get("text"):
                                yield part["text"]


PROVIDERS: dict[str, type[Provider]] = {
    "scratch": ScratchProvider,
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str | None = None) -> Provider:
    name = (name or os.getenv("PROVIDER", "ollama")).lower()
    if name not in PROVIDERS:
        raise SystemExit(
            f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[name]()
