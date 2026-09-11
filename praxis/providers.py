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
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

import httpx

from . import limits

Message = dict[str, str]

ROOT = Path(__file__).resolve().parent.parent


class ProviderError(RuntimeError):
    """Raised when a backend cannot serve a request. Triggers the fallback chain."""


class RateLimited(ProviderError):
    """A quota was hit. Carries which one, and when it lifts.

    Separate from ProviderError because the correct response is different:
    a rate limit is not a broken backend, it is a working one asking you to
    wait. Free tiers hit this constantly — Groq's on-demand tier allows 8,000
    tokens per minute — and an assistant that gives up on the first 429 is
    unusable on exactly the tier most people start on.

    `window` is the field that matters. A per-minute refusal is worth waiting
    out; a daily one is not, and treating the two alike is how a seven-hour
    cooldown became a 45-second wait, three wasted retries, and half the
    conversation deleted to make a request smaller that was never too big.

    Every new field is keyword-only with a default, so existing raise sites and
    every `except RateLimited` keep working untouched."""

    def __init__(self, message: str, retry_after: float = 0.0, *,
                 provider: str = "", model: str = "", budget_key: str = "",
                 window: str = "unknown", scope: str = "unknown",
                 limit: int = 0, remaining: int = -1, reset_at: float = 0.0,
                 source: str = "inferred") -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.provider = provider
        self.model = model
        self.budget_key = budget_key
        self.window = window
        self.scope = scope
        self.limit = limit
        self.remaining = remaining
        # Absolute epoch, never a delta: this gets persisted and compared after
        # a restart, where "600 seconds from now" would mean nothing.
        self.reset_at = reset_at
        self.source = source

    @property
    def is_daily(self) -> bool:
        """Waiting cannot fix this one."""
        return self.window in ("day", "hour")

    @classmethod
    def from_response(cls, label: str, body: str, headers, *, provider: str = "",
                      model: str = "", budget_key: str = "") -> "RateLimited":
        fact = limits.classify(body, headers)
        return cls(f"{label} rate limit: {_tidy(body)}", fact.retry_after,
                   provider=provider, model=model, budget_key=budget_key,
                   window=fact.window, scope=fact.scope, limit=fact.limit,
                   remaining=fact.remaining, reset_at=fact.reset_at,
                   source=fact.source)


class EmptyAnswer(ProviderError):
    """The request succeeded and produced no answer.

    This has its own class because it used to be invisible. A reasoning model
    streams its thinking in one field and its answer in another; when the
    thinking consumes the whole output budget, the answer field never arrives,
    the stream closes cleanly, and the product stores an empty string. The user
    sees a blank reply and no error — the worst possible failure, because
    nothing anywhere says what went wrong."""


# Models that think before they answer. They stream chain-of-thought in a
# separate field (`reasoning` / `reasoning_content`) and can spend an entire
# output budget on it. Matched by name because no OpenAI-compatible endpoint
# advertises the capability in its catalogue.
REASONING_MODELS = re.compile(
    r"gpt-oss|(^|[-/])o[1-4]($|[-.])|deepseek-r1|qwen3|magistral|"
    r"thinking|reasoner|-r1", re.IGNORECASE)

# "Please try again in 7.482s" — Groq puts the wait in the body, not only in
# the Retry-After header.
_RETRY_HINT = re.compile(r"try again in ([\d.]+)\s*(ms|s|m)?", re.IGNORECASE)


def _retry_after(headers, body: str) -> float:
    """How long the provider asked us to wait. 0 when it did not say."""
    raw = headers.get("retry-after", "")
    try:
        if raw:
            return float(raw)
    except (TypeError, ValueError):
        pass
    match = _RETRY_HINT.search(body)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = (match.group(2) or "s").lower()
    return value / 1000 if unit == "ms" else value * 60 if unit == "m" else value


def _tidy(body: str) -> str:
    """The human sentence out of a provider's JSON error, or the raw body.

    Providers wrap one useful sentence in three layers of envelope. Showing the
    envelope to the user makes a solvable problem look like a crash."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return body.strip()[:240]
    error = data.get("error", data) if isinstance(data, dict) else {}
    if isinstance(error, dict):
        return str(error.get("message") or error)[:240]
    return str(error)[:240]


def _rejected_param(body: str) -> str:
    """Which request parameter an endpoint just refused, if it named one.

    OpenAI-compatible is a family resemblance, not a specification. `max_tokens`
    and `reasoning_effort` are both understood by most of it and rejected by
    some. When one is refused the fix is to drop it and try again, not to make
    the user edit .env to discover which of a dozen services they picked is the
    strict one."""
    if not re.search(r"unsupported|unrecogni[sz]ed|unknown|not supported|"
                     r"invalid|unexpected", body, re.IGNORECASE):
        return ""
    for name in ("stream_options", "reasoning_effort", "max_completion_tokens",
                 "max_tokens"):
        if name in body:
            return name
    return ""


# A health probe costs a network round trip, and the interface asks for one
# every time it loads. Cache briefly so opening Settings three times does not
# mean nine requests, while still noticing a backend that died a minute ago.
_HEALTH_TTL = 30.0
_health_cache: dict[str, tuple[float, tuple[bool, str]]] = {}


def _cached_health(key: str) -> tuple[bool, str] | None:
    hit = _health_cache.get(key)
    if hit and (time.monotonic() - hit[0]) < _HEALTH_TTL:
        return hit[1]
    return None


def _remember_health(key: str, result: tuple[bool, str]) -> tuple[bool, str]:
    _health_cache[key] = (time.monotonic(), result)
    return result


class Provider(ABC):
    name = "provider"
    label = "Provider"
    free = False
    # Set by the router. Called with LimitFact objects and usage dicts as they
    # are observed. None means nobody is listening, which is the case for the
    # demo backend, the scratch model, and every test that predates this.
    meter = None
    budget_key = ""

    def report(self, facts=None, usage=None) -> None:
        """Hand what a response revealed to whoever is keeping the ledger."""
        if self.meter is None:
            return
        try:
            self.meter(self, facts or [], usage or {})
        except Exception:
            # Accounting must never be able to break a turn.
            pass

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
        content_chars = thinking_chars = 0
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{self.host}/api/chat",
                                         json=payload) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        message = data.get("message", {})
                        # Local thinking models (deepseek-r1, qwen3) put the
                        # scratchpad here. Counted, never shown — same reasoning
                        # as the hosted case.
                        thinking_chars += len(message.get("thinking") or "")
                        chunk = message.get("content", "")
                        if chunk:
                            content_chars += len(chunk)
                            yield chunk
                        if data.get("done"):
                            break
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama: {type(e).__name__}") from e
        if content_chars == 0:
            raise EmptyAnswer(
                f"{self.model} returned no answer"
                + (f" — {thinking_chars:,} characters of internal reasoning and "
                   f"nothing else. Try a non-thinking model, or ask again."
                   if thinking_chars else ". The model produced no text."))


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
        # A ceiling on the answer, not on the conversation. It exists mostly to
        # stop a reasoning model from spending an unbounded budget thinking.
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
        # low | medium | high, or empty to leave the model's default alone.
        # Sent only to models that understand it, and dropped automatically if
        # an endpoint refuses it.
        #
        # Default medium, not low. Low was tempting because thinking tokens are
        # what blows a free tier's per-minute allowance — but a rate limit is
        # now waited out and shrunk rather than failed, whereas a model starved
        # of thinking is just worse at the questions that need it most, and no
        # amount of retrying fixes that. Set low if you would rather wait less
        # than think more.
        self.reasoning_effort = os.getenv("REASONING_EFFORT", "medium").strip().lower()
        if self.reasoning_effort in ("", "off", "none", "default"):
            self.reasoning_effort = ""
        if label:
            self.label = label

    def payload(self, messages: list[Message], drop: set[str]) -> dict:
        body: dict = {"model": self.model, "messages": messages, "stream": True,
                      "max_tokens": self.max_tokens,
                      # Real token counts in the final chunk. Dropped
                      # automatically by any endpoint that refuses it.
                      "stream_options": {"include_usage": True}}
        if self.reasoning_effort and REASONING_MODELS.search(self.model):
            body["reasoning_effort"] = self.reasoning_effort
        for key in drop:
            body.pop(key, None)
        return body

    async def health(self) -> tuple[bool, str]:
        """Actually reach the endpoint, rather than checking that a key exists.

        The previous version only inspected configuration, so a port with
        nothing listening on it reported healthy — which meant /api/council
        showed a green dot for a dead backend and the council wasted a slot on
        it every turn. A health check that cannot fail is not a health check."""
        local = "localhost" in self.base or "127.0.0.1" in self.base
        if not self.api_key and not local:
            return False, f"No API key set for {self.base}"

        key = f"{self.name}|{self.base}|{self.model}"
        cached = _cached_health(key)
        if cached:
            return cached

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                response = await client.get(f"{self.base}/models", headers=headers)
        except httpx.HTTPError as e:
            return _remember_health(
                key, (False, f"unreachable at {self.base} ({type(e).__name__})"))
        if response.status_code in (401, 403):
            return _remember_health(key, (False, "rejected the API key"))
        if response.status_code >= 400:
            return _remember_health(
                key, (False, f"HTTP {response.status_code} from {self.base}"))
        return _remember_health(key, (True, f"{self.model} · {self.base}"))

    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        """Stream the answer, and refuse to return silence.

        Three things here are not in the textbook version, and each one exists
        because it broke in the user's hands:

        `reasoning` is read but never yielded. A thinking model streams its
        chain of thought in that field and its answer in `content`. Yielding
        the thinking would put the model's scratchpad on screen as if it were
        the reply; ignoring the field entirely — the previous behaviour — meant
        that when the thinking used up the whole budget, the product stored an
        empty string and showed a blank bubble with no error anywhere.

        `finish_reason` is kept so that when nothing comes back we can say
        which of the several possible reasons it was.

        A run that yields no visible characters raises instead of returning.
        An empty answer is a failure, and a failure that reports success is the
        one bug a user cannot work around."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        drop: set[str] = set()

        while True:
            content_chars = 0
            thinking_chars = 0
            finish = ""
            retry_with_fewer_params = False

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream(
                            "POST", f"{self.base}/chat/completions",
                            json=self.payload(messages, drop),
                            headers=headers) as r:
                        # Every response carries the account's standing, not
                        # just the refusals. Reading it here is what lets the
                        # router see a quota running low instead of finding out
                        # by being turned away.
                        self.report(facts=limits.read_headers(r.headers))

                        if r.status_code >= 400:
                            body = (await r.aread()).decode()[:400]
                            refused = _rejected_param(body)
                            if refused and refused not in drop:
                                drop.add(refused)
                                retry_with_fewer_params = True
                            elif r.status_code == 429:
                                raise RateLimited.from_response(
                                    self.label, body, r.headers,
                                    provider=self.name, model=self.model,
                                    budget_key=self.budget_key)
                            elif r.status_code == 404 and "model" in body.lower():
                                raise ProviderError(
                                    f"{self.model!r} is not a model this account "
                                    f"can reach. Run  python scripts/list_models.py  "
                                    f"to see what is actually available.")
                            else:
                                raise ProviderError(
                                    f"HTTP {r.status_code}: {_tidy(body)}")
                        else:
                            async for line in r.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                body = line[6:].strip()
                                if body == "[DONE]":
                                    break
                                try:
                                    parsed = json.loads(body)
                                except json.JSONDecodeError:
                                    continue
                                # The final chunk carries real token counts
                                # when the endpoint supports it — measured
                                # beats four-characters-per-token every time.
                                measured = limits.read_usage(parsed)
                                if measured:
                                    self.report(usage=measured)
                                try:
                                    choice = parsed["choices"][0]
                                except (KeyError, IndexError):
                                    continue
                                finish = choice.get("finish_reason") or finish
                                delta = choice.get("delta") or {}
                                thinking = (delta.get("reasoning")
                                            or delta.get("reasoning_content") or "")
                                if thinking:
                                    thinking_chars += len(thinking)
                                if delta.get("content"):
                                    content_chars += len(delta["content"])
                                    yield delta["content"]
            except httpx.HTTPError as e:
                raise ProviderError(f"{self.label}: {type(e).__name__}") from e

            if retry_with_fewer_params:
                continue
            if content_chars == 0:
                raise EmptyAnswer(self.explain_silence(thinking_chars, finish))
            return

    def explain_silence(self, thinking_chars: int, finish: str) -> str:
        """Say precisely why an answer never arrived, and what to change."""
        if thinking_chars and finish == "length":
            return (f"{self.model} spent its entire {self.max_tokens}-token budget "
                    f"thinking ({thinking_chars:,} characters of reasoning) and "
                    f"never started the answer. Set REASONING_EFFORT=low in .env, "
                    f"or raise OPENAI_MAX_TOKENS.")
        if thinking_chars:
            return (f"{self.model} returned {thinking_chars:,} characters of "
                    f"internal reasoning and no answer. This usually means the "
                    f"question tripped its refusal path silently — try rephrasing, "
                    f"or a different model.")
        if finish == "length":
            return (f"{self.model} hit the {self.max_tokens}-token output limit "
                    f"before writing anything. Raise OPENAI_MAX_TOKENS.")
        if finish == "content_filter":
            return f"{self.label} blocked the answer with its content filter."
        return (f"{self.model} returned an empty response"
                + (f" (finish_reason: {finish})" if finish else "")
                + ". Nothing was wrong with the request; the model simply "
                  "produced no text.")


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
        ok, detail = await super().health()
        return ok, (f"{self.model} · Groq" if ok else f"Groq: {detail}")


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
        key = f"gemini|{self.model}"
        cached = _cached_health(key)
        if cached:
            return cached
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}?key={self.api_key}")
        try:
            async with httpx.AsyncClient(timeout=6) as client:
                response = await client.get(url)
        except httpx.HTTPError as e:
            return _remember_health(key, (False, f"unreachable ({type(e).__name__})"))
        if response.status_code in (400, 401, 403):
            return _remember_health(key, (False, "rejected the API key"))
        if response.status_code >= 400:
            return _remember_health(key, (False, f"HTTP {response.status_code}"))
        return _remember_health(key, (True, f"{self.model} · Google AI Studio"))

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
        content_chars = 0
        finish = ""
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", url, json=payload) as r:
                    if r.status_code >= 400:
                        body = (await r.aread()).decode()[:400]
                        if r.status_code == 429:
                            raise RateLimited.from_response(
                                "Gemini", body, r.headers, provider=self.name,
                                model=self.model, budget_key=self.budget_key)
                        raise ProviderError(f"HTTP {r.status_code}: {_tidy(body)}")
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        for cand in data.get("candidates", []):
                            finish = cand.get("finishReason") or finish
                            for part in cand.get("content", {}).get("parts", []):
                                # `thought` parts are the model's scratchpad,
                                # not its answer. Skipped, but they do not count
                                # as an answer having been given.
                                if part.get("text") and not part.get("thought"):
                                    content_chars += len(part["text"])
                                    yield part["text"]
        except httpx.HTTPError as e:
            raise ProviderError(f"Gemini: {type(e).__name__}") from e
        if content_chars == 0:
            reason = {
                "SAFETY": "Gemini's safety filter blocked the answer.",
                "RECITATION": "Gemini stopped: the answer looked like recitation.",
                "MAX_TOKENS": "Gemini hit its output limit before writing anything.",
            }.get(finish, f"{self.model} returned an empty response"
                          + (f" (finishReason: {finish})" if finish else "") + ".")
            raise EmptyAnswer(reason)


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

        # Deliberately NOT threaded, and the measurements are worth recording
        # because the obvious fix is the wrong one here.
        #
        # A synchronous forward pass holds the event loop, so three council
        # members took 3.24x one member -- exactly sequential, while the
        # interface promised parallel. The obvious remedy is to offload to a
        # worker thread. Measured, that made things worse:
        #
        #   per-token to_thread : single-member 0.87s -> 2.17s, ratio 2.35x
        #   whole-loop to_thread: single-member 0.87s -> 1.66s, ratio 2.34x
        #
        # The reason: this model is GIL-bound, not compute-bound. Running its
        # forward pass in three threads takes 6.41x as long as one -- worse
        # than sequential -- because at 800k parameters the time goes on Python
        # dispatch, which holds the GIL, rather than on kernels that release
        # it. Threads add contention and buy nothing.
        #
        # This does not affect any real backend. Ollama, Groq, Gemini and
        # Claude are all reached over HTTP through httpx, which releases the
        # loop for the whole network round trip, so the council is genuinely
        # concurrent there (verified separately against a local HTTP model
        # server). The scratch provider is a teaching artifact; it is the one
        # backend where "parallel" does not hold, and single-member latency
        # matters more for it than council throughput.
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
            await asyncio.sleep(0)   # let other coroutines advance between tokens


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
    """Build a provider from a spec string.

        groq:llama-3.3-70b-versatile
        ollama:qwen2.5:14b                    (model names may contain colons)
        openai:mistral-7b@http://localhost:1234/v1

    The council needs several differently-configured providers alive at once,
    which get_provider() cannot express because it reads one model per backend
    from the environment.

    The @base-url suffix exists because "several OpenAI-compatible endpoints"
    is a real configuration — LM Studio beside vLLM beside OpenRouter — and
    without it every such member would inherit the single OPENAI_BASE_URL and
    silently become the same backend three times over."""
    spec = spec.strip()
    if not spec:
        raise ProviderError("Empty provider spec")

    # Split the URL off first: it contains colons that are not separators.
    base_url = ""
    if "@" in spec:
        spec, _, base_url = spec.rpartition("@")
        base_url = base_url.strip()
        if not base_url.startswith(("http://", "https://")):
            raise ProviderError(
                f"'{base_url}' is not a base URL — expected http:// or https://")

    backend, _, model = spec.partition(":")
    provider = get_provider(backend.strip())

    if base_url:
        if not hasattr(provider, "base"):
            raise ProviderError(f"'{backend}' does not take a base URL")
        provider.base = base_url.rstrip("/")

    if model.strip():
        # Every backend keeps its model on `.model`; scratch and demo have none.
        if hasattr(provider, "model"):
            provider.model = model.strip()
        else:
            raise ProviderError(f"'{backend}' does not take a model name")
    return provider


def describe(provider: Provider) -> str:
    """A stable label for one configured provider, for display and dedupe.

    Includes the host when a custom base URL is set, because two members that
    differ only by endpoint must not collapse into one entry — the council
    dedupes on this string."""
    model = getattr(provider, "model", "")
    label = f"{provider.name}:{model}" if model else provider.name
    # Only the generic openai backend needs its host shown. A named backend
    # already implies its endpoint, so "groq:llama@api.groq.com" is noise.
    if provider.name == "openai":
        base = getattr(provider, "base", "")
        default = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if base and base.rstrip("/") != default.rstrip("/"):
            label = f"{label}@{base.split('//', 1)[-1].split('/')[0]}"
    return label


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
