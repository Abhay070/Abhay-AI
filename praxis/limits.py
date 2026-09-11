"""
Reading what a provider actually told you about its limits.

Every hosted backend meters two different things and refuses you for two very
different reasons, and until now Praxis could not tell them apart:

    per-minute   "you are going too fast"     — wait eight seconds, continue
    per-day      "you are done until tomorrow" — waiting cannot help at all

Both arrive as HTTP 429 with a number in them, so the old code treated them
identically: it clamped the wait to 45 seconds, retried three times, and threw
away half the conversation on each attempt to make the request smaller. Against
a daily cap that is three wasted requests and a destroyed history in exchange
for nothing. The user watches their assistant go quiet and has no idea why.

So this module reads the sentence, not just the number.

It also reads the headers, which is the part that matters most. Groq and every
OpenAI-compatible endpoint report remaining quota on *every* response — success
included — and Praxis was discarding all of it and guessing at four characters
per token instead. Reading them means capacity is known in advance rather than
discovered by being refused.

Nothing here imports from providers.py, so there is no cycle: providers reports
facts, the ledger stores them, the router acts on them.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

# Windows, shortest first. The order matters — `_WINDOW_FROM_RESET` walks it.
MINUTE = "minute"
HOUR = "hour"
DAY = "day"
UNKNOWN = "unknown"

TOKENS = "tokens"
REQUESTS = "requests"


@dataclass
class LimitFact:
    """What one response told us about the account's standing.

    Every field is optional because providers disagree about what to send.
    `source` records how we know, so the interface can say "Groq's own reset
    header said so" rather than presenting a guess with the same confidence as
    a measurement."""

    window: str = UNKNOWN           # minute | hour | day | unknown
    scope: str = UNKNOWN            # tokens | requests | unknown
    limit: int = 0
    used: int = 0
    remaining: int = -1             # -1 means never observed, 0 means empty
    retry_after: float = 0.0        # seconds, relative
    reset_at: float = 0.0           # absolute epoch — survives a restart
    source: str = "inferred"        # body | header | inferred
    detail: str = ""

    @property
    def is_daily(self) -> bool:
        return self.window in (DAY, HOUR)


# --- durations -------------------------------------------------------------
# Providers write waits as "7.66s", "2m59.56s", "1h30m", "6h12m30s", "550ms".
# Anything that fails to parse returns 0.0 rather than raising — a missing wait
# must never be the reason a turn dies.

_DURATION_PART = re.compile(r"(?P<value>[\d.]+)\s*(?P<unit>ms|s|m|h|d)", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(text: str) -> float:
    """Seconds from a provider's duration string. 0.0 when unreadable.

    Handles the compound forms ("2m59.56s") that a naive float() misses and
    that a naive "first number" regex gets catastrophically wrong — reading
    "6h12m" as six seconds is how a daily cap became a 45-second wait."""
    if not text:
        return 0.0
    raw = str(text).strip()
    # A bare number is seconds, which is what Retry-After sends.
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    total = 0.0
    for part in _DURATION_PART.finditer(raw):
        try:
            total += float(part.group("value")) * _UNIT_SECONDS[part.group("unit").lower()]
        except (ValueError, KeyError):
            continue
    return total


def _window_from_seconds(seconds: float) -> str:
    """Infer which bucket a reset belongs to from how far away it is.

    Necessary because header *names* do not say. Groq's
    `x-ratelimit-limit-requests` is a daily budget while
    `x-ratelimit-limit-tokens` is a per-minute one — same prefix, different
    window — so the only honest signal is the magnitude of the reset."""
    if seconds <= 0:
        return UNKNOWN
    if seconds > 3600:
        return DAY
    if seconds > 90:
        return HOUR
    return MINUTE


# --- the 429 body ----------------------------------------------------------

_PER_WINDOW = re.compile(
    r"\b(?P<scope>tokens?|requests?)\s+per\s+(?P<window>second|minute|hour|day)\b",
    re.IGNORECASE)
_ACRONYM = re.compile(r"\b(?P<code>TPM|RPM|TPH|RPH|TPD|RPD|TPS|RPS)\b")
_LIMIT_USED = re.compile(
    r"Limit\s+(?P<limit>[\d,]+).{0,40}?Used\s+(?P<used>[\d,]+)",
    re.IGNORECASE | re.DOTALL)
_TRY_AGAIN = re.compile(r"try again in\s+(?P<wait>[\dhms.\s]+?)(?:[.,]|\s|$)",
                        re.IGNORECASE)

_ACRONYM_WINDOW = {"S": MINUTE, "M": MINUTE, "H": HOUR, "D": DAY}


def _int(raw: str) -> int:
    try:
        return int(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0


def classify(body: str, headers=None, status: int = 429) -> LimitFact:
    """Read a refusal and say which limit was hit and when it lifts.

    Deliberately conservative: anything it cannot read stays `unknown`, and the
    caller treats unknown as per-minute. Retrying a per-minute limit costs a few
    seconds; benching a healthy backend for a day on a bad guess costs the user
    their assistant."""
    body = body or ""
    headers = headers or {}
    fact = LimitFact(source="inferred")

    match = _PER_WINDOW.search(body)
    if match:
        fact.scope = (TOKENS if match.group("scope").lower().startswith("token")
                      else REQUESTS)
        window = match.group("window").lower()
        fact.window = MINUTE if window == "second" else window
        fact.source = "body"
    else:
        acronym = _ACRONYM.search(body)
        if acronym:
            code = acronym.group("code").upper()
            fact.scope = TOKENS if code[0] == "T" else REQUESTS
            fact.window = _ACRONYM_WINDOW.get(code[-1], UNKNOWN)
            fact.source = "body"

    numbers = _LIMIT_USED.search(body)
    if numbers:
        fact.limit = _int(numbers.group("limit"))
        fact.used = _int(numbers.group("used"))
        fact.remaining = max(0, fact.limit - fact.used)

    # Gemini refuses with structured JSON rather than a sentence.
    if fact.window == UNKNOWN and ("QuotaFailure" in body or "RESOURCE_EXHAUSTED" in body):
        fact = _gemini(body, fact)

    wait = 0.0
    retry_header = headers.get("retry-after") or headers.get("Retry-After") or ""
    if retry_header:
        wait = parse_duration(retry_header)
        if wait:
            fact.source = "header" if fact.source == "inferred" else fact.source
    if not wait:
        hint = _TRY_AGAIN.search(body)
        if hint:
            wait = parse_duration(hint.group("wait"))
    fact.retry_after = wait
    if wait:
        fact.reset_at = time.time() + wait
        if fact.window == UNKNOWN:
            fact.window = _window_from_seconds(wait)

    fact.detail = " ".join(body.split())[:200]
    return fact


def _gemini(body: str, fact: LimitFact) -> LimitFact:
    """Gemini puts the quota in `error.details[]` rather than in prose."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return fact
    error = data.get("error", data) if isinstance(data, dict) else {}
    for entry in (error.get("details") or []):
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("@type", ""))
        if "QuotaFailure" in kind:
            for violation in (entry.get("violations") or []):
                quota_id = str(violation.get("quotaId", ""))
                low = quota_id.lower()
                if "perday" in low:
                    fact.window = DAY
                elif "perminute" in low:
                    fact.window = MINUTE
                if "request" in low:
                    fact.scope = REQUESTS
                elif "token" in low:
                    fact.scope = TOKENS
                fact.limit = _int(violation.get("quotaValue", 0)) or fact.limit
                fact.source = "body"
        elif "RetryInfo" in kind:
            delay = entry.get("retryDelay")
            seconds = parse_duration(delay if isinstance(delay, str) else "")
            if seconds:
                fact.retry_after = seconds
    return fact


# --- headers, on every response --------------------------------------------

_HEADER_KEYS = (
    ("x-ratelimit-limit-tokens", "limit", TOKENS),
    ("x-ratelimit-remaining-tokens", "remaining", TOKENS),
    ("x-ratelimit-reset-tokens", "reset", TOKENS),
    ("x-ratelimit-limit-requests", "limit", REQUESTS),
    ("x-ratelimit-remaining-requests", "remaining", REQUESTS),
    ("x-ratelimit-reset-requests", "reset", REQUESTS),
)


def read_headers(headers) -> list[LimitFact]:
    """Everything the response said about standing, success or failure.

    This is the self-correcting part of the whole design: real numbers arrive
    on the happy path, so the router can see a quota running low before it is
    refused, and no limit is ever hardcoded and left to go stale."""
    if not headers:
        return []
    out: dict[str, LimitFact] = {}
    for name, kind, scope in _HEADER_KEYS:
        raw = headers.get(name)
        if raw is None:
            continue
        fact = out.setdefault(scope, LimitFact(scope=scope, source="header"))
        if kind == "reset":
            seconds = parse_duration(raw)
            if seconds:
                fact.reset_at = time.time() + seconds
                fact.retry_after = seconds
                fact.window = _window_from_seconds(seconds)
        elif kind == "limit":
            fact.limit = _int(raw)
        else:
            fact.remaining = _int(raw)
    # A limit with no reset still tells us the shape of the budget.
    for fact in out.values():
        if fact.window == UNKNOWN and fact.limit:
            fact.window = MINUTE if fact.scope == TOKENS else DAY
    return list(out.values())


def read_usage(payload) -> dict:
    """Real token counts from a final stream chunk, if the provider sent any.

    Groq nests them under `x_groq`, OpenAI-compatible endpoints put `usage` at
    the top level when asked, Gemini calls it `usageMetadata`, Ollama uses
    `prompt_eval_count`/`eval_count`. All four are read here so the ledger can
    record what was actually spent rather than a four-characters-per-token
    guess."""
    if not isinstance(payload, dict):
        return {}
    usage = (payload.get("usage")
             or (payload.get("x_groq") or {}).get("usage")
             or payload.get("usageMetadata"))
    if isinstance(usage, dict):
        prompt = (usage.get("prompt_tokens") or usage.get("input_tokens")
                  or usage.get("promptTokenCount") or 0)
        completion = (usage.get("completion_tokens") or usage.get("output_tokens")
                      or usage.get("candidatesTokenCount") or 0)
        if prompt or completion:
            return {"prompt_tokens": int(prompt),
                    "completion_tokens": int(completion), "measured": True}
    if "prompt_eval_count" in payload or "eval_count" in payload:
        return {"prompt_tokens": int(payload.get("prompt_eval_count") or 0),
                "completion_tokens": int(payload.get("eval_count") or 0),
                "measured": True}
    return {}


def describe_reset(reset_at: float, estimated: bool = False, now: float = 0.0) -> str:
    """'in 6h 12m' — and says when that is a guess rather than a fact."""
    now = now or time.time()
    seconds = max(0.0, reset_at - now)
    if not reset_at:
        return "unknown — the provider did not say when"
    if seconds < 60:
        when = f"in {int(seconds)}s"
    elif seconds < 3600:
        when = f"in {int(seconds // 60)}m"
    else:
        hours, minutes = divmod(int(seconds // 60), 60)
        when = f"in {hours}h {minutes:02d}m"
    return when + (" (estimated)" if estimated else "")
