"""
Which backend answers, and what happens when it cannot.

The failure this exists to end: Praxis picked one backend before the turn
started and then held it, whatever happened next. `resolve()` was a pre-flight
health probe — it asked "is this thing up?" once, and a backend that was up at
second zero and out of quota at second one got no fallback at all. The user saw
the assistant go quiet. Nothing in the code could rescue a turn that had begun.

So: a pool, not a provider.

Ordering is authored, not computed. The order you write in CAPACITY_POOL is the
quality order, because Praxis has no evidence about which model is better and
must not invent a score. Every entry should be one you would be happy to get
the answer from — a failover that quietly drops to a weaker model while
reporting success is precisely the failure this product refuses to make.

Two ways the pool is used, and the difference matters:

  sequential   one turn, one backend. If it runs dry mid-answer the next one
               picks the turn up. Quality order decides who goes first.
  concurrent   several people at once. The best backend is preferred, but a
               second simultaneous request goes to the *next* backend rather
               than queueing behind the first. Three backends answer three
               people at the same time instead of one after another.

What this cannot do, stated plainly because the alternative is a lie: free
tiers are finite. A pool of N makes running out roughly N times less likely and
means a single exhausted backend is invisible rather than fatal. It does not
make capacity infinite. When the pool really is dry, `exhaustion_report` says so
and says when each one comes back.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

from . import limits
from .providers import Provider, ProviderError, from_spec, get_provider

# A backend refused for a reason we could not read gets a bounded rest, never a
# day. Guessing "daily" on an unreadable 429 would bench a healthy backend for
# hours; guessing "per-minute" costs one wasted retry.
UNKNOWN_COOLDOWN_CAP = 900.0
MINUTE_COOLDOWN_CAP = 90.0
# A backend that is erroring (not rate limited) backs off, doubling each time.
ERROR_BACKOFF_BASE = 30.0
ERROR_BACKOFF_CAP = 900.0

# Env vars holding keys, by backend name. Comma-separated values become
# separate budgets — see `keys_for` for the caveat that matters.
# Backends that answer but are not models. Excluded from any pool Praxis
# derives for itself.
_NOT_A_MODEL = {"demo", "scratch"}

_KEY_VARS = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def key_id(key: str) -> str:
    """A stable, non-reversible label for a key. The key itself is never
    stored, logged or serialised anywhere."""
    if not key:
        return "none"
    return hashlib.sha256(key.encode()).hexdigest()[:6]


def keys_for(backend: str) -> list[str]:
    """Keys configured for a backend, in order.

    Comma-separated values are split so one backend can carry several accounts.
    The caveat, which belongs in the user's head and not just in .env.example:
    providers meter per *account*, so two keys from the same account share one
    bucket and buy nothing. This helps across genuinely separate accounts only.
    """
    raw = os.getenv(_KEY_VARS.get(backend, ""), "") or ""
    found, seen = [], set()
    for part in raw.split(","):
        part = part.strip()
        if len(part) >= 8 and part not in seen:
            seen.add(part)
            found.append(part)
    return found or [""]


@dataclass(frozen=True)
class Slot:
    """One (backend, model, key) triple the router can send a turn to."""

    spec: str
    key: str
    rank: int

    @property
    def budget_key(self) -> str:
        return f"{self.spec}|{key_id(self.key)}"

    @property
    def label(self) -> str:
        return self.spec


@dataclass
class Standing:
    """What the ledger believes about one slot right now."""

    budget_key: str
    cooldown_until: float = 0.0
    cooldown_reason: str = ""
    reset_estimated: bool = False
    day_remaining: int = -1
    day_reset_at: float = 0.0
    minute_remaining: int = -1
    inflight: int = 0

    def available(self, now: float) -> bool:
        if self.cooldown_until > now:
            return False
        if self.day_remaining == 0 and self.day_reset_at > now:
            return False
        return True


class Router:
    """Holds the pool, the ledger, and the decision about who answers next."""

    def __init__(self, settings, store) -> None:
        self.settings = settings
        self.store = store
        self._inflight: dict[str, int] = {}
        self._pending: dict[str, dict] = {}     # budget_key -> unflushed usage

    # -- the pool ----------------------------------------------------------

    def specs(self) -> list[str]:
        """Pool entries in the order the user wrote them.

        Falls back to PROVIDER + FALLBACK_CHAIN so an existing .env keeps
        working with no edit at all."""
        raw = (self.settings.capacity_pool or "").strip()
        if raw:
            return [s.strip() for s in raw.split(",") if s.strip()]
        chain = [self.settings.provider, *self.settings.fallback_chain]
        out, seen = [], set()
        for name in chain:
            # Never auto-derive a pool that can fail over to something which is
            # not a model. `demo` is a script and `scratch` is an untrained
            # network; either would answer, and the turn would report success
            # while the quality collapsed silently — the exact failure this
            # design refuses. A user who writes them into CAPACITY_POOL
            # explicitly gets them; inferring them is never right.
            if name in _NOT_A_MODEL or not name or name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    def slots(self) -> list[Slot]:
        out: list[Slot] = []
        rank = 0
        for spec in self.specs():
            backend = spec.split(":", 1)[0].split("@", 1)[0]
            for key in keys_for(backend):
                out.append(Slot(spec=spec, key=key, rank=rank))
                rank += 1
        return out

    def build(self, slot: Slot) -> Provider:
        """A live provider for a slot, wired to report what it learns."""
        provider = (from_spec(slot.spec) if (":" in slot.spec or "@" in slot.spec)
                    else get_provider(slot.spec))
        if slot.key and hasattr(provider, "api_key"):
            provider.api_key = slot.key
        provider.budget_key = slot.budget_key
        provider.meter = self._observe
        return provider

    # -- the ledger --------------------------------------------------------

    def standing(self, slot: Slot) -> Standing:
        row = self.store.get_budget(slot.budget_key) or {}
        return Standing(
            budget_key=slot.budget_key,
            cooldown_until=row.get("cooldown_until", 0.0) or 0.0,
            cooldown_reason=row.get("cooldown_reason", "") or "",
            reset_estimated=bool(row.get("reset_estimated", 0)),
            day_remaining=row.get("day_remaining", -1),
            day_reset_at=row.get("day_reset_at", 0.0) or 0.0,
            minute_remaining=row.get("minute_remaining", -1),
            inflight=self._inflight.get(slot.budget_key, 0),
        )

    def _observe(self, provider, facts, usage) -> None:
        """Called by a provider as a response reveals its standing."""
        budget = getattr(provider, "budget_key", "") or provider.name
        fields: dict = {"observed_at": time.time()}
        for fact in facts:
            if fact.window == limits.DAY or (
                    fact.scope == limits.REQUESTS and fact.window != limits.MINUTE):
                if fact.limit:
                    fields["day_limit"] = fact.limit
                if fact.remaining >= 0:
                    fields["day_remaining"] = fact.remaining
                if fact.reset_at:
                    fields["day_reset_at"] = fact.reset_at
            else:
                if fact.limit:
                    fields["minute_limit"] = fact.limit
                if fact.remaining >= 0:
                    fields["minute_remaining"] = fact.remaining
                if fact.reset_at:
                    fields["minute_reset_at"] = fact.reset_at
        if len(fields) > 1:
            self.store.save_budget(budget, provider.name,
                                   getattr(provider, "model", ""), **fields)
        if usage:
            self._pending[budget] = {
                "backend": provider.name,
                "model": getattr(provider, "model", ""),
                **usage,
            }

    def flush(self, conversation_id: str | None = None,
              fallback: dict | None = None) -> None:
        """Write the turn's cost once, after it finishes. Never per chunk —
        SQLite writes on the event loop are not free."""
        pending = self._pending
        self._pending = {}
        for budget, usage in pending.items():
            self.store.record_usage(
                budget, usage.get("backend", ""), usage.get("model", ""),
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                estimated=not usage.get("measured"), conversation_id=conversation_id)
        if not pending and fallback:
            self.store.record_usage(
                fallback.get("budget_key", ""), fallback.get("backend", ""),
                fallback.get("model", ""), fallback.get("prompt_tokens", 0),
                fallback.get("completion_tokens", 0), estimated=True,
                conversation_id=conversation_id)

    # -- choosing ----------------------------------------------------------

    def pick(self, exclude: set[str] | None = None) -> Slot | None:
        """The best slot that can serve right now.

        Quality order first — the pool's order is the user's ranking and is
        never reordered by headroom. Among slots that are *equally* usable, a
        slot already busy with another user's turn loses to an idle one, which
        is what lets several people be answered at once instead of queueing on
        the best backend."""
        now = time.time()
        exclude = exclude or set()
        usable = [(s, self.standing(s)) for s in self.slots()
                  if s.budget_key not in exclude]
        usable = [(s, st) for s, st in usable if st.available(now)]
        if not usable:
            return None
        # rank is the authored quality order; inflight only breaks ties between
        # backends that are all available, so concurrency spreads without ever
        # demoting a better model when it is free.
        idle = [(s, st) for s, st in usable if st.inflight == 0]
        pool = idle or usable
        return min(pool, key=lambda pair: (pair[1].inflight, pair[0].rank))[0]

    def slot_for(self, provider) -> Slot | None:
        """The pool slot matching a provider the caller already built.

        Returns None for anything outside the pool — a pinned `?provider=`, a
        test's scripted stub, the demo backend. Those turns still run; they
        simply are not routed, which is right: the user asked for that one."""
        model = getattr(provider, "model", "")
        key = getattr(provider, "api_key", "")
        for slot in self.slots():
            backend = slot.spec.split(":", 1)[0].split("@", 1)[0]
            if backend != provider.name:
                continue
            wanted = slot.spec.partition(":")[2].rpartition("@")[0] or \
                slot.spec.partition(":")[2]
            if wanted and model and wanted != model:
                continue
            if slot.key and key and slot.key != key:
                continue
            return slot
        return None

    def begin(self, slot: Slot) -> None:
        self._inflight[slot.budget_key] = self._inflight.get(slot.budget_key, 0) + 1

    def end(self, slot: Slot) -> None:
        left = self._inflight.get(slot.budget_key, 1) - 1
        if left > 0:
            self._inflight[slot.budget_key] = left
        else:
            self._inflight.pop(slot.budget_key, None)

    # -- recording outcomes ------------------------------------------------

    def note_rate_limit(self, slot: Slot, error) -> None:
        """Bench a slot for as long as its own refusal justifies — no longer."""
        now = time.time()
        reset = getattr(error, "reset_at", 0.0) or 0.0
        window = getattr(error, "window", "unknown")
        estimated = getattr(error, "source", "inferred") == "inferred"

        if window in ("day", "hour"):
            until = reset or (now + 3600.0)
            fields = {"cooldown_until": until, "cooldown_reason": f"{window} quota spent",
                      "day_remaining": 0, "day_reset_at": until,
                      "reset_estimated": 1 if not reset else 0}
        elif window == "minute":
            until = min(reset or (now + 20.0), now + MINUTE_COOLDOWN_CAP)
            fields = {"cooldown_until": until, "cooldown_reason": "per-minute limit",
                      "minute_remaining": 0, "reset_estimated": 0}
        else:
            wait = getattr(error, "retry_after", 0.0) or 600.0
            until = now + min(wait, UNKNOWN_COOLDOWN_CAP)
            fields = {"cooldown_until": until,
                      "cooldown_reason": "rate limited (window not stated)",
                      "reset_estimated": 1}
        self.store.save_budget(slot.budget_key, slot.spec.split(":")[0],
                               slot.spec, observed_at=now, **fields)

    def note_error(self, slot: Slot) -> None:
        """A backend that is broken rather than spent. Backs off, doubling,
        and is labelled differently so the interface can say which it is."""
        row = self.store.get_budget(slot.budget_key) or {}
        failures = int(row.get("failures", 0)) + 1
        wait = min(ERROR_BACKOFF_BASE * (2 ** (failures - 1)), ERROR_BACKOFF_CAP)
        self.store.save_budget(
            slot.budget_key, slot.spec.split(":")[0], slot.spec,
            failures=failures, cooldown_until=time.time() + wait,
            cooldown_reason="backend erroring", observed_at=time.time())

    def note_success(self, slot: Slot) -> None:
        row = self.store.get_budget(slot.budget_key) or {}
        if row.get("failures") or row.get("cooldown_until"):
            self.store.save_budget(slot.budget_key, slot.spec.split(":")[0],
                                   slot.spec, failures=0, cooldown_until=0.0,
                                   cooldown_reason="")

    # -- reporting ---------------------------------------------------------

    def status(self) -> list[dict]:
        now = time.time()
        out = []
        for slot in self.slots():
            st = self.standing(slot)
            out.append({
                "spec": slot.spec,
                "key": key_id(slot.key),
                "rank": slot.rank,
                "available": st.available(now),
                "reason": st.cooldown_reason,
                "resets_in": max(0.0, max(st.cooldown_until, st.day_reset_at) - now),
                "estimated": st.reset_estimated,
                "inflight": st.inflight,
                "day_remaining": st.day_remaining,
                "minute_remaining": st.minute_remaining,
            })
        return out

    def exhaustion_report(self, lead: str = "") -> str:
        """What the user sees when the whole pool is dry.

        Never invents a reset boundary. When a provider did not say, this says
        it did not say — a confident wrong time is worse than an honest
        unknown, and the free tiers do not agree on when midnight is."""
        now = time.time()
        lines = []
        if lead:
            # Lead with what actually happened. On a one-backend pool this is
            # the whole useful message — "add another backend" is not an answer
            # to "this single request is larger than your per-minute
            # allowance", which needs a smaller request, not a wider pool.
            lines += [lead.rstrip(), ""]
        lines += ["Out of capacity on every backend in the pool. "
                  "Your message is saved — nothing was lost.", ""]
        soonest = None
        for row in self.status():
            if row["available"]:
                # Available but the turn still failed: report it honestly
                # rather than pretending it was spent.
                lines.append(f"  {row['spec']}  —  available, but did not answer")
                continue
            when = limits.describe_reset(now + row["resets_in"], row["estimated"], now)
            reason = row["reason"] or "unavailable"
            lines.append(f"  {row['spec']}  —  {reason}, back {when}")
            if row["resets_in"] > 0:
                soonest = row["resets_in"] if soonest is None else min(
                    soonest, row["resets_in"])
        lines.append("")
        if soonest is not None:
            lines.append(f"Soonest is {limits.describe_reset(now + soonest, False, now)}. "
                         f"Ask again then and it will answer normally.")
        lines += [
            "",
            "To make this rarer:",
            "  - Add another free tier to CAPACITY_POOL in .env.",
            "  - Add a key from a *different* account: GROQ_API_KEY=key1,key2 "
            "(two keys on one account share one quota).",
            "  - Add `ollama:llama3.2` to the end of CAPACITY_POOL as a local "
            "backstop that never runs out.",
        ]
        return "\n".join(lines)
