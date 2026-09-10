"""
The council: several models answer, the best answer wins.

The premise. A single model has a single set of blind spots. Ask three
different models the same hard question and they fail in *different* places —
one hallucinates a date, one misses the sarcasm, one nails it. That
disagreement is information, and a single-model product throws it away.

This module spends it instead. Four strategies, each a real trade:

    single    one model. Fastest, cheapest, the right default for chat.
    race      all members in parallel, first usable answer wins. Latency
              insurance — a rate-limited or sleeping backend stops mattering.
    council   all members in parallel, then a judge reads every answer and
              picks or synthesizes. Best quality, costs N+1 calls.
    cascade   a cheap model first; escalate to a strong one only when the
              cheap answer looks weak. Most of the quality, a fraction of
              the spend.

What makes this honest rather than theatre: the judge is shown the answers
blind — labelled A, B, C with no model names — so it cannot pick the answer
from the brand it recognizes. And when the judge itself fails, the fallback is
a stated heuristic, not a silent coin flip.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from .providers import Provider, ProviderError, describe, from_spec

# A judge that rambles costs latency for no benefit; it only needs a verdict.
JUDGE_MAX_CHARS = 1200

STRATEGIES = ("single", "race", "council", "cascade")


@dataclass
class Candidate:
    """One model's attempt at the question."""

    label: str                  # "A", "B", "C" — what the judge sees
    provider: str               # "groq:llama-3.3-70b" — what the user sees
    text: str = ""
    elapsed: float = 0.0
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and len(self.text.strip()) > 1


@dataclass
class Verdict:
    winner: Candidate
    reason: str
    method: str                 # "judge" | "heuristic" | "only-candidate"
    candidates: list[Candidate] = field(default_factory=list)
    judge_elapsed: float = 0.0


# --- gathering -------------------------------------------------------------

async def _collect(provider: Provider, messages: list[dict], label: str,
                   timeout: float) -> Candidate:
    """Run one member to completion. Never raises — a dead member is data."""
    started = time.time()
    cand = Candidate(label=label, provider=describe(provider))
    try:
        chunks: list[str] = []

        async def pump():
            async for piece in provider.stream(messages):
                chunks.append(piece)

        await asyncio.wait_for(pump(), timeout=timeout)
        cand.text = "".join(chunks).strip()
        if not cand.text:
            cand.error = "returned nothing"
    except asyncio.TimeoutError:
        cand.error = f"timed out after {timeout:.0f}s"
    except ProviderError as e:
        cand.error = str(e)
    except Exception as e:
        cand.error = f"{type(e).__name__}: {e}"
    cand.elapsed = round(time.time() - started, 2)
    return cand


async def gather(providers: list[Provider], messages: list[dict],
                 timeout: float = 90.0) -> list[Candidate]:
    """Every member, concurrently. Order of results matches order of members."""
    labels = [chr(ord("A") + i) for i in range(len(providers))]
    tasks = [_collect(p, list(messages), label, timeout)
             for p, label in zip(providers, labels)]
    return list(await asyncio.gather(*tasks))


async def race(providers: list[Provider], messages: list[dict],
               timeout: float = 90.0) -> tuple[Candidate, list[Candidate]]:
    """First usable answer wins; the rest are cancelled.

    This is latency insurance, not quality selection. It exists because a free
    tier that is rate-limiting today should not become your outage."""
    labels = [chr(ord("A") + i) for i in range(len(providers))]
    tasks = {asyncio.create_task(_collect(p, list(messages), lab, timeout)): lab
             for p, lab in zip(providers, labels)}

    done_cands: list[Candidate] = []
    pending = set(tasks)
    winner: Candidate | None = None
    while pending:
        finished, pending = await asyncio.wait(
            pending, return_when=asyncio.FIRST_COMPLETED)
        for task in finished:
            cand = task.result()
            done_cands.append(cand)
            if cand.usable and winner is None:
                winner = cand
        if winner is not None:
            break

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    if winner is None:
        winner = done_cands[0] if done_cands else Candidate("A", "none",
                                                            error="no members ran")
    return winner, done_cands


# --- judging ---------------------------------------------------------------

JUDGE_SYSTEM = """\
You are judging candidate answers to one question. You did not write any of
them and you have no stake in which wins.

Judge on, in order of weight:
1. Correctness. A confident wrong claim loses to an honest "I don't know".
   A fabricated date, name or citation disqualifies an answer outright.
2. Whether it actually answers what was asked, including any stated
   constraint (a word count, a format, a forbidden letter).
3. Directness. No filler, no restating the question, no empty praise.

Ignore length, ignore confidence of tone, and ignore writing polish except
where it affects clarity. The longest answer is not the best answer.

Reply in exactly this format and nothing else:

WINNER: <letter>
REASON: <one sentence, naming the specific thing that decided it>
"""

_WINNER_RE = re.compile(r"WINNER:\s*([A-Z])", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _ballot(question: str, candidates: list[Candidate]) -> str:
    """The judge sees letters, never model names — so it cannot vote on brand."""
    parts = [f"QUESTION:\n{question.strip()[:2000]}\n"]
    for c in candidates:
        parts.append(f"\n--- ANSWER {c.label} ---\n{c.text[:4000]}")
    parts.append(f"\n\nWhich answer is best? Choose from: "
                 f"{', '.join(c.label for c in candidates)}")
    return "\n".join(parts)


def _heuristic(candidates: list[Candidate]) -> tuple[Candidate, str]:
    """Fallback when no judge is available or the judge fails.

    Deliberately simple and stated out loud rather than pretending to be
    clever: prefer answers that hedge honestly over ones that don't, then
    prefer the faster model. Never silently prefer the longest — length is
    the single most misleading proxy for quality."""
    usable = [c for c in candidates if c.usable]
    if not usable:
        return candidates[0], "no member produced a usable answer"

    def score(c: Candidate) -> tuple[int, float]:
        text = c.text.lower()
        # An answer that admits a limit is more trustworthy than one that
        # cannot conceive of one.
        honest = any(p in text for p in (
            "i don't know", "i do not know", "not certain", "unverified",
            "could not verify", "no record of", "does not exist"))
        return (1 if honest else 0, -c.elapsed)

    best = max(usable, key=score)
    return best, f"picked by fallback heuristic ({len(usable)} usable answers)"


async def judge(judge_provider: Provider | None, question: str,
                candidates: list[Candidate]) -> Verdict:
    """Pick a winner. Falls back to a stated heuristic, never to a coin flip."""
    usable = [c for c in candidates if c.usable]

    if not usable:
        return Verdict(candidates[0], "every member failed", "heuristic", candidates)
    if len(usable) == 1:
        return Verdict(usable[0], "the only member that answered",
                       "only-candidate", candidates)
    if judge_provider is None:
        winner, reason = _heuristic(candidates)
        return Verdict(winner, reason, "heuristic", candidates)

    started = time.time()
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": _ballot(question, usable)},
    ]
    try:
        chunks: list[str] = []

        async def pump():
            async for piece in judge_provider.stream(messages):
                chunks.append(piece)
                if sum(len(c) for c in chunks) > JUDGE_MAX_CHARS:
                    break

        await asyncio.wait_for(pump(), timeout=60)
        raw = "".join(chunks)
    except Exception as e:
        winner, reason = _heuristic(candidates)
        return Verdict(winner, f"{reason} — judge failed ({type(e).__name__})",
                       "heuristic", candidates, round(time.time() - started, 2))

    elapsed = round(time.time() - started, 2)
    match = _WINNER_RE.search(raw)
    picked = None
    if match:
        letter = match.group(1).upper()
        picked = next((c for c in usable if c.label == letter), None)

    if picked is None:
        winner, reason = _heuristic(candidates)
        return Verdict(winner, f"{reason} — judge gave no clear verdict",
                       "heuristic", candidates, elapsed)

    reason_match = _REASON_RE.search(raw)
    reason = (reason_match.group(1).strip().split("\n")[0][:240]
              if reason_match else "selected by judge")
    return Verdict(picked, reason, "judge", candidates, elapsed)


# --- cascade ---------------------------------------------------------------

# Signals that a cheap model is out of its depth. Deliberately conservative:
# escalating when unnecessary just doubles the cost.
_WEAK_SIGNALS = (
    "i cannot", "i can't help", "as an ai language model",
    "i couldn't find any information", "i'm not sure", "i am not sure",
)


def looks_weak(text: str, min_chars: int = 40) -> tuple[bool, str]:
    """Should this answer be escalated to a stronger model?"""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return True, f"answer was only {len(stripped)} characters"
    lowered = stripped.lower()
    for signal in _WEAK_SIGNALS:
        if signal in lowered:
            return True, f'answer contained "{signal}"'
    # A model repeating itself has usually lost the thread.
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", stripped)
                 if len(s.strip()) > 15]
    if len(sentences) >= 3 and len(set(sentences)) < len(sentences) * 0.6:
        return True, "answer repeated itself"
    return False, ""


# --- configuration ---------------------------------------------------------

def build_members(specs: str) -> list[Provider]:
    """Parse "groq:llama-3.3-70b, gemini:gemini-2.0-flash" into providers.

    Members that cannot be constructed are skipped rather than fatal — a
    council with two live members still beats no answer at all."""
    members: list[Provider] = []
    seen: set[str] = set()
    for raw in specs.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            provider = from_spec(raw)
        except ProviderError:
            continue
        key = describe(provider)
        if key in seen:
            continue
        seen.add(key)
        members.append(provider)
    return members


async def healthy_members(members: list[Provider]) -> tuple[list[Provider], list[str]]:
    """Drop members that cannot serve, and say why each was dropped."""
    checks = await asyncio.gather(*(m.health() for m in members),
                                  return_exceptions=True)
    live, notes = [], []
    for member, result in zip(members, checks):
        if isinstance(result, BaseException):
            notes.append(f"{describe(member)}: {type(result).__name__}")
            continue
        ok, detail = result
        if ok:
            live.append(member)
        else:
            notes.append(f"{describe(member)}: {detail}")
    return live, notes
