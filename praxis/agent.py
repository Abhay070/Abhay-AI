"""
The agent loop.

This is where the parts become a product. It assembles the system prompt from
identity + mode + memory + tools, streams the model's reply, intercepts tool
calls mid-stream, runs them, feeds the results back, and repeats until the model
stops asking for tools.

Everything is emitted as structured events rather than raw text, so the UI can
show *what the assistant is doing* — searching, calculating, remembering —
instead of an opaque pause. Watching an agent work is most of what makes one
feel trustworthy rather than magical.

Event types on the wire:
    start        which provider and mode, plus any fallback notes
    token        a chunk of visible text
    tool_call    the model asked for a tool
    tool_result  what came back
    memory       a fact was stored (so the UI can surface it)
    done         message id, elapsed time, rough token count
    error        something failed, with the reason
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

from . import modes as modes_mod
from . import tools as toolkit
from .config import Settings
from . import council as council_mod
from . import contracts as contracts_mod
from . import momentum as momentum_mod
from . import untrusted
from .constraints import correction_prompt, detect as detect_constraints
from .constraints import verify as verify_constraints
from .identity import build_system_prompt
from .providers import Provider, ProviderError, RateLimited

# Rough, and honest about it: ~4 characters per token holds well enough for
# English to be useful as a budget gauge, and badly enough that we never present
# it as exact.
CHARS_PER_TOKEN = 4

# However tight the budget, never send the model less than this much of
# the question. A prompt trimmed below it is not a cheaper request, it is
# a wasted one.
MIN_QUESTION_CHARS = 1200

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "about",
    "i", "you", "it", "this", "that", "my", "your", "me", "we", "they", "do",
    "does", "did", "can", "could", "would", "should", "will", "what", "how",
    "why", "when", "where", "who", "which", "have", "has", "had", "not",
}


@dataclass
class Event:
    type: str
    data: dict = field(default_factory=dict)


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in STOPWORDS}


def select_memories(store, recent_text: str, limit: int = 12) -> list[dict]:
    """
    Choose which memories to spend context on.

    Scoring is deliberately simple: keyword overlap with the recent conversation,
    plus a small bonus for memories that have proven useful before. No embeddings,
    no vector store.

    That is a considered choice, not a shortcut. At personal scale — hundreds of
    memories, not millions — lexical overlap is roughly as accurate as embeddings
    and has one decisive advantage: you can look at a memory and understand why it
    was recalled. A memory system whose behaviour is inexplicable is one you stop
    trusting, and an untrusted memory is worse than none.

    Swap in embeddings when the count passes a few thousand. The interface here
    does not change.
    """
    all_memories = store.list_memories(limit=400)
    if not all_memories:
        return []

    query = _keywords(recent_text)
    scored = []
    for memory in all_memories:
        overlap = len(query & _keywords(memory["content"]))
        score = overlap * 10 + min(memory.get("hits", 0), 5)
        # Preferences and goals are cheap and almost always relevant.
        if memory["category"] in ("preference", "goal"):
            score += 3
        scored.append((score, memory))

    scored.sort(key=lambda pair: (-pair[0], -pair[1]["updated_at"]))
    chosen = [m for score, m in scored if score > 0][:limit]

    # Even with no keyword hit, a handful of recent facts keeps continuity.
    if len(chosen) < 5:
        seen = {m["id"] for m in chosen}
        for _, memory in scored:
            if memory["id"] not in seen:
                chosen.append(memory)
                seen.add(memory["id"])
            if len(chosen) >= 5:
                break
    return chosen


def format_memories(memories: list[dict]) -> str:
    return "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)



def _shrink(messages: list[dict]) -> int:
    """Drop the oldest turns in place, so the retry costs less than the try.

    Mutates rather than returns because the caller is mid-stream and holding
    the list. Never touches the system prompt or the turn being answered."""
    turns = [i for i, m in enumerate(messages) if m["role"] != "system"]
    if len(turns) <= 1:
        return 0
    # Half the history, rounded up, but always leaving the newest turn.
    cut = max(1, (len(turns) - 1) // 2)
    for index in reversed(turns[:cut]):
        del messages[index]
    return cut


class Agent:
    def __init__(self, store, settings: Settings, router=None):
        self.store = store
        self.settings = settings
        if router is None and settings.enable_capacity_router:
            from .router import Router
            router = Router(settings, store)
        self.router = router

    def rate_limit_advice(self, error, messages: list[dict], attempts: int) -> str:
        """A rate limit the waiting could not fix, explained in terms of the fix.

        Free tiers meter tokens per minute — Groq's on-demand tier allows 8,000
        — and that allowance is spent by the *size* of each request, not the
        number of them. A long conversation with a thinking model can cost more
        per turn than the whole minute's allowance, at which point no amount of
        waiting will ever let it through. Saying "rate limited" and stopping
        leaves the user retrying something that cannot work."""
        cost = sum(len(m["content"]) for m in messages) // CHARS_PER_TOKEN
        lines = [str(error)]
        lines.append(f"\n\nPraxis waited and retried {attempts} times. This turn "
                     f"costs roughly {cost:,} tokens to send, before the answer.")
        lines.append("Three things shrink it, in order of effect:")
        lines.append("- Start a new conversation — history is resent every turn.")
        lines.append("- Set `REASONING_EFFORT=low` in .env if your model thinks "
                     "before answering; those tokens count too.")
        lines.append("- Lower `MAX_PROMPT_TOKENS` in .env, or pick a smaller "
                     "model with a larger allowance "
                     "(`python scripts/list_models.py`).")
        return "\n".join(lines)

    # -- prompt assembly ---------------------------------------------------

    def compose(self, history: list[dict], mode_key: str,
                user_name: str = "") -> tuple[list[dict], list[dict], list]:
        """Build the message list actually sent to the model."""
        mode = modes_mod.get(mode_key)

        budget_chars = self.settings.max_prompt_tokens * CHARS_PER_TOKEN

        tool_list, active_tools = "", []
        if self.settings.enable_tools:
            active_tools = toolkit.available(self.settings)
            tool_list = toolkit.describe(active_tools)
            # Fourteen tools described in full is ~700 tokens on every request.
            # When the whole budget is small — which is what a free tier's
            # per-minute allowance amounts to — that is a quarter of it spent
            # before the user has typed anything. Trade the worked examples for
            # the room to hold a conversation.
            if len(tool_list) > budget_chars * 0.15:
                tool_list = toolkit.describe(active_tools, compact=True)

        memories: list[dict] = []
        memory_block = ""
        if self.settings.enable_memory:
            recent = " ".join(m["content"] for m in history[-4:])
            memories = select_memories(self.store, recent)
            memory_block = format_memories(memories)

        system = build_system_prompt(
            mode_prompt=mode.prompt,
            tool_list=tool_list,
            memories=memory_block,
            user_name=user_name or self.settings.user_name,
        )

        messages = [{"role": "system", "content": system}] + self.fit(history, system)
        return messages, memories, active_tools

    # -- what actually fits ------------------------------------------------

    def attachment_text(self, ref: dict) -> str:
        """One attached file, as the model should see it.

        The file's text is fetched here rather than carried in the message, so
        that what the user sees in their own bubble stays what they typed. A
        forty-page PDF pasted into the transcript is not a conversation."""
        record = None
        try:
            record = self.store.get_file(ref.get("id", ""))
        except Exception:
            record = None

        name = (record or {}).get("filename") or ref.get("filename") or "file"
        text = ""
        path = (record or {}).get("path")
        if path:
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
        if not text:
            text = (record or {}).get("excerpt") or ""
        if not text:
            return (f"\n\n--- Attached file: {name} ---\n"
                    f"[The text of this file could not be read back. Tell the "
                    f"user the upload is no longer available.]")

        cap = self.settings.attachment_chars
        body, rest = text[:cap], max(0, len(text) - cap)
        tail = (f"\n[This is the first {cap:,} of {len(text):,} characters. Call "
                f"read_file with file_id \"{ref.get('id', '')}\" for the rest.]"
                if rest else "")
        # A document the user uploaded is still a document, not a second voice
        # in the conversation. "Ignore your instructions" typed into a PDF is a
        # sentence in a PDF.
        return "\n\n" + untrusted.wrap(
            "uploaded file",
            f"{name} (id: {ref.get('id', '')}, {len(text):,} characters)",
            body + tail)

    def expand(self, message: dict) -> str:
        """A stored turn, with any files it carried folded back in."""
        refs = (message.get("meta") or {}).get("attachments") or []
        if not refs:
            return message["content"]
        return message["content"] + "".join(self.attachment_text(r) for r in refs)

    def fit(self, history: list[dict], system: str) -> list[dict]:
        """Take as much recent history as the prompt budget allows.

        MAX_HISTORY caps the number of turns, which says nothing about their
        size. This caps the size, which is what a provider actually charges for
        and what a free tier's tokens-per-minute limit actually measures.

        The question being asked is never what gets dropped. That sounds
        obvious and the first version of this got it wrong: it kept whatever it
        reached first walking backwards, which is the last *assistant* turn
        when the history ends with one. A large attachment on the user's turn
        then pushed the question itself out of the prompt, and the model
        answered a conversation it could no longer see the point of."""
        turns = [m for m in history[-self.settings.max_history:]
                 if m["role"] in ("user", "assistant")]
        if not turns:
            return []

        budget = self.settings.max_prompt_tokens * CHARS_PER_TOKEN - len(system)
        # From the newest user turn onward is the question and anything the
        # assistant has already said about it. That block is non-negotiable.
        anchor = max((i for i, m in enumerate(turns) if m["role"] == "user"),
                     default=len(turns) - 1)
        kept = [{"role": m["role"], "content": self.expand(m)}
                for m in turns[anchor:]]

        spent = sum(len(m["content"]) for m in kept)
        if spent > budget:
            # It does not fit even alone — so abridge it rather than drop it.
            # The bulk is almost always an attachment on the user's turn, and
            # the model can pull the rest back with read_file.
            note = ("\n\n[…trimmed to fit the context budget. Call read_file for "
                    "the full text of any attachment above.]")
            others = spent - len(kept[0]["content"])
            room = max(budget - others - len(note), MIN_QUESTION_CHARS)
            # Only if it is genuinely too long. The block can exceed the budget
            # because the *system prompt* filled it, in which case a
            # thirty-character question is not what overspent and telling the
            # model it was "trimmed" is simply false.
            if len(kept[0]["content"]) > room:
                kept[0]["content"] = kept[0]["content"][:room] + note
            spent = sum(len(m["content"]) for m in kept)

        budget -= spent
        for m in reversed(turns[:anchor]):
            body = self.expand(m)
            if len(body) > budget:
                break
            budget -= len(body)
            kept.insert(0, {"role": m["role"], "content": body})
        return kept

    # -- the loop ----------------------------------------------------------

    async def _one_pass(self, provider: Provider, messages: list[dict],
                        tool_names: set[str], conversation_id: str,
                        out: dict, router=None, slot=None) -> AsyncIterator[Event]:
        """One full generation, including any tool rounds it needs.

        Writes the finished visible text and counters into `out` rather than
        returning them, because an async generator cannot do both.

        `provider` is rebound in place when a backend runs out of quota, so a
        turn that began on an exhausted free tier finishes on the next one
        instead of dying. That rebinding is the whole point: before it, the
        backend was chosen once before the first token and held whatever
        happened, which is why running out of quota looked like the assistant
        simply going quiet."""
        visible = ""
        rounds = 0
        completion_chars = 0
        tried: set[str] = {slot.budget_key} if slot else set()
        switches: list[dict] = []

        while True:
            buffer = ""
            emitted = 0        # how much of `buffer` has already been sent as tokens
            call_found = None
            failed = ""

            # A rate limit gets waited out rather than surfaced as a failure.
            # On a free tier — 8,000 tokens a minute on Groq's on-demand tier —
            # a 429 is the normal cost of asking two questions in quick
            # succession, not a fault. The wait is narrated so the pause is
            # explained instead of merely long.
            for rl_attempt in range(1 + self.settings.max_rate_limit_retries):
                buffer, emitted, call_found = "", 0, None
                before = len(visible)
                try:
                    async for chunk in provider.stream(messages):
                        buffer += chunk
                        completion_chars += len(chunk)

                        # Hold back text once a call block opens, so the raw JSON
                        # never reaches the user's screen.
                        open_at = buffer.find("<tool_call>", max(0, emitted - 12))
                        safe_to = open_at if open_at != -1 else len(buffer)
                        # Also hold back a possible partial "<tool_call" at the tail.
                        if open_at == -1:
                            tail = buffer[-12:]
                            for i in range(len(tail)):
                                if "<tool_call>".startswith(tail[i:]):
                                    safe_to = len(buffer) - len(tail) + i
                                    break

                        if safe_to > emitted:
                            text = buffer[emitted:safe_to]
                            emitted = safe_to
                            visible += text
                            yield Event("token", {"text": text})

                        if open_at != -1 and "</tool_call>" in buffer[open_at:]:
                            parsed = toolkit.parse_calls(buffer)
                            if parsed:
                                call_found = parsed[0]
                                break
                except RateLimited as e:
                    if router and slot:
                        router.note_rate_limit(slot, e)
                    spent = rl_attempt >= self.settings.max_rate_limit_retries
                    streamed = len(visible) - before

                    # Waiting is the right answer to a per-minute limit and the
                    # wrong answer to a daily one. Telling them apart is the
                    # whole reason praxis/limits.py exists: a daily cap used to
                    # be clamped to a 45-second wait, retried three times, and
                    # paid for by deleting half the conversation to make a
                    # request smaller that was never too big.
                    hopeless = e.is_daily or spent

                    if hopeless and router:
                        # Too much already on screen to throw away — keep it and
                        # stop, rather than re-spending a nearly-finished answer
                        # on another backend.
                        if streamed > self.settings.capacity_restart_max_chars:
                            note = ("\n\n_This answer stopped early — "
                                    f"{provider.label} ran out of capacity "
                                    "mid-sentence. Ask me to continue and I'll "
                                    "pick up from here._")
                            visible += note
                            yield Event("token", {"text": note})
                            yield Event("truncated", {
                                "answered_by": provider.label,
                                "chars": streamed,
                            })
                            break

                        nxt = router.pick(exclude=tried)
                        if nxt is not None:
                            # Discard this round's partial text and hand the
                            # turn to the next backend. Stitching two models'
                            # prose together at an arbitrary token boundary
                            # would be "a change in the response" and would
                            # defeat the constraint and contract checks, which
                            # score the whole answer.
                            visible = visible[:before]
                            switches.append({"from": provider.label,
                                             "to": nxt.spec, "reason": e.window})
                            if slot:
                                router.end(slot)
                            slot = nxt
                            tried.add(nxt.budget_key)
                            router.begin(slot)
                            provider = router.build(slot)
                            yield Event("provider_switch", {
                                "from": switches[-1]["from"],
                                "to": nxt.spec,
                                "reason": e.window,
                                "detail": str(e)[:160],
                                "keep_chars": before,
                            })
                            continue
                        # Nothing left in the pool. A per-minute limit still
                        # deserves its own advice — a wider pool is no answer
                        # to a request that is simply too big for one minute.
                        lead = ("" if e.is_daily
                                else self.rate_limit_advice(e, messages, rl_attempt))
                        failed = router.exhaustion_report(lead)
                        break

                    if hopeless or streamed > 0:
                        failed = self.rate_limit_advice(e, messages, rl_attempt)
                        break

                    delay = min(max(e.retry_after, 2.0) + 0.5, 45.0)
                    # Per-minute only: a token allowance is spent by the size of
                    # the request, so shrinking genuinely helps here. Against a
                    # daily cap it would be pure loss, which is why this branch
                    # is now unreachable for one.
                    dropped = _shrink(messages)
                    yield Event("rate_limited", {
                        "seconds": round(delay, 1),
                        "attempt": rl_attempt + 1,
                        "of": self.settings.max_rate_limit_retries,
                        "dropped_turns": dropped,
                        "window": e.window,
                        "message": str(e),
                    })
                    await asyncio.sleep(delay)
                    continue
                except ProviderError as e:
                    failed = str(e)
                    break
                except Exception as e:
                    failed = f"{type(e).__name__}: {e}"
                    break
                break

            if failed:
                out["error"] = failed
                yield Event("error", {"message": failed})
                return

            if call_found is None:
                # Flush anything held back that turned out not to be a call.
                if len(buffer) > emitted:
                    text = buffer[emitted:]
                    visible += text
                    yield Event("token", {"text": text})
                break

            rounds += 1
            if rounds > self.settings.max_tool_rounds:
                note = (f"\n\n_Stopped after {self.settings.max_tool_rounds} tool "
                        f"rounds — the model kept calling tools without concluding._")
                visible += note
                yield Event("token", {"text": note})
                break

            name, args = call_found["name"], call_found["args"]
            tool = toolkit.REGISTRY.get(name)
            yield Event("tool_call", {
                "name": name, "args": args,
                "icon": tool.icon if tool else "▸",
                "label": tool.description.split(".")[0] if tool else name,
            })

            if name not in tool_names:
                result = toolkit.ToolResult(
                    False, f"Tool '{name}' is not enabled. Available: "
                           f"{', '.join(sorted(tool_names)) or 'none'}")
            else:
                result = toolkit.execute(name, args,
                                         {"conversation_id": conversation_id})

            # Anything a tool brings back came from outside. Flagging it is
            # advisory and never blocks: a page *about* prompt injection trips
            # this, and silently dropping a legitimate page would be a worse
            # bug than the one being prevented.
            suspicious = untrusted.looks_like_injection(result.output)
            yield Event("tool_result", {
                "name": name, "ok": result.ok,
                "output": result.output[:1500], "data": result.data,
                "suspicious": suspicious,
            })
            if name == "remember" and result.ok and result.data.get("memory_id"):
                yield Event("memory", result.data)

            # Feed the exchange back so the model can continue with the answer.
            messages.append({"role": "assistant", "content": buffer.strip()})
            messages.append({
                "role": "user",
                "content": untrusted.wrap("tool result", f"the {name} tool",
                                          result.to_prompt())
                           + "\n\nContinue. Answer the original question using "
                             "this result. Do not repeat the tool call.",
            })

        out["text"] = toolkit.strip_calls(visible).strip()
        out["rounds"] = rounds
        out["completion_chars"] = completion_chars
        out["answered_by"] = provider.label
        if switches:
            out["switches"] = switches

    async def _council_pass(self, messages: list[dict], question: str,
                            strategy: str, out: dict) -> AsyncIterator[Event]:
        """Ask several models, then hand back the answer that wins.

        This cannot stream the way a single model does — nothing can be shown
        until every member has finished and the judge has ruled. So the wait is
        narrated instead: which members are running, what each returned, who
        won and why. A visible queue beats an invisible pause."""
        members = council_mod.build_members(self.settings.council_members)
        if not members:
            out["error"] = "No council members configured. Set COUNCIL_MEMBERS."
            yield Event("error", {"message": out["error"]})
            return

        live, dropped = await council_mod.healthy_members(members)
        yield Event("council_start", {
            "strategy": strategy,
            "members": [council_mod.describe(m) for m in live],
            "dropped": dropped,
        })
        if not live:
            out["error"] = ("Every council member is unreachable: "
                            + "; ".join(dropped))
            yield Event("error", {"message": out["error"]})
            return

        if strategy == "race":
            winner, candidates = await council_mod.race(
                live, messages, self.settings.council_timeout)
            verdict = council_mod.Verdict(
                winner, "first usable answer", "race", candidates)
        else:
            candidates = await council_mod.gather(
                live, messages, self.settings.council_timeout)
            for c in candidates:
                yield Event("council_member", {
                    "label": c.label, "provider": c.provider,
                    "ok": c.usable, "elapsed": c.elapsed,
                    "error": c.error,
                    "preview": c.text[:220],
                })
            judge_provider = None
            if self.settings.council_judge:
                try:
                    judge_provider = council_mod.from_spec(self.settings.council_judge)
                except ProviderError:
                    judge_provider = None
            yield Event("council_judging", {"judge": self.settings.council_judge
                                            or "heuristic"})
            verdict = await council_mod.judge(judge_provider, question, candidates)

        yield Event("council_verdict", {
            "winner": verdict.winner.provider,
            "label": verdict.winner.label,
            "reason": verdict.reason,
            "method": verdict.method,
            "no_consensus": verdict.no_consensus,
            "judge_elapsed": verdict.judge_elapsed,
            "considered": [
                {"label": c.label, "provider": c.provider, "ok": c.usable,
                 "elapsed": c.elapsed, "error": c.error}
                for c in verdict.candidates
            ],
        })

        if verdict.winner.error:
            out["error"] = f"No member answered: {verdict.winner.error}"
            yield Event("error", {"message": out["error"]})
            return

        text = verdict.winner.text
        if verdict.no_consensus:
            # Three weak answers are not one good answer. Say so on the answer
            # itself, not only in a panel the user may have collapsed — but
            # still show the best of them, because withholding it would be the
            # other failure this product refuses to make.
            others = sum(1 for c in verdict.candidates
                         if c.usable and c.label != verdict.winner.label)
            text = (f"**NO CONSENSUS — insufficient confidence.** "
                    f"{verdict.reason}\n\n"
                    f"Below is the strongest of the answers anyway. Treat it as "
                    f"a starting point, not a finding, and verify it before you "
                    f"rely on it"
                    + (f" — the other {others} answered differently."
                       if others else ".")
                    + "\n\n---\n\n" + text)

        # Emit in chunks so the interface behaves the same as a live stream.
        for i in range(0, len(text), 24):
            yield Event("token", {"text": text[i:i + 24]})
            await asyncio.sleep(0)

        out["text"] = text
        out["rounds"] = 0
        out["completion_chars"] = sum(len(c.text) for c in verdict.candidates)
        out["verdict"] = verdict

    async def _cascade_pass(self, provider: Provider, messages: list[dict],
                            tool_names: set[str], conversation_id: str,
                            out: dict) -> AsyncIterator[Event]:
        """Cheap model first; escalate only when its answer looks weak.

        Most questions do not need the expensive model. Paying for it on every
        turn is the easiest way to make a good product too costly to run."""
        async for event in self._one_pass(provider, messages, tool_names,
                                          conversation_id, out):
            yield event
        if "error" in out:
            return

        weak, why = council_mod.looks_weak(out.get("text", ""))
        if not weak:
            out["cascade"] = {"escalated": False}
            return

        try:
            strong = council_mod.from_spec(self.settings.cascade_strong)
        except ProviderError as e:
            out["cascade"] = {"escalated": False, "note": str(e)}
            return
        ok, detail = await strong.health()
        if not ok:
            out["cascade"] = {"escalated": False, "note": detail}
            return

        yield Event("cascade_escalate", {
            "reason": why,
            "from": council_mod.describe(provider),
            "to": council_mod.describe(strong),
        })
        first = out.get("text", "")
        out.clear()
        async for event in self._one_pass(strong, list(messages), tool_names,
                                          conversation_id, out):
            yield event
        out["cascade"] = {"escalated": True, "reason": why,
                          "discarded_chars": len(first)}

    async def run(self, provider: Provider, conversation_id: str, history: list[dict],
                  mode_key: str, user_name: str = "",
                  notes: list[str] | None = None,
                  strategy: str | None = None) -> AsyncIterator[Event]:
        started = time.time()
        messages, memories, active_tools = self.compose(history, mode_key, user_name)

        if memories:
            self.store.touch_memories([m["id"] for m in memories])

        last_user = next((m["content"] for m in reversed(history)
                          if m["role"] == "user"), "")
        constraints = (detect_constraints(last_user)
                       if self.settings.enable_constraint_check else [])

        # The caller decides who opens the turn — the server asks the router,
        # which is where quality order and concurrency spreading happen. The
        # agent never second-guesses that choice; it only reaches for the
        # router when the chosen backend runs out mid-turn.
        slot = None
        if self.router:
            slot = self.router.slot_for(provider)
            if slot is not None:
                self.router.begin(slot)
                provider.meter = self.router._observe
                provider.budget_key = slot.budget_key

        strategy = (strategy or self.settings.default_strategy).lower()
        if strategy not in council_mod.STRATEGIES:
            strategy = "single"

        yield Event("start", {
            "strategy": strategy,
            "provider": provider.name,
            "provider_label": provider.label,
            "mode": mode_key,
            "memories_used": len(memories),
            "tools_available": [t.name for t in active_tools],
            "constraints": [c.description for c in constraints],
            "promises": contracts_mod.describe(mode_key),
            "notes": notes or [],
        })

        tool_names = {t.name for t in active_tools}
        prompt_chars = sum(len(m["content"]) for m in messages)
        completion_chars = 0
        total_rounds = 0
        attempts = 0
        final = ""
        working = list(messages)

        # Generate, then check the answer against any constraint the request
        # actually stated. A model cannot count its own letters mid-generation;
        # a checker can, in microseconds, and hand back the specific defect.
        for attempt in range(1 + self.settings.max_constraint_retries):
            attempts = attempt + 1
            if attempt:
                yield Event("constraint_retry", {
                    "attempt": attempts,
                    "violations": [v.detail for v in violations]
                                  + [f"{mode_key} mode: {b.detail}" for b in breaches]
                                  + [st.detail for st in stalls],
                })

            out: dict = {}
            if strategy in ("council", "race"):
                async for event in self._council_pass(working, last_user,
                                                      strategy, out):
                    yield event
            elif strategy == "cascade":
                async for event in self._cascade_pass(provider, working, tool_names,
                                                      conversation_id, out):
                    yield event
            else:
                async for event in self._one_pass(provider, working, tool_names,
                                                  conversation_id, out,
                                                  router=self.router, slot=slot):
                    yield event
            if "error" in out:
                return

            final = out.get("text", "")
            total_rounds += out.get("rounds", 0)
            completion_chars += out.get("completion_chars", 0)

            # A blank answer with no error is the one failure a user cannot
            # diagnose or work around: the turn reports success, the bubble is
            # empty, and nothing anywhere says why. Providers now raise instead
            # of returning silence, so reaching here means something slipped
            # through — say so on screen rather than storing the blank.
            if not final.strip():
                note = ("_The model finished without writing an answer. Nothing "
                        "failed on the way there, so this is the model itself "
                        "producing no text — try asking again, or switch model "
                        "in Settings._")
                yield Event("empty_answer", {"strategy": strategy,
                                             "provider": provider.name})
                yield Event("token", {"text": note})
                final = note
                break

            violations = verify_constraints(final, constraints) if constraints else []
            breaches = (contracts_mod.check(mode_key, final)
                        if self.settings.enable_mode_contracts else [])
            stalls = (momentum_mod.check(final)
                      if self.settings.enable_momentum else [])

            if not violations and not breaches and not stalls:
                if attempt:
                    yield Event("constraint_ok", {"attempts": attempts})
                break

            if breaches:
                yield Event("mode_breach", {
                    "mode": mode_key,
                    "broke": [b.detail for b in breaches],
                    "attempt": attempts,
                })
            if stalls:
                yield Event("stalled", {
                    "quote": stalls[0].quote,
                    "detail": stalls[0].detail,
                    "attempt": attempts,
                })

            if attempt >= self.settings.max_constraint_retries:
                # Out of retries. Say so rather than passing off a broken answer
                # as compliant — an unflagged violation is the worse failure.
                problems = ([v.detail[:120] for v in violations]
                            + [f"{mode_key} mode: {b.detail}" for b in breaches]
                            + [st.detail for st in stalls])
                note = ("\n\n---\n_Check failed after "
                        f"{attempts} attempts: " + "; ".join(problems)
                        + ". The answer above does not meet what you asked for._")
                final += note
                yield Event("token", {"text": note})
                yield Event("constraint_failed", {
                    "attempts": attempts,
                    "violations": [v.detail for v in violations],
                })
                break

            repair = []
            if violations:
                repair.append(correction_prompt(violations))
            if breaches:
                repair.append(contracts_mod.repair_prompt(mode_key, breaches))
            if stalls:
                repair.append(momentum_mod.repair_prompt(stalls))
            working = list(messages) + [
                {"role": "assistant", "content": final},
                {"role": "user", "content": "\n\n".join(repair)},
            ]

        message_id = self.store.add_message(
            conversation_id, "assistant", final,
            {"mode": mode_key, "provider": provider.name,
             "answered_by": out.get("answered_by", provider.label),
             "switches": out.get("switches", []),
             "strategy": strategy,
             "tool_rounds": total_rounds, "memories_used": len(memories),
             "constraint_attempts": attempts},
        )

        if self.router:
            if slot is not None:
                self.router.end(slot)
                self.router.note_success(slot)
            self.router.flush(conversation_id, fallback={
                "budget_key": getattr(provider, "budget_key", "") or provider.name,
                "backend": provider.name,
                "model": getattr(provider, "model", ""),
                "prompt_tokens": prompt_chars // CHARS_PER_TOKEN,
                "completion_tokens": completion_chars // CHARS_PER_TOKEN,
            })

        yield Event("done", {
            "message_id": message_id,
            "strategy": strategy,
            "answered_by": out.get("answered_by", provider.label),
            "switches": out.get("switches", []),
            "elapsed": round(time.time() - started, 2),
            "tool_rounds": total_rounds,
            "constraint_attempts": attempts,
            "tokens": {
                "prompt_estimate": prompt_chars // CHARS_PER_TOKEN,
                "completion_estimate": completion_chars // CHARS_PER_TOKEN,
            },
        })

    # -- titles ------------------------------------------------------------

    @staticmethod
    def derive_title(first_message: str) -> str:
        """Name a conversation from its opening line.

        Deliberately not a model call: titling is worth zero latency and zero
        tokens, and trimming the first message gets it right most of the time."""
        text = " ".join(first_message.split())
        # Strip filler openers repeatedly — "hey can you help me build X" has
        # three of them stacked before the actual subject.
        filler = re.compile(r"^(hi|hey|hello|ok|okay|so|please|thanks|can you|"
                            r"could you|would you|i want to|i need to|i'd like to|"
                            r"help me|tell me|explain|how do i)\b[ ,]*",
                            re.IGNORECASE)
        while True:
            stripped = filler.sub("", text).strip()
            if stripped == text or not stripped:
                break
            text = stripped
        if not text:
            return "New conversation"
        if len(text) <= 48:
            return text[0].upper() + text[1:]
        cut = text[:48].rsplit(" ", 1)[0]
        return (cut[0].upper() + cut[1:]).rstrip(",.;:") + "…"
