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

import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from . import modes as modes_mod
from . import tools as toolkit
from .config import Settings
from .identity import build_system_prompt
from .providers import Provider, ProviderError

# Rough, and honest about it: ~4 characters per token holds well enough for
# English to be useful as a budget gauge, and badly enough that we never present
# it as exact.
CHARS_PER_TOKEN = 4

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


class Agent:
    def __init__(self, store, settings: Settings):
        self.store = store
        self.settings = settings

    # -- prompt assembly ---------------------------------------------------

    def compose(self, history: list[dict], mode_key: str,
                user_name: str = "") -> tuple[list[dict], list[dict], list]:
        """Build the message list actually sent to the model."""
        mode = modes_mod.get(mode_key)

        tool_list, active_tools = "", []
        if self.settings.enable_tools:
            active_tools = toolkit.available(self.settings)
            tool_list = toolkit.describe(active_tools)

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

        trimmed = history[-self.settings.max_history:]
        messages = [{"role": "system", "content": system}] + [
            {"role": m["role"], "content": m["content"]}
            for m in trimmed if m["role"] in ("user", "assistant")
        ]
        return messages, memories, active_tools

    # -- the loop ----------------------------------------------------------

    async def run(self, provider: Provider, conversation_id: str, history: list[dict],
                  mode_key: str, user_name: str = "",
                  notes: list[str] | None = None) -> AsyncIterator[Event]:
        started = time.time()
        messages, memories, active_tools = self.compose(history, mode_key, user_name)

        if memories:
            self.store.touch_memories([m["id"] for m in memories])

        yield Event("start", {
            "provider": provider.name,
            "provider_label": provider.label,
            "mode": mode_key,
            "memories_used": len(memories),
            "tools_available": [t.name for t in active_tools],
            "notes": notes or [],
        })

        tool_names = {t.name for t in active_tools}
        visible = ""       # what the user sees, tool syntax removed
        prompt_chars = sum(len(m["content"]) for m in messages)
        completion_chars = 0
        rounds = 0

        while True:
            buffer = ""
            emitted = 0        # how much of `buffer` has already been sent as tokens
            call_found = None

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
            except ProviderError as e:
                yield Event("error", {"message": str(e)})
                return
            except Exception as e:
                yield Event("error", {"message": f"{type(e).__name__}: {e}"})
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

            yield Event("tool_result", {
                "name": name, "ok": result.ok,
                "output": result.output[:1500], "data": result.data,
            })
            if name == "remember" and result.ok and result.data.get("memory_id"):
                yield Event("memory", result.data)

            # Feed the exchange back so the model can continue with the answer.
            messages.append({"role": "assistant", "content": buffer.strip()})
            messages.append({
                "role": "user",
                "content": f"<tool_result name=\"{name}\">\n{result.to_prompt()}\n"
                           f"</tool_result>\n\nContinue. Answer the original question "
                           f"using this result. Do not repeat the tool call.",
            })

        final = toolkit.strip_calls(visible).strip()
        message_id = self.store.add_message(
            conversation_id, "assistant", final,
            {"mode": mode_key, "provider": provider.name,
             "tool_rounds": rounds, "memories_used": len(memories)},
        )

        yield Event("done", {
            "message_id": message_id,
            "elapsed": round(time.time() - started, 2),
            "tool_rounds": rounds,
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
