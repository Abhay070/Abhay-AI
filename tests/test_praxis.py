"""
Praxis test suite.

Run:  python tests/test_praxis.py

No pytest dependency — a personal project should not need a test framework
installed before you can check it still works. Every test is a function whose
name starts with `test_`; failures print the assertion and the suite continues,
so one break does not hide the other twelve.

What is covered: the parts where a silent regression would be expensive —
constraint detection and verification, the council's selection and its
fallbacks, memory dedupe and recall scoring, the tool registry and its safety
boundaries, provider spec parsing, and the agent loop end to end against
scripted models.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from praxis import constraints, council, modes                      # noqa: E402
from praxis.agent import Agent                                      # noqa: E402
from praxis.config import Settings                                  # noqa: E402
from praxis.identity import build_system_prompt                     # noqa: E402
from praxis.providers import Provider, from_spec, describe          # noqa: E402
from praxis.store import Store                                      # noqa: E402
from praxis.tools import REGISTRY, execute, parse_calls, strip_calls  # noqa: E402
from praxis.tools.calculate import calculate                        # noqa: E402


# --- helpers ---------------------------------------------------------------

class Scripted(Provider):
    """A model whose output you choose, so tests are deterministic."""

    def __init__(self, name: str, *replies: str, fail: bool = False,
                 delay: float = 0.0):
        self.name = name
        self.label = name
        self.model = name
        self.replies = list(replies) or [""]
        self.calls = 0
        self._fail = fail
        self._delay = delay

    async def health(self):
        return (not self._fail), ("ok" if not self._fail else "unreachable")

    async def stream(self, messages):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError("provider is down")
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        for word in reply.split(" "):
            yield word + " "


def fresh_store() -> tuple[Store, str]:
    path = tempfile.mktemp(suffix=".db")
    return Store(path), path


# --- constraints -----------------------------------------------------------

def test_letter_exclusion_detected_and_verified():
    cs = constraints.detect("Write a line. Do NOT use the letters 'e' or 's'.")
    kinds = {c.kind for c in cs}
    assert "forbidden_letters" in kinds, kinds
    letters = next(c for c in cs if c.kind == "forbidden_letters").spec["letters"]
    assert letters == ["e", "s"], letters

    bad = constraints.verify("These sentences are everywhere.", cs)
    assert bad, "violating text passed"
    good = constraints.verify("A bright dog ran.", cs)
    assert not good, f"clean text flagged: {[v.detail for v in good]}"


def test_word_count_scoped_to_named_field():
    cs = constraints.detect(
        "Return JSON with keys id and description. "
        "Limit every description to exactly 8 words")
    wc = next(c for c in cs if c.kind == "word_count")
    assert wc.spec["field"] == "description", wc.spec
    payload = json.dumps({"items": [
        {"id": 1, "description": "one two three four five six seven eight",
         "other": "short"},
    ]})
    # `other` is 1 word but must not be checked — only `description` was named.
    assert not constraints.verify(payload, cs), "wrong field was checked"


def test_word_count_catches_the_real_failure():
    cs = constraints.detect("Limit every description to exactly 8 words")
    payload = json.dumps([{"description": "Boosts energy levels and mental clarity always"}])
    bad = constraints.verify(payload, cs)
    assert bad and "7" in bad[0].detail, bad and bad[0].detail


def test_json_shape_and_no_false_positive():
    cs = constraints.detect("Format the output as a valid JSON object.")
    assert any(c.kind == "json" for c in cs)
    assert constraints.verify("[1,2,3]", cs), "array accepted where object asked"
    assert not constraints.verify('{"a":1}', cs)
    # A question *about* JSON is not a formatting instruction.
    assert not constraints.detect("Tell me about JSON parsing in Python.")


def test_sentence_count():
    cs = constraints.detect("Write a 4-sentence summary.")
    assert constraints.verify("One. Two.", cs), "short text not flagged"
    assert not constraints.verify("One. Two. Three. Four.", cs)


def test_correction_prompt_names_the_defect():
    cs = constraints.detect("Do not use the letter 'e'.")
    vs = constraints.verify("These letters everywhere.", cs)
    prompt = constraints.correction_prompt(vs)
    assert "'e' appears" in prompt, prompt[:200]
    assert "not a word" in prompt, "gibberish rule missing"


# --- council ---------------------------------------------------------------

def test_council_gathers_and_survives_a_dead_member():
    async def go():
        return await council.gather(
            [Scripted("a", "Answer from A"), Scripted("b", "", fail=True),
             Scripted("c", "Answer from C")],
            [{"role": "user", "content": "q"}])
    cands = asyncio.run(go())
    assert len(cands) == 3
    assert sum(c.usable for c in cands) == 2, [c.error for c in cands]


def test_council_ballot_is_blind():
    """The judge must never see which model wrote which answer."""
    seen = {}

    class Spy(Scripted):
        async def stream(self, messages):
            seen["ballot"] = messages[1]["content"]
            yield "WINNER: A\nREASON: because."

    cands = [council.Candidate("A", "groq:llama-3.3-70b", "first"),
             council.Candidate("B", "gemini:flash", "second")]
    asyncio.run(council.judge(Spy("j"), "q", cands))
    ballot = seen["ballot"]
    assert "groq" not in ballot and "gemini" not in ballot, "model names leaked"
    assert "ANSWER A" in ballot and "ANSWER B" in ballot


def test_council_falls_back_when_judge_fails():
    cands = [council.Candidate("A", "a", "first answer here"),
             council.Candidate("B", "b", "second answer here")]
    v = asyncio.run(council.judge(Scripted("j", fail=True), "q", cands))
    assert v.method == "heuristic", v.method
    assert "judge failed" in v.reason, v.reason


def test_council_falls_back_on_unparseable_verdict():
    cands = [council.Candidate("A", "a", "one"), council.Candidate("B", "b", "two")]
    v = asyncio.run(council.judge(Scripted("j", "I liked them all"), "q", cands))
    assert v.method == "heuristic", v.method


def test_council_single_usable_candidate_short_circuits():
    cands = [council.Candidate("A", "a", "only answer"),
             council.Candidate("B", "b", "", error="down")]
    v = asyncio.run(council.judge(None, "q", cands))
    assert v.method == "only-candidate" and v.winner.label == "A"


def test_race_returns_the_fastest():
    async def go():
        return await council.race(
            [Scripted("slow", "slow", delay=0.4), Scripted("quick", "quick", delay=0.01)],
            [{"role": "user", "content": "q"}])
    winner, _ = asyncio.run(go())
    assert winner.provider.startswith("quick"), winner.provider


def test_cascade_weakness_signals():
    assert council.looks_weak("ok")[0]
    assert council.looks_weak("I couldn't find any information about that.")[0]
    assert not council.looks_weak(
        "George M. Dallas broke the Senate tie on the Walker Tariff of 1846.")[0]


def test_members_dedupe_and_skip_unknown():
    members = council.build_members("demo, demo, nonsense-backend, demo")
    assert [describe(m) for m in members] == ["demo"], [describe(m) for m in members]


# --- providers -------------------------------------------------------------

def test_provider_spec_handles_colons_in_model_names():
    p = from_spec("ollama:qwen2.5:14b")
    assert p.name == "ollama" and p.model == "qwen2.5:14b", p.model


# --- tools -----------------------------------------------------------------

def test_calculator_is_correct_and_safe():
    assert "8294400" in calculate("3840*2160").output
    assert not calculate('__import__("os").system("ls")').ok, "code execution allowed!"
    assert not calculate("2**10000000").ok, "DoS not blocked"
    assert not calculate("1/0").ok


def test_tool_call_parsing_and_stripping():
    raw = 'Sure.\n<tool_call>\n{"name": "calculate", "args": {"expression": "2+2"}}\n</tool_call>'
    calls = parse_calls(raw)
    assert calls and calls[0]["name"] == "calculate", calls
    assert "<tool_call>" not in strip_calls(raw)
    assert parse_calls("<tool_call>{bad json}</tool_call>") == []


def test_unknown_tool_is_reported_not_raised():
    r = execute("does_not_exist", {})
    assert not r.ok and "No tool named" in r.output


def test_every_registered_tool_has_a_usable_signature():
    for name, tool in REGISTRY.items():
        assert tool.description, f"{name} has no description"
        assert tool.signature().startswith(f"- {name}:"), name


# --- memory ----------------------------------------------------------------

def test_memory_dedupes_case_and_punctuation():
    store, path = fresh_store()
    try:
        assert store.add_memory("Prefers Rust for systems work", "preference")
        assert store.add_memory("prefers rust for systems work.", "preference") is None
        assert len(store.list_memories()) == 1
    finally:
        os.remove(path)


def test_memory_recall_prefers_relevant():
    from praxis.agent import select_memories
    store, path = fresh_store()
    try:
        store.add_memory("Building Praxis, a personal AI", "project")
        store.add_memory("Allergic to shellfish", "fact")
        store.add_memory("Prefers Rust for systems programming", "preference")
        picked = select_memories(store, "what should I write my Rust parser in?")
        assert picked, "nothing recalled"
        assert "Rust" in picked[0]["content"], [m["content"] for m in picked]
    finally:
        os.remove(path)


def test_conversation_search_and_cascade_delete():
    store, path = fresh_store()
    try:
        cid = store.create_conversation("Rust parser design")
        store.add_message(cid, "user", "how do I write a tokenizer")
        assert store.search_conversations("tokenizer"), "body search failed"
        assert store.search_conversations("Rust parser"), "title search failed"
        store.delete_conversation(cid)
        assert store.get_messages(cid) == [], "messages outlived the conversation"
    finally:
        os.remove(path)


# --- identity & modes ------------------------------------------------------

def test_every_mode_is_distinct_and_reaches_the_prompt():
    prompts = {}
    for m in modes.catalogue():
        text = build_system_prompt(mode_prompt=modes.get(m["key"]).prompt)
        assert text not in prompts.values() or m["key"] == "standard", \
            f"{m['key']} duplicates {[k for k, v in prompts.items() if v == text]}"
        prompts[m["key"]] = text
    assert len(prompts) == 10, len(prompts)


def test_core_identity_carries_the_hard_rules():
    text = build_system_prompt()
    for needle in ("quietly validates a false claim",   # premise checking
                   "tokens, not characters",             # constraint honesty
                   "buffer overflow"):                   # no over-refusal
        assert needle in text, f"missing: {needle}"


# --- agent end to end ------------------------------------------------------

def test_agent_runs_a_tool_and_folds_in_the_result():
    store, path = fresh_store()
    try:
        s = Settings(enable_constraint_check=False)
        agent = Agent(store, s)
        cid = store.create_conversation("t")
        store.add_message(cid, "user", "what is 3840*2160")
        provider = Scripted(
            "m",
            '<tool_call>\n{"name":"calculate","args":{"expression":"3840*2160"}}\n</tool_call>',
            "That gives 8294400 pixels.")

        async def go():
            return [e async for e in agent.run(provider, cid,
                                               store.get_messages(cid), "standard")]
        events = asyncio.run(go())
        kinds = [e.type for e in events]
        assert "tool_call" in kinds and "tool_result" in kinds, kinds
        result = next(e for e in events if e.type == "tool_result")
        assert "8294400" in result.data["output"], result.data["output"]
        saved = store.get_messages(cid)[-1]["content"]
        assert "<tool_call>" not in saved, "raw tool syntax reached the user"
    finally:
        os.remove(path)


def test_agent_retries_a_constraint_violation_then_succeeds():
    store, path = fresh_store()
    try:
        agent = Agent(store, Settings(max_constraint_retries=2))
        cid = store.create_conversation("t")
        store.add_message(cid, "user", "One line, no letter 'e' or 's'.")
        provider = Scripted("m", "These sentences fail.", "A bright noon lit my dog.")

        async def go():
            return [e async for e in agent.run(provider, cid,
                                               store.get_messages(cid), "standard")]
        events = asyncio.run(go())
        kinds = [e.type for e in events]
        assert "constraint_retry" in kinds, kinds
        assert "constraint_ok" in kinds, kinds
        saved = store.get_messages(cid)[-1]["content"].lower()
        assert "e" not in saved and "s" not in saved, repr(saved)
    finally:
        os.remove(path)


def test_agent_discloses_when_it_cannot_comply():
    store, path = fresh_store()
    try:
        agent = Agent(store, Settings(max_constraint_retries=1))
        cid = store.create_conversation("t")
        store.add_message(cid, "user", "One line, no letter 'e'.")
        provider = Scripted("m", "These always fail everywhere.")

        async def go():
            return [e async for e in agent.run(provider, cid,
                                               store.get_messages(cid), "standard")]
        events = asyncio.run(go())
        assert any(e.type == "constraint_failed" for e in events)
        saved = store.get_messages(cid)[-1]["content"]
        assert "does not meet what you asked for" in saved, saved[-120:]
    finally:
        os.remove(path)


def test_agent_stops_a_runaway_tool_loop():
    store, path = fresh_store()
    try:
        agent = Agent(store, Settings(max_tool_rounds=3, enable_constraint_check=False))
        cid = store.create_conversation("t")
        store.add_message(cid, "user", "loop forever")
        # A model that only ever emits tool calls and never concludes.
        provider = Scripted(
            "m", '<tool_call>\n{"name":"calculate","args":{"expression":"1+1"}}\n</tool_call>')

        async def go():
            return [e async for e in agent.run(provider, cid,
                                               store.get_messages(cid), "standard")]
        events = asyncio.run(go())
        calls = sum(1 for e in events if e.type == "tool_call")
        assert calls <= 4, f"loop was not bounded: {calls} calls"
        assert any(e.type == "done" for e in events), "never terminated"
    finally:
        os.remove(path)


def test_agent_titles_conversations_sensibly():
    assert Agent.derive_title("hey can you help me build a rust web scraper") \
        == "Build a rust web scraper"
    assert Agent.derive_title("") == "New conversation"
    assert len(Agent.derive_title("x" * 200)) < 60


# --- runner ----------------------------------------------------------------

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    print(f"Praxis test suite — {len(tests)} tests\n")
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  \033[32mPASS\033[0m  {name}")
        except Exception as e:
            failed.append((name, e, traceback.format_exc()))
            print(f"  \033[31mFAIL\033[0m  {name}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("\n" + "=" * 66)
        for name, _, tb in failed:
            print(f"\n{name}\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
