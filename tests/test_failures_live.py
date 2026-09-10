"""
The four ways a turn came back empty — each one now a test.

Run:  python tests/test_failures_live.py

Every case here was a real failure in someone's hands, not a hypothetical.
A user attached a PDF, asked a question, and got a blank reply with no error
anywhere; asked a few questions in a row and got blanks again; and found his
own PDF pasted into his own message. What follows is those failures, reduced
to their mechanism and pinned down so they cannot come back.

  1. A reasoning model streams its thinking in one field and its answer in
     another. When the thinking uses up the output budget, the answer field
     never arrives, the stream closes cleanly, and the old code stored an
     empty string. Success was reported. Nothing said why.

  2. A free tier allows 8,000 tokens a minute. Two questions in quick
     succession is an HTTP 429, which is not a broken backend — it is a
     working one asking you to wait.

  3. "OpenAI-compatible" is a family resemblance, not a specification. Some
     endpoints refuse parameters others require.

  4. An attached file belongs in the prompt, not in the transcript.

The servers are real HTTP servers over real sockets; only the intelligence is
fake. See tests/fake_model_server.py for why that distinction is the point.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from praxis.agent import Agent                                   # noqa: E402
from praxis.config import Settings                               # noqa: E402
from praxis.providers import (EmptyAnswer, OpenAICompatibleProvider,   # noqa: E402
                              ProviderError, RateLimited)
from praxis.store import Store                                   # noqa: E402

# thinker: reasons forever, never answers.  limited: 429s twice, then works.
# strict:  refuses reasoning_effort.        walled: 429s more than we retry.
PORTS = {"thinker": 9111, "limited": 9112, "strict": 9113,
         "walled": 9114, "plain": 9115, "staller": 9116}
ARGS = {
    "thinker": ["--persona", "thinker"],
    "limited": ["--persona", "accurate", "--rate-limit", "2"],
    "strict":  ["--persona", "accurate", "--strict"],
    "walled":  ["--persona", "accurate", "--rate-limit", "99"],
    "plain":   ["--persona", "accurate"],
    "staller": ["--persona", "staller"],
}
_servers: list[subprocess.Popen] = []

QUESTION = [{"role": "user",
             "content": "Who was the Vice President under James K. Polk?"}]


def member(kind: str, **kw) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        base_url=f"http://127.0.0.1:{PORTS[kind]}/v1",
        api_key="local", model=kw.pop("model", f"gpt-oss-{kind}"))
    provider.name = provider.label = kind
    for key, value in kw.items():
        setattr(provider, key, value)
    return provider


async def drain(provider, messages=None) -> str:
    return "".join([chunk async for chunk in
                    provider.stream(messages or list(QUESTION))])


def start_servers() -> bool:
    script = ROOT / "tests" / "fake_model_server.py"
    for kind, port in PORTS.items():
        _servers.append(subprocess.Popen(
            [sys.executable, str(script), "--port", str(port), "--delay", "0",
             *ARGS[kind]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    import httpx
    for _ in range(40):
        time.sleep(0.4)
        try:
            for port in PORTS.values():
                httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
            return True
        except httpx.HTTPError:
            continue
    return False


def stop_servers() -> None:
    for proc in _servers:
        proc.terminate()
    for proc in _servers:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# --- 1. the blank reply ----------------------------------------------------

def test_a_model_that_only_thinks_raises_instead_of_returning_silence():
    """The bug exactly: reasoning arrives, content never does, stream ends OK.

    Returning "" here is what put a blank bubble on screen with a green tick
    and an elapsed time beside it."""
    try:
        text = asyncio.run(drain(member("thinker")))
    except EmptyAnswer as e:
        message = str(e)
        assert "budget" in message and "thinking" in message, message
        # It must say what to change, not merely that something went wrong.
        assert "REASONING_EFFORT" in message or "OPENAI_MAX_TOKENS" in message, message
        return
    raise AssertionError(f"returned {text!r} instead of raising")


def test_the_blank_reply_reaches_the_user_as_an_error_with_a_reason():
    """End to end: the agent must emit an error event, and store no blank."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False))

    async def go():
        cid = store.create_conversation("t")
        store.add_message(cid, "user", QUESTION[0]["content"])
        seen = []
        async for event in agent.run(member("thinker"), cid,
                                     store.get_messages(cid), "standard"):
            seen.append(event)
        return cid, seen

    try:
        cid, seen = asyncio.run(go())
        kinds = [e.type for e in seen]
        assert "error" in kinds, kinds
        detail = next(e.data["message"] for e in seen if e.type == "error")
        assert "thinking" in detail or "reasoning" in detail, detail
        # And nothing blank was written to the transcript.
        stored = [m for m in store.get_messages(cid) if m["role"] == "assistant"]
        assert not stored, f"stored a blank assistant turn: {stored}"
    finally:
        os.remove(path)


# --- 2. the rate limit -----------------------------------------------------

def test_a_rate_limit_is_waited_out_rather_than_failing_the_turn():
    """Two 429s, then an answer. The user should see a wait, not an error."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False,
                                  enable_mode_contracts=False,
                                  max_rate_limit_retries=3))

    async def go():
        cid = store.create_conversation("t")
        store.add_message(cid, "user", QUESTION[0]["content"])
        seen = []
        async for event in agent.run(member("limited"), cid,
                                     store.get_messages(cid), "standard"):
            seen.append(event)
        return cid, seen

    try:
        started = time.time()
        cid, seen = asyncio.run(go())
        elapsed = time.time() - started
        kinds = [e.type for e in seen]
        assert "error" not in kinds, [e.data for e in seen if e.type == "error"]
        waits = [e for e in seen if e.type == "rate_limited"]
        assert len(waits) == 2, f"expected two waits, saw {len(waits)}"
        # The wait is narrated with a number the interface can count down.
        assert waits[0].data["seconds"] >= 1, waits[0].data
        answer = store.get_messages(cid)[-1]
        assert answer["role"] == "assistant" and "Dallas" in answer["content"], answer
        # It really waited; it did not simply hammer the endpoint.
        assert elapsed >= 4.0, f"retried without waiting ({elapsed:.1f}s)"
    finally:
        os.remove(path)


def test_a_rate_limit_that_never_lifts_ends_in_an_error_not_a_blank():
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False,
                                  max_rate_limit_retries=1))

    async def go():
        cid = store.create_conversation("t")
        store.add_message(cid, "user", QUESTION[0]["content"])
        return [e async for e in agent.run(member("walled"), cid,
                                           store.get_messages(cid), "standard")]

    try:
        seen = asyncio.run(go())
        errors = [e for e in seen if e.type == "error"]
        assert errors, [e.type for e in seen]
        message = errors[0].data["message"]
        assert "rate limit" in message.lower(), message
        # And it did try again before giving up.
        assert any(e.type == "rate_limited" for e in seen), [e.type for e in seen]
        # Waiting did not work, so the message must say what would.
        assert "new conversation" in message, message
        assert "tokens to send" in message, message
    finally:
        os.remove(path)


def test_a_retry_sends_less_than_the_request_that_was_refused():
    """Waiting alone cannot fix a request that is bigger than the whole
    per-minute allowance. Each retry has to be cheaper than the last."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False,
                                  max_rate_limit_retries=2))

    async def go():
        cid = store.create_conversation("t")
        for i in range(6):
            store.add_message(cid, "user", f"question number {i}")
            store.add_message(cid, "assistant", f"answer number {i}")
        store.add_message(cid, "user", QUESTION[0]["content"])
        return [e async for e in agent.run(member("walled"), cid,
                                           store.get_messages(cid), "standard")]

    try:
        seen = asyncio.run(go())
        waits = [e for e in seen if e.type == "rate_limited"]
        assert len(waits) == 2, [e.type for e in seen]
        assert waits[0].data["dropped_turns"] > 0, waits[0].data
        # Each retry drops more, so the request keeps getting smaller.
        assert waits[1].data["dropped_turns"] > 0, waits[1].data
    finally:
        os.remove(path)


def test_shrinking_never_drops_the_question_being_asked():
    from praxis.agent import _shrink
    messages = [{"role": "system", "content": "identity"}]
    messages += [{"role": "user", "content": f"q{i}"} for i in range(5)]
    while _shrink(messages):
        pass
    assert [m["content"] for m in messages] == ["identity", "q4"], messages


def test_the_wait_length_comes_from_what_the_provider_asked_for():
    """Groq puts the wait in the body — "try again in 1.1s" — not only in the
    header. Ignoring it means either hammering the endpoint or over-waiting."""
    async def go():
        try:
            await drain(member("walled"))
        except RateLimited as e:
            return e
        return None

    error = asyncio.run(go())
    assert isinstance(error, RateLimited), error
    assert error.retry_after >= 1.0, error.retry_after
    assert "8000" in str(error), str(error)


# --- 3. the parameter nobody agrees on -------------------------------------

def test_an_endpoint_that_refuses_reasoning_effort_is_retried_without_it():
    """The user must never have to learn that this parameter exists."""
    provider = member("strict", reasoning_effort="low", model="gpt-oss-120b")
    text = asyncio.run(drain(provider))
    assert "Dallas" in text, text


def test_reasoning_effort_is_only_sent_to_models_that_think():
    thinker = member("plain", reasoning_effort="low", model="gpt-oss-120b")
    plain = member("plain", reasoning_effort="low", model="llama-3.3-70b-versatile")
    assert "reasoning_effort" in thinker.payload([], set())
    assert "reasoning_effort" not in plain.payload([], set())
    # And it can be turned off entirely.
    off = member("plain", reasoning_effort="", model="gpt-oss-120b")
    assert "reasoning_effort" not in off.payload([], set())


# --- 4. the attachment -----------------------------------------------------

def _with_file(store, cid, typed: str, text: str, name: str = "report.pdf"):
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text, encoding="utf-8")
    fid = store.add_file(name, str(path), "application/pdf", len(text),
                         text[:1500], cid)
    store.add_message(cid, "user", typed,
                      {"attachments": [{"id": fid, "filename": name,
                                        "characters": len(text)}]})
    return fid


def test_an_attachment_reaches_the_model_without_entering_the_transcript():
    """What the user typed stays what the user typed.

    The first version pasted the file's text into the message itself, so a
    two-page PDF turned the user's own bubble into two pages of PDF."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False))
    try:
        cid = store.create_conversation("t")
        body = "JAL-RAKSHAK is a water conservation project. " * 40
        _with_file(store, cid, "what is this and tell me its lacks", body)

        stored = store.get_messages(cid)[-1]
        assert stored["content"] == "what is this and tell me its lacks", stored

        messages, _, _ = agent.compose(store.get_messages(cid), "standard")
        prompt = messages[-1]["content"]
        assert prompt.startswith("what is this and tell me its lacks"), prompt[:80]
        assert "JAL-RAKSHAK" in prompt, "the model never saw the file"
        assert "report.pdf" in prompt, prompt[:200]
    finally:
        os.remove(path)


def test_a_huge_attachment_cannot_blow_the_prompt_budget():
    """A 400 KB PDF must not become a 400 KB prompt. That is how a free tier's
    tokens-per-minute allowance gets spent in a single turn."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    settings = Settings(enable_tools=False, enable_memory=False,
                        max_prompt_tokens=2000, attachment_chars=4000)
    agent = Agent(store, settings)
    try:
        cid = store.create_conversation("t")
        _with_file(store, cid, "summarise this", "lorem ipsum dolor sit amet. " * 20000)
        messages, _, _ = agent.compose(store.get_messages(cid), "standard")
        total = sum(len(m["content"]) for m in messages)
        # The budget, or the floor below which a request stops being worth
        # sending at all — whichever is larger. Never the size of the file.
        from praxis.agent import MIN_QUESTION_CHARS
        ceiling = max(settings.max_prompt_tokens * 4,
                      len(messages[0]["content"]) + MIN_QUESTION_CHARS) + 120
        assert total <= ceiling, f"{total} characters, ceiling {ceiling}"
        prompt = messages[-1]["content"]
        assert "lorem ipsum" in prompt, "the model got none of the file"
        assert "read_file" in prompt, "nothing told it how to reach the rest"
    finally:
        os.remove(path)


def test_older_attachments_are_dropped_before_the_question_being_asked():
    """When the budget runs out, the newest turn survives. Answering the wrong
    question is worse than answering with less context."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False,
                                  max_prompt_tokens=1200, attachment_chars=3000))
    try:
        cid = store.create_conversation("t")
        _with_file(store, cid, "first question", "OLDFILE " * 2000, "old.pdf")
        store.add_message(cid, "assistant", "an answer to the first question")
        store.add_message(cid, "user", "and what about the second thing?")
        messages, _, _ = agent.compose(store.get_messages(cid), "standard")
        assert messages[-1]["content"] == "and what about the second thing?"
        assert "OLDFILE" not in "".join(m["content"] for m in messages)
    finally:
        os.remove(path)


def test_the_question_survives_even_when_it_is_not_the_newest_turn():
    """Found by composing a finished conversation, where the last turn is the
    assistant's. Walking backwards and keeping "the first thing reached" kept
    the *answer* and dropped the question that produced it — so a big
    attachment could push the user's own words out of their own prompt."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False,
                                  max_prompt_tokens=900, attachment_chars=2500))
    try:
        cid = store.create_conversation("t")
        _with_file(store, cid, "what is this and what does it lack?",
                   "JAL-RAKSHAK is a water conservation project. " * 200)
        store.add_message(cid, "assistant", "a previous answer")
        messages, _, _ = agent.compose(store.get_messages(cid), "standard")
        prompt = " ".join(m["content"] for m in messages if m["role"] == "user")
        assert "what is this and what does it lack?" in prompt, [
            (m["role"], m["content"][:60]) for m in messages]
        assert "JAL-RAKSHAK" in prompt, "the attachment was dropped entirely"
        assert "read_file" in prompt, "nothing pointed at the rest of the file"
    finally:
        os.remove(path)


def test_the_tool_list_shrinks_when_the_budget_is_small():
    """Fourteen tools described in full is ~700 tokens on every request. On a
    free tier that is a quarter of the minute's allowance, spent before the
    user types."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    try:
        cid = store.create_conversation("t")
        store.add_message(cid, "user", "hello")
        tight = Agent(store, Settings(max_prompt_tokens=4000))
        roomy = Agent(store, Settings(max_prompt_tokens=32000))
        small = tight.compose(store.get_messages(cid), "standard")[0][0]["content"]
        large = roomy.compose(store.get_messages(cid), "standard")[0][0]["content"]
        assert len(small) < len(large), (len(small), len(large))
        # Compact or not, every tool is still offered by name.
        for name in ("calculate", "read_file", "web_search", "remember"):
            assert name in small, name
    finally:
        os.remove(path)


# --- 5. the mode that keeps breaking its promise ---------------------------

def test_a_mode_that_never_satisfies_its_contract_still_returns_an_answer():
    """Founder mode wants something testable this week. The stub answer never
    provides one, so every attempt breaches — and the user must still end up
    with the answer plus an honest note, never an empty bubble."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False,
                                  enable_mode_contracts=True,
                                  max_constraint_retries=1))

    async def go():
        cid = store.create_conversation("t")
        store.add_message(cid, "user", "should I build a water app?")
        seen = [e async for e in agent.run(member("plain"), cid,
                                           store.get_messages(cid), "founder")]
        return cid, seen

    try:
        cid, seen = asyncio.run(go())
        answer = store.get_messages(cid)[-1]
        assert answer["role"] == "assistant"
        assert answer["content"].strip(), "founder mode stored an empty answer"
        assert any(e.type == "mode_breach" for e in seen), [e.type for e in seen]
        assert "founder mode" in answer["content"].lower(), answer["content"][-200:]
    finally:
        os.remove(path)


# --- 6. don't stall --------------------------------------------------------

def test_an_answer_that_stops_at_uncertainty_is_sent_back():
    """The failure this product is named against: honest, useless, and
    comfortable about it. "I cannot confidently determine this" is a complete
    sentence and a complete failure."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, Settings(enable_tools=False, enable_memory=False,
                                  enable_mode_contracts=False,
                                  enable_momentum=True,
                                  max_constraint_retries=1))

    async def go():
        cid = store.create_conversation("t")
        store.add_message(cid, "user", "Should we use Postgres or MySQL?")
        seen = [e async for e in agent.run(member("staller"), cid,
                                           store.get_messages(cid), "standard")]
        return cid, seen

    try:
        cid, seen = asyncio.run(go())
        kinds = [e.type for e in seen]
        assert "stalled" in kinds, kinds

        stalled = next(e for e in seen if e.type == "stalled")
        # It quotes the sentence it stopped on, not a general complaint.
        assert "cannot confidently determine" in stalled.data["quote"], stalled.data

        answer = store.get_messages(cid)[-1]["content"]
        assert "I'd go with" in answer, answer[:160]
        assert "measure them before you commit" in answer, answer[:160]
        # And the honesty survived the repair — this is not a check that turns
        # an uncertain answer into a confident one.
        assert "not certain" in answer.lower(), answer[:160]
    finally:
        os.remove(path)


def test_a_safety_refusal_is_never_pushed_on():
    """A repair loop that argues with a refusal is a loop that eventually gets
    one overturned. Refusals are exempt, in the checker itself, not by luck."""
    from praxis import momentum
    refusals = [
        "I can't help with building a credential-stuffing tool. If you are "
        "testing your own login flow I can help you write a rate-limit test.",
        "I have to decline that one.",
        "I won't write malware. That's not something I can help with.",
    ]
    for text in refusals:
        assert momentum.check(text) == [], text[:60]


def test_an_honest_answer_that_keeps_moving_is_left_alone():
    """The target behaviour must never be nagged. A false accusation here costs
    a whole regeneration and teaches the model that hedging is punished, which
    is the opposite of the intent."""
    from praxis import momentum
    good = [
        "I'd choose Postgres. I'm not certain about your JSON query volume — "
        "load a day of real traffic into both and compare.",
        "I don't know your p99 offhand. Run pg_stat_statements and check the "
        "top five queries; that answers it in ten minutes.",
        "It depends on whether writes or reads dominate. If writes, Postgres.",
        "Canberra.",
        "I'm not sure that library is still maintained — my best guess is it "
        "isn't, so check the last commit date before building on it.",
    ]
    nagged = [t[:50] for t in good if momentum.check(t)]
    assert not nagged, f"falsely flagged as stalling: {nagged}"


def test_the_empty_answer_check_owns_blankness_not_this_one():
    """Two checks reporting one fault helps nobody."""
    from praxis import momentum
    assert momentum.check("") == []
    assert momentum.check("   \n  ") == []


# --- 7. modes must actually differ -----------------------------------------

def _system_for(store, mode: str, **settings) -> str:
    agent = Agent(store, Settings(enable_memory=False, **settings))
    cid = store.create_conversation("t")
    store.add_message(cid, "user", "Should I rewrite our billing service in Go?")
    return agent.compose(store.get_messages(cid), mode)[0][0]["content"]


def test_every_mode_sends_a_different_system_prompt():
    """If two modes send the same prompt they cannot produce different answers,
    and the mode picker is decoration."""
    from praxis import modes as modes_mod
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    try:
        prompts = {k: _system_for(store, k) for k in modes_mod.MODES}
        clashes = [(a, b) for i, a in enumerate(prompts)
                   for b in list(prompts)[i + 1:] if prompts[a] == prompts[b]]
        assert not clashes, f"identical system prompts: {clashes}"
        # And each carries its own overlay verbatim, not a paraphrase of it.
        for key, mode in modes_mod.MODES.items():
            if mode.prompt.strip():
                assert mode.prompt.strip() in prompts[key], \
                    f"{key}'s overlay never reached the prompt"
    finally:
        os.remove(path)


def test_the_mode_overlay_survives_the_tightest_prompt_budget():
    """Trimming for a free tier must never trim away the thing that makes the
    mode a mode. History is negotiable; identity is not."""
    from praxis import modes as modes_mod
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    try:
        for key, mode in modes_mod.MODES.items():
            if not mode.prompt.strip():
                continue
            system = _system_for(store, key, max_prompt_tokens=600)
            assert mode.prompt.strip() in system, \
                f"{key}'s overlay was trimmed away at a 600-token budget"
    finally:
        os.remove(path)


def test_every_mode_that_makes_a_checkable_promise_has_a_contract():
    """A promise nothing checks is a promise the product does not keep. Standard
    is the deliberate exception: it promises nothing beyond the core identity."""
    from praxis import contracts as contracts_mod
    from praxis import modes as modes_mod
    unchecked = [k for k in modes_mod.MODES
                 if k != "standard" and not contracts_mod.describe(k)]
    assert not unchecked, f"modes with no enforced promise: {unchecked}"


def test_a_contract_never_fires_on_an_answer_that_honours_the_mode():
    """The other half of the same coin: a false accusation costs a whole
    regeneration and teaches the user the checker is noise."""
    from praxis import contracts as contracts_mod
    good = {
        "direct": "Yes. Go's concurrency suits billing throughput, and your "
                  "team already ships Go. Start with the ledger writer.",
        "brief": ("**Situation** Billing runs on a five-year-old Ruby service.\n"
                  "**Problem** It cannot keep up at month end.\n"
                  "**Options** Rewrite in Go, shard the database, or buy.\n"
                  "**Recommendation** Rewrite the ledger writer in Go first.\n"
                  "**Next action** Spike the writer this week and measure."),
        "reality": "TEST FIRST. The throughput claim is unproven, and a rewrite "
                   "is the most expensive way to find out you were wrong.",
        "socratic": "What part of the current service is actually the "
                    "bottleneck — the language, or the database round trips?",
        "exam": "Before we decide: what is your p99 latency at month end?",
        "founder": "Rewrite the ledger writer only. This week, replay one day "
                   "of production traffic against a Go prototype and measure.",
        "build": ("Use Go 1.23 with pgx v5.\n\n```go\nfunc main() { "
                  "ledger.Serve() }\n```\n\nRun it: `go run ./cmd/ledger`"),
        "teacher": ("Start with what a ledger writer does: it appends, it never "
                    "updates. Why do you think that matters for throughput?"),
        "research": ("What is established: Go handles concurrent writes well. "
                     "What is contested: whether the language is your "
                     "bottleneck. What is unknown: your actual p99."),
    }
    wrong = []
    for mode, answer in good.items():
        breaches = contracts_mod.check(mode, answer)
        if breaches:
            wrong.append(f"{mode}: falsely accused of "
                         f"{[b.detail for b in breaches]}")
    assert not wrong, "\n  " + "\n  ".join(wrong)


def test_a_contract_does_fire_when_the_mode_is_ignored():
    """An answer that ignores the mode entirely must be caught by every mode
    that promises something checkable."""
    from praxis import contracts as contracts_mod
    from praxis import modes as modes_mod
    ignored = ("Well, that depends on a great many things, and reasonable "
               "people differ. " * 30)
    missed = [k for k in modes_mod.MODES
              if contracts_mod.describe(k) and not contracts_mod.check(k, ignored)]
    assert not missed, f"these contracts let a mode-ignoring answer through: {missed}"


# --- runner ----------------------------------------------------------------

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"Failure suite — {len(tests)} tests against real HTTP servers\n")
    print("  starting model servers…")
    if not start_servers():
        print("  FAILED to start the fake model servers")
        stop_servers()
        return 1
    print(f"  up on ports {sorted(PORTS.values())}\n")

    passed, failed = 0, []
    try:
        for name, fn in tests:
            try:
                fn()
                passed += 1
                print(f"  \033[32mPASS\033[0m  {name}")
            except Exception as e:
                failed.append((name, traceback.format_exc()))
                print(f"  \033[31mFAIL\033[0m  {name}: {e}")
    finally:
        stop_servers()

    print(f"\n{passed}/{len(tests)} passed")
    for name, tb in failed:
        print(f"\n{name}\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
