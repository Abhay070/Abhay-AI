"""
Reliability — the fourth axis. Did Praxis recover when something went wrong?

    python tests/test_reliability.py

Machinery, structure and substance are what the 100-question benchmark measures,
and all three assume the happy path: a model that answers. This measures the
unhappy path, which is where an execution layer actually earns its name. A raw
model has no unhappy path — it generates or it errors. Praxis is supposed to
observe, verify and recover, and a claim to recover that is never tested under
real failure is just a comment.

So each scenario here injects a genuine fault through a real HTTP server — a
provider that 429s, one that dies mid-stream, one that only ever thinks, one
whose answer breaks the constraint it was given, a council where every member
is weak — and asserts on what the user ends up with. The bar is not "it
succeeded". Some of these cannot succeed; the model really is broken. The bar
is that the failure is handled the way the philosophy says it must be:

  - never a blank reply reported as success,
  - never a fabricated compliance,
  - never a silent stall,
  - never a crash reaching the user as a bare 500,
  - and, wherever a path forward exists, taken.

A reliability score is the fraction of injected failures that were handled,
not the fraction that were made to disappear. Recovering from a fault and
reporting a fault honestly both count. Only pretending a fault did not happen
fails.
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
from praxis.providers import OpenAICompatibleProvider            # noqa: E402
from praxis.store import Store                                   # noqa: E402
import praxis.agent as agent_mod                                 # noqa: E402

GREEN, RED, YELLOW, DIM, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")

PORTS = {"thinker": 9121, "walled": 9122, "broken": 9123,
         "staller": 9124, "wrong": 9125, "accurate": 9126, "hedging": 9127,
         "spent": 9128, "spare": 9129, "limited": 9130}
ARGS = {
    "thinker": ["--persona", "thinker"],
    "walled":  ["--persona", "accurate", "--rate-limit", "99"],
    "broken":  ["--broken"],
    "staller": ["--persona", "staller"],
    "wrong":   ["--persona", "wrong"],
    "accurate": ["--persona", "accurate"],
    "hedging": ["--persona", "hedging"],
    # Out of quota for the day — waiting cannot help, only another backend can.
    "spent":   ["--persona", "accurate", "--rate-limit", "99",
                "--rate-limit-window", "day"],
    "spare":   ["--persona", "accurate"],
    "limited": ["--persona", "accurate", "--rate-limit", "1"],
}
_servers: list[subprocess.Popen] = []


def member(kind: str) -> OpenAICompatibleProvider:
    p = OpenAICompatibleProvider(base_url=f"http://127.0.0.1:{PORTS[kind]}/v1",
                                 api_key="local", model=f"m-{kind}")
    p.name = p.label = kind
    return p


def start_servers() -> bool:
    script = ROOT / "tests" / "fake_model_server.py"
    for kind, port in PORTS.items():
        _servers.append(subprocess.Popen(
            [sys.executable, str(script), "--port", str(port), "--delay", "0",
             *ARGS[kind]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
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


def run(provider, question: str, settings: Settings, strategy: str = "single",
        primary: str = "accurate") -> tuple[str, list]:
    """Ask one question and return (stored answer, events)."""
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    agent = Agent(store, settings)

    async def go():
        cid = store.create_conversation("t")
        store.add_message(cid, "user", question)
        events = [e async for e in agent.run(provider, cid,
                                             store.get_messages(cid), "standard",
                                             strategy=strategy)]
        answer = store.get_messages(cid)[-1]
        return (answer["content"] if answer["role"] == "assistant" else ""), events

    try:
        return asyncio.run(go())
    finally:
        os.remove(path)


# Each scenario returns (handled: bool, note: str). Handled means the failure
# was recovered OR reported honestly — never hidden.
SCENARIOS = {}


def scenario(label):
    def wrap(fn):
        SCENARIOS[label] = fn
        return fn
    return wrap


@scenario("a model that only thinks never yields a blank success")
def s_thinker():
    answer, events = run(member("thinker"), "What is the capital of France?",
                         Settings(enable_tools=False, enable_memory=False))
    kinds = [e.type for e in events]
    if "error" not in kinds:
        return False, f"no error event: {kinds}"
    if answer.strip() and "empty" not in answer.lower() and "thinking" not in answer.lower():
        return False, f"a blank/thinking failure was stored as an answer: {answer[:80]}"
    return True, "raised an error naming the cause"


@scenario("a rate limit that never lifts is reported, not hidden")
def s_walled():
    answer, events = run(member("walled"), "What is 2+2?",
                         Settings(enable_tools=False, enable_memory=False,
                                  max_rate_limit_retries=1))
    errs = [e for e in events if e.type == "error"]
    if not errs:
        return False, "no error surfaced"
    if "rate limit" not in errs[0].data["message"].lower():
        return False, "error did not mention the rate limit"
    if not any(e.type == "rate_limited" for e in events):
        return False, "gave up without retrying"
    if "new conversation" not in errs[0].data["message"]:
        return False, "did not tell the user how to fix it"
    return True, "retried, then explained the fix"


@scenario("a mid-stream crash reaches the user as a message, not a 500")
def s_broken():
    answer, events = run(member("broken"), "Hello",
                         Settings(enable_tools=False, enable_memory=False))
    if not any(e.type == "error" for e in events):
        return False, "the crash produced no error event"
    return True, "surfaced as an error event"


@scenario("an unfixable constraint is flagged, never faked")
def s_constraint():
    # gpt-oss-style accurate model, asked for something it will not satisfy: a
    # letter exclusion the scripted answer ignores.
    answer, events = run(
        member("accurate"),
        "Describe water. Do NOT use the letter 'e' anywhere.",
        Settings(enable_tools=False, enable_memory=False,
                 enable_constraint_check=True, max_constraint_retries=1))
    if "does not meet what you asked" not in answer.lower() \
            and "check failed" not in answer.lower():
        return False, f"a violating answer was not flagged: {answer[:80]}"
    return True, "flagged the unmet constraint honestly"


@scenario("a stall is turned into a next step")
def s_stall():
    answer, events = run(member("staller"), "Postgres or MySQL for us?",
                         Settings(enable_tools=False, enable_memory=False,
                                  enable_momentum=True, max_constraint_retries=1))
    if not any(e.type == "stalled" for e in events):
        return False, "the stall was not detected"
    if "cannot confidently determine" in answer.lower() \
            and "measure" not in answer.lower():
        return False, "the answer still just stalls"
    return True, "detected the stall and repaired it"


@scenario("a council of only-weak members returns no consensus")
def s_no_consensus():
    # Both members hedge — the field is uniformly weak, which is the case
    # no-consensus is built for. (What it deliberately cannot catch is a
    # *confident fabrication* sitting among the answers: detecting that a
    # specific, assertive answer is false needs the ground truth, which the
    # council does not have. That is a real limit, honestly out of scope for a
    # weakness heuristic, and not something this scenario pretends to test.)
    agent_mod.council_mod.from_spec = lambda spec: member(spec.split(":")[0])
    answer, events = run(
        member("accurate"), "What will our Q3 revenue be?",
        Settings(council_members="staller,hedging", council_judge="accurate",
                 enable_constraint_check=False, enable_mode_contracts=False,
                 enable_momentum=False),
        strategy="council")
    verdicts = [e for e in events if e.type == "council_verdict"]
    if not verdicts or not verdicts[0].data.get("no_consensus"):
        return False, "crowned a winner from a weak field"
    if "NO CONSENSUS" not in answer:
        return False, "the answer did not carry the no-consensus label"
    if len(answer.strip()) < 40:
        return False, "withheld the answer entirely"
    return True, "declined to crown, showed the best anyway"


@scenario("a dead primary fails over to a live backend")
def s_failover():
    # resolve() is what the server uses; here we prove the chain picks the live
    # one. A broken primary must not strand the user when a spare is reachable.
    from praxis import providers
    async def go():
        # Monkeypatch get_provider to hand back our fakes by name.
        real = providers.get_provider
        providers.get_provider = lambda n: member(
            "broken" if n == "broken" else "accurate")
        try:
            prov, notes = await providers.resolve("broken", ("accurate",))
            ok, _ = await prov.health()
            return prov, notes, ok
        finally:
            providers.get_provider = real
    prov, notes, ok = asyncio.run(go())
    if prov.name != "accurate" or not ok:
        return False, f"did not fail over (landed on {prov.name}, ok={ok})"
    if not notes:
        return False, "failed over but said nothing about the dead primary"
    return True, "skipped the dead primary, noted why"


def _pool_run(pool: str, question: str, **extra):
    """Run one turn with a real router over a real pool of HTTP servers."""
    from praxis.router import Router
    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    settings = Settings(enable_tools=False, enable_memory=False,
                        enable_mode_contracts=False, enable_momentum=False,
                        max_rate_limit_retries=1, capacity_pool=pool, **extra)
    router = Router(settings, store)
    # The pool specs name personas; build them against the local fake servers.
    router.build = lambda slot: member(slot.spec.split(":")[0])
    agent = Agent(store, settings, router=router)

    async def go():
        cid = store.create_conversation("t")
        store.add_message(cid, "user", question)
        first = router.pick()
        provider = router.build(first)
        events = [e async for e in agent.run(provider, cid,
                                             store.get_messages(cid), "standard")]
        answer = store.get_messages(cid)[-1]
        return (answer["content"] if answer["role"] == "assistant" else ""), events, router

    try:
        return asyncio.run(go())
    finally:
        os.remove(path)


@scenario("a daily quota hands the turn to the next backend")
def s_daily_failover():
    answer, events, _ = _pool_run("spent:m,spare:m", "What is the capital of Australia?")
    kinds = [e.type for e in events]
    if "provider_switch" not in kinds:
        return False, f"no failover happened: {kinds}"
    if "error" in kinds:
        return False, "the turn still failed"
    if not answer.strip():
        return False, "no answer was stored"
    switch = next(e for e in events if e.type == "provider_switch")
    if switch.data.get("reason") != "day":
        return False, f"misread the limit as {switch.data.get('reason')}"
    return True, f"switched on a daily cap and answered ({len(answer)} chars)"


@scenario("a daily quota is never waited out or paid for with history")
def s_daily_no_wait():
    started = time.time()
    answer, events, _ = _pool_run("spent:m,spare:m", "What is the capital of Australia?")
    elapsed = time.time() - started
    # The old code clamped a 6-hour reset to 45s and retried three times.
    if elapsed > 10:
        return False, f"waited {elapsed:.0f}s against a daily cap"
    if any(e.type == "rate_limited" for e in events):
        return False, "treated a daily cap as a waitable per-minute limit"
    return True, f"failed over in {elapsed:.1f}s without waiting or shrinking"


@scenario("a per-minute limit still waits rather than burning a backend")
def s_minute_still_waits():
    answer, events, _ = _pool_run("limited:m,spare:m", "What is 2+2?")
    kinds = [e.type for e in events]
    if "rate_limited" not in kinds:
        return False, f"did not wait out a per-minute limit: {kinds}"
    if not answer.strip():
        return False, "no answer"
    return True, "waited and retried the same backend, as it should"


@scenario("a whole pool spent says so, naming every backend and its reset")
def s_pool_exhausted():
    answer, events, router = _pool_run("spent:m", "What is the capital of Australia?")
    errors = [e for e in events if e.type == "error"]
    if not errors:
        return False, "no error surfaced"
    message = errors[0].data["message"]
    for needed in ("Out of capacity", "back in", "CAPACITY_POOL"):
        if needed.lower() not in message.lower():
            return False, f"message missing {needed!r}"
    return True, "named each backend, its reset, and how to widen the pool"


@scenario("a cooldown survives a restart")
def s_cooldown_persists():
    from praxis.router import Router
    path = tempfile.mktemp(suffix=".db")
    try:
        settings = Settings(capacity_pool="spent:m,spare:m")
        store = Store(path)
        router = Router(settings, store)
        slot = router.slots()[0]

        class DailyLimit:
            window, reset_at, retry_after, source = "day", time.time() + 6 * 3600, 0, "body"
        router.note_rate_limit(slot, DailyLimit())

        # A fresh process, same file. A daily quota does not reset because the
        # server did, and an in-memory ledger would forget that.
        reborn = Router(settings, Store(path))
        if reborn.standing(reborn.slots()[0]).available(time.time()):
            return False, "the cooldown was forgotten on restart"
        if reborn.pick().spec != "spare:m":
            return False, "did not skip the benched backend after restart"
        return True, "cooldown reloaded from disk and respected"
    finally:
        os.remove(path)


def main() -> int:
    print("Reliability suite — recovery under injected failure\n")
    print("  starting model servers…")
    if not start_servers():
        print("  FAILED to start servers")
        stop_servers()
        return 1
    print(f"  up on {sorted(PORTS.values())}\n")

    handled = 0
    results = []
    try:
        for label, fn in SCENARIOS.items():
            try:
                ok, note = fn()
            except Exception as e:
                ok, note = False, f"scenario crashed: {type(e).__name__}: {e}"
            results.append((label, ok, note))
            handled += ok
            mark = f"{GREEN}HANDLED{OFF}" if ok else f"{RED}UNHANDLED{OFF}"
            print(f"  {mark}  {label}")
            print(f"           {DIM}{note}{OFF}")
    finally:
        stop_servers()

    total = len(results)
    pct = round(100 * handled / total) if total else 0
    colour = GREEN if handled == total else (YELLOW if handled >= total * 0.7 else RED)
    print(f"\n  Reliability: {colour}{handled}/{total} failures handled "
          f"({pct}%){OFF}")
    print(f"  {DIM}Handled = recovered OR reported honestly. Only hiding a "
          f"fault counts as unhandled.{OFF}")
    return 0 if handled == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
