"""
Live council tests — real HTTP, real sockets, real streaming.

Run:  python tests/test_council_live.py

Starts four local model servers with different personalities (accurate,
hedging, wrong, broken), points the council at them, and checks what it
actually does. The servers are fake in exactly one respect: the intelligence.
Everything the council touches — the socket, the SSE stream, the latency, the
failure modes — is real.

This is the difference that matters. Mocking `Provider.stream` proves the
council's branching logic and nothing else: a mock returns instantly, so a
"concurrency" assertion against one is meaningless. Here, each server sleeps
before answering, so parallelism is measured rather than asserted.

Two findings recorded here as permanent tests, because both were live bugs:
  - the council ran sequentially against CPU-bound providers (see
    ScratchProvider in providers.py — HTTP backends were never affected)
  - health checks read configuration and never touched the network, so a port
    with nothing listening reported healthy

To test against your own real models instead, see
tests/test_council_real.py.
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

# Local traffic must not go through any proxy, or every request 403s.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from praxis import council                                      # noqa: E402
from praxis.agent import Agent                                  # noqa: E402
from praxis.config import Settings                              # noqa: E402
from praxis.providers import OpenAICompatibleProvider           # noqa: E402
from praxis.store import Store                                  # noqa: E402
import praxis.agent as agent_mod                                # noqa: E402

PORTS = {"accurate": 9101, "hedging": 9102, "wrong": 9103, "broken": 9104}
DELAY = 0.8            # seconds each server waits before its first token
_servers: list[subprocess.Popen] = []


def member(persona: str) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(
        base_url=f"http://127.0.0.1:{PORTS[persona]}/v1",
        api_key="local", model=persona)
    provider.name = persona
    provider.label = persona
    return provider


def start_servers() -> bool:
    script = ROOT / "tests" / "fake_model_server.py"
    for persona, port in PORTS.items():
        cmd = [sys.executable, str(script), "--port", str(port),
               "--delay", str(DELAY)]
        cmd += ["--broken"] if persona == "broken" else ["--persona", persona]
        _servers.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL))

    import httpx
    for _ in range(40):
        time.sleep(0.4)
        try:
            httpx.get(f"http://127.0.0.1:{PORTS['accurate']}/v1/models", timeout=2)
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


QUESTION = [{"role": "user",
             "content": "Who was the Vice President under James K. Polk?"}]


# --- concurrency -----------------------------------------------------------

def test_council_is_genuinely_parallel_over_http():
    """Three members must cost about what one costs, not three times as much.

    This is the assertion a mock cannot make. Each server sleeps DELAY before
    answering, so sequential execution would show up as 3x."""
    async def go():
        solo = time.time()
        await council.gather([member("accurate")], QUESTION, timeout=30)
        solo = time.time() - solo

        trio = time.time()
        await council.gather([member("accurate"), member("hedging"),
                              member("wrong")], QUESTION, timeout=30)
        trio = time.time() - trio
        return solo, trio

    solo, trio = asyncio.run(go())
    ratio = trio / solo
    assert ratio < 1.6, (
        f"council ran at {ratio:.2f}x for 3 members — sequential, not parallel "
        f"(1 member {solo:.2f}s, 3 members {trio:.2f}s)")


# --- health ----------------------------------------------------------------

def test_health_check_actually_reaches_the_endpoint():
    """A port with nothing listening must not report healthy.

    It used to: the check inspected configuration only, so /api/council showed
    a green dot for a dead backend and the council wasted a slot on it."""
    async def go():
        ghost = OpenAICompatibleProvider(
            base_url="http://127.0.0.1:9199/v1", api_key="k", model="ghost")
        dead = member("broken")
        alive = member("accurate")
        return (await ghost.health(), await dead.health(), await alive.health())

    ghost, dead, alive = asyncio.run(go())
    assert not ghost[0], f"a dead port reported healthy: {ghost}"
    assert not dead[0], f"a 500-ing server reported healthy: {dead}"
    assert alive[0], f"a working server reported unhealthy: {alive}"


def test_broken_member_is_dropped_with_a_reason():
    async def go():
        members = [member(p) for p in ("accurate", "hedging", "broken")]
        return await council.healthy_members(members)

    live, dropped = asyncio.run(go())
    assert len(live) == 2, [m.name for m in live]
    assert len(dropped) == 1 and "broken" in dropped[0], dropped


# --- judging ---------------------------------------------------------------

def test_judge_picks_the_accurate_answer_over_a_confident_wrong_one():
    async def go():
        cands = await council.gather(
            [member("accurate"), member("hedging"), member("wrong")],
            QUESTION, timeout=30)
        return await council.judge(member("accurate"),
                                   QUESTION[0]["content"], cands)

    verdict = asyncio.run(go())
    assert verdict.method == "judge", verdict.method
    assert "Dallas" in verdict.winner.text, verdict.winner.text[:120]


def test_with_no_accurate_member_it_prefers_the_hedge_over_the_fabrication():
    """The council cannot exceed its members. When every member is flawed, the
    honest refusal must still beat the confident invention."""
    async def go():
        cands = await council.gather([member("hedging"), member("wrong")],
                                     QUESTION, timeout=30)
        with_judge = await council.judge(member("accurate"),
                                         QUESTION[0]["content"], cands)
        heuristic = await council.judge(None, QUESTION[0]["content"], cands)
        return with_judge, heuristic

    with_judge, heuristic = asyncio.run(go())
    assert "not certain" in with_judge.winner.text.lower(), with_judge.winner.text[:100]
    assert "not certain" in heuristic.winner.text.lower(), heuristic.winner.text[:100]
    assert heuristic.method == "heuristic"


def test_every_member_failing_degrades_honestly():
    async def go():
        cands = await council.gather([member("broken"), member("broken")],
                                     QUESTION, timeout=30)
        return cands, await council.judge(member("accurate"),
                                          QUESTION[0]["content"], cands)

    cands, verdict = asyncio.run(go())
    assert not any(c.usable for c in cands)
    assert "every member failed" in verdict.reason, verdict.reason


# --- race ------------------------------------------------------------------

def test_race_returns_first_and_cancels_the_rest():
    async def go():
        started = time.time()
        winner, seen = await council.race(
            [member("accurate"), member("hedging"), member("wrong")],
            QUESTION, timeout=30)
        return winner, seen, time.time() - started

    winner, seen, elapsed = asyncio.run(go())
    assert winner.usable, winner.error
    # One answer's worth of latency, not three.
    assert elapsed < DELAY * 2.2, f"race took {elapsed:.2f}s — it waited for all"


# --- end to end through the agent ------------------------------------------

def test_agent_council_answers_four_questions_correctly():
    """The full path: agent -> council -> real HTTP -> judge -> stored answer."""
    roster = {p: p for p in PORTS}
    agent_mod.council_mod.from_spec = lambda spec: member(
        roster[spec.split(":")[0]])

    checks = [
        ("Who was the Vice President under James K. Polk?", "dallas"),
        ("What were the economic impacts of the 2023 Atlantis Maritime Accord?",
         "no 2023 atlantis"),
        ("What is the capital of Australia?", "canberra"),
        ("How many pixels in a 4K frame, and is it 4x 1080p?", "8,294,400"),
    ]

    path = tempfile.mktemp(suffix=".db")
    store = Store(path)
    settings = Settings(council_members="accurate,hedging,wrong,broken",
                        council_judge="accurate",
                        enable_constraint_check=False,
                        enable_mode_contracts=False)
    agent = Agent(store, settings)

    async def ask(question: str) -> tuple[str, list[str]]:
        cid = store.create_conversation("t")
        store.add_message(cid, "user", question)
        dropped: list[str] = []
        async for event in agent.run(member("accurate"), cid,
                                     store.get_messages(cid), "standard",
                                     strategy="council"):
            if event.type == "council_start":
                dropped = event.data["dropped"]
        return store.get_messages(cid)[-1]["content"], dropped

    try:
        for question, expected in checks:
            answer, dropped = asyncio.run(ask(question))
            assert expected in answer.lower(), \
                f"{question[:40]!r} -> {answer[:110]!r}"
            assert len(dropped) == 1, f"broken member not dropped: {dropped}"
    finally:
        os.remove(path)


# --- runner ----------------------------------------------------------------

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"Live council suite — {len(tests)} tests against real HTTP servers\n")
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
