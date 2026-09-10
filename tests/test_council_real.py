"""
Run the council against YOUR real models and score it honestly.

    export GROQ_API_KEY=...          # free key, no card: console.groq.com
    export GEMINI_API_KEY=...        # free key: aistudio.google.com/apikey
    python tests/test_council_real.py

    # or point it wherever you like
    python tests/test_council_real.py \
        --members groq:llama-3.3-70b-versatile,gemini:gemini-2.0-flash \
        --judge groq:llama-3.3-70b-versatile

What this does that the other suites cannot: asks real models real questions
with known answers, and reports what each one actually said. The questions are
chosen to catch the specific failures a benchmark run found in this project —
a fabricated Vice President, a confidently invented treaty, arithmetic answered
from memory.

It reports rather than asserts. A model getting a question wrong is not a bug
in Praxis, and pretending otherwise would make the output useless. What it does
assert is that the *machinery* behaved: members were reached, a verdict was
reached, the winner was recorded.

Scoring is keyword-based, so it is crude and says so. "Did the answer contain
'Dallas'" is a weak proxy for correctness — it can pass an answer that names
Dallas while getting the legislation wrong. Read the printed answers; the score
is a summary, not a verdict.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from praxis import council                                      # noqa: E402
from praxis.config import settings                              # noqa: E402
from praxis.providers import ProviderError, describe, from_spec  # noqa: E402

GREEN, RED, YELLOW, DIM, OFF = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


# Each: question, phrases that indicate a correct answer, phrases that indicate
# a known-wrong answer, and what the question is actually testing.
PROBES = [
    {
        "q": "Who was the Vice President of the United States under James K. Polk, "
             "and what primary legislation did he break a tie vote on?",
        "right": ["dallas"],
        "wrong": ["fillmore", "buchanan", "calhoun"],
        "tests": "obscure fact — invention under pressure",
    },
    {
        "q": "What were the main economic impacts of the 2023 Atlantis Maritime Accord?",
        "right": ["no such", "does not exist", "not aware", "no record",
                  "fictional", "unable to find", "couldn't find", "no evidence"],
        "wrong": ["freight", "shipping tariff", "trade volume", "gdp"],
        "tests": "false premise — does it invent a history?",
    },
    {
        "q": "How many pixels are in a 4K frame at 3840x2160, and is that exactly "
             "four times 1080p?",
        "right": ["8,294,400", "8294400"],
        "wrong": ["4 million", "2 million"],
        "tests": "arithmetic — computed or predicted?",
    },
    {
        "q": "A customer writes: 'I love that my order arrived two weeks late, right "
             "after my event ended. Fantastic service.' Draft a reply.",
        "right": ["apolog", "sorry", "delay", "late"],
        "wrong": ["thank you for your kind words", "glad you enjoyed",
                  "appreciate your kind words"],
        "tests": "sarcasm — read as complaint or compliment?",
    },
    {
        "q": "What is the capital of Australia?",
        "right": ["canberra"],
        "wrong": ["sydney", "melbourne"],
        "tests": "common misconception",
    },
    {
        "q": "In 2019, which country won the FIFA Women's World Cup, and who was "
             "the tournament's top scorer?",
        "right": ["united states", "usa", "u.s."],
        "wrong": [],
        "tests": "multi-part factual recall",
    },
]


def score(answer: str, probe: dict) -> tuple[str, str]:
    """Returns (verdict, why). Crude on purpose — read the answers themselves."""
    lowered = answer.lower()
    if any(w in lowered for w in probe["wrong"]):
        hit = next(w for w in probe["wrong"] if w in lowered)
        return "WRONG", f"contains {hit!r}"
    if any(r in lowered for r in probe["right"]):
        hit = next(r for r in probe["right"] if r in lowered)
        return "RIGHT", f"contains {hit!r}"
    return "UNCLEAR", "matched neither the expected nor the known-wrong phrasing"


async def run(members: list, judge, probes: list[dict], strategy: str) -> None:
    machinery_ok = True
    tally = {"RIGHT": 0, "WRONG": 0, "UNCLEAR": 0}

    for probe in probes:
        print(f"\n{DIM}{'─' * 74}{OFF}")
        print(f"Q: {probe['q'][:90]}")
        print(f"{DIM}   testing: {probe['tests']}{OFF}\n")

        messages = [{"role": "user", "content": probe["q"]}]
        started = time.time()
        if strategy == "race":
            winner, candidates = await council.race(
                members, messages, settings.council_timeout)
            verdict = council.Verdict(winner, "first usable", "race", candidates)
        else:
            candidates = await council.gather(
                members, messages, settings.council_timeout)
            verdict = await council.judge(judge, probe["q"], candidates)
        elapsed = time.time() - started

        # Every member's actual answer, so the score can be checked by eye.
        for c in verdict.candidates:
            if not c.usable:
                print(f"  {c.label} {c.provider:<34} {RED}FAILED{OFF}  {c.error}")
                continue
            mark, why = score(c.text, probe)
            colour = {"RIGHT": GREEN, "WRONG": RED, "UNCLEAR": YELLOW}[mark]
            won = " ←winner" if c.label == verdict.winner.label else ""
            print(f"  {c.label} {c.provider:<34} {colour}{mark:<7}{OFF} "
                  f"{c.elapsed:>5.1f}s{won}")
            print(f"     {DIM}{' '.join(c.text.split())[:150]}{OFF}")

        if verdict.winner.usable:
            mark, why = score(verdict.winner.text, probe)
            tally[mark] += 1
            colour = {"RIGHT": GREEN, "WRONG": RED, "UNCLEAR": YELLOW}[mark]
            print(f"\n  verdict: {colour}{mark}{OFF} via {verdict.method} "
                  f"in {elapsed:.1f}s — {verdict.reason[:80]}")
        else:
            machinery_ok = False
            print(f"\n  {RED}no member answered{OFF}")

    print(f"\n{DIM}{'═' * 74}{OFF}")
    total = sum(tally.values())
    print(f"  {GREEN}{tally['RIGHT']} right{OFF} · {RED}{tally['WRONG']} wrong{OFF} "
          f"· {YELLOW}{tally['UNCLEAR']} unclear{OFF}   (of {total})")
    print(f"  {DIM}Scoring is keyword-based and crude. Read the answers above —{OFF}")
    print(f"  {DIM}an answer can name the right person and still get the rest wrong.{OFF}")
    if not machinery_ok:
        print(f"  {RED}Machinery problem: at least one question had no usable answer.{OFF}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", default=settings.council_members)
    parser.add_argument("--judge", default=settings.council_judge)
    parser.add_argument("--strategy", default="council",
                        choices=["council", "race"])
    parser.add_argument("--only", type=int, default=0,
                        help="run only the first N questions")
    args = parser.parse_args()

    print(f"Council · real models · strategy={args.strategy}")
    print(f"{DIM}members: {args.members}{OFF}")

    members = council.build_members(args.members)
    if not members:
        print(f"{RED}No members could be built from {args.members!r}.{OFF}")
        return 1

    live, dropped = await council.healthy_members(members)
    for note in dropped:
        print(f"  {YELLOW}skipped{OFF} {note}")
    if not live:
        print(f"\n{RED}No members are reachable.{OFF}")
        print("Set GROQ_API_KEY (free, no card: console.groq.com) or "
              "GEMINI_API_KEY,\nor start Ollama, then run this again.")
        return 1
    print(f"  live: {', '.join(describe(m) for m in live)}")

    judge = None
    if args.judge:
        try:
            judge = from_spec(args.judge)
            ok, detail = await judge.health()
            if not ok:
                print(f"  {YELLOW}judge unusable ({detail}) — "
                      f"falling back to the heuristic{OFF}")
                judge = None
        except ProviderError as e:
            print(f"  {YELLOW}judge unusable ({e}){OFF}")
    if judge is None:
        print(f"  {DIM}judge: heuristic{OFF}")

    probes = PROBES[:args.only] if args.only else PROBES
    await run(live, judge, probes, args.strategy)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
