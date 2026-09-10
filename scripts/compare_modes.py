"""
Ask one question in every mode, side by side, and show what actually changed.

    python server.py                                  # in one terminal
    python scripts/compare_modes.py "Should I rewrite our billing service?"

    python scripts/compare_modes.py --modes direct,brief,reality "..."
    python scripts/compare_modes.py --pace 8 "..."    # slower, for a free tier

A mode is a promise, and a promise you cannot see kept is indistinguishable
from decoration. This runs the same question through each mode, prints what
each one promised and whether it kept it, and writes every answer to a file so
the difference can be read rather than taken on trust.

What to look for, and what it means:

  LENGTH      Direct should be a fraction of Research. If they come back the
              same size, the mode overlay is not reaching the model — check
              /api/prompt?mode=direct to see the prompt that was actually sent.
  SHAPE       Brief must have five named headings. Build must contain code or
              commands. Socratic must end on a question and must NOT contain
              the answer.
  KEPT        Whether the mechanical contract passed. A "rewritten" note means
              the model broke its own mode's promise and was made to try again;
              a "not kept" means it broke it three times and Praxis said so
              rather than passing the answer off as compliant.

If every mode answers identically, that is a real finding and worth reporting.
It usually means one of: the model is too small to follow a system prompt, the
prompt budget trimmed the overlay away, or ENABLE_MODE_CONTRACTS is off.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx                                                    # noqa: E402

from praxis import contracts as contracts_mod                   # noqa: E402
from praxis import modes as modes_mod                           # noqa: E402

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m")


def ask(client: httpx.Client, base: str, question: str, mode: str) -> dict:
    out = {"answer": "", "error": "", "events": [], "elapsed": 0.0,
           "promises": [], "breaches": [], "attempts": 1}
    started = time.time()
    body = {"message": question, "mode": mode, "strategy": "single",
            "conversation_id": None}
    try:
        with client.stream("POST", f"{base}/api/chat", json=body,
                           timeout=300) as response:
            if response.status_code >= 400:
                out["error"] = f"HTTP {response.status_code}"
                return out
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                kind, data = event.get("type"), event.get("data") or {}
                out["events"].append(kind)
                if kind == "token":
                    out["answer"] += data.get("text", "")
                elif kind == "start":
                    out["promises"] = data.get("promises", [])
                elif kind == "mode_breach":
                    out["breaches"].append(data.get("broke", []))
                elif kind == "error":
                    out["error"] = data.get("message", "")
                elif kind == "done":
                    out["attempts"] = data.get("constraint_attempts", 1)
    except httpx.HTTPError as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["elapsed"] = time.time() - started
    return out


def shape(answer: str) -> str:
    """A one-line fingerprint, so two modes can be told apart at a glance."""
    words = len(re.findall(r"\S+", answer))
    bits = [f"{words}w"]
    headings = len(re.findall(r"(^|\n)\s*(#{1,4}\s|\*\*[A-Z])", answer))
    if headings:
        bits.append(f"{headings} headings")
    if "```" in answer:
        bits.append(f"{answer.count('```') // 2} code")
    bullets = len(re.findall(r"(^|\n)\s*[-*+]\s", answer))
    if bullets:
        bits.append(f"{bullets} bullets")
    if answer.rstrip().endswith("?"):
        bits.append("ends on a question")
    return " · ".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--modes", default="")
    parser.add_argument("--pace", type=float, default=0.0,
                        help="seconds between modes; raise on a free tier")
    parser.add_argument("--out", default="mode-comparison.md")
    args = parser.parse_args()

    question = " ".join(args.question)
    keys = ([k.strip() for k in args.modes.split(",") if k.strip()]
            or list(modes_mod.MODES))

    client = httpx.Client(trust_env=False)
    try:
        health = client.get(f"{args.base}/api/health", timeout=20).json()
    except httpx.HTTPError:
        print(f"{RED}No Praxis server at {args.base}.{OFF} "
              f"Start one first: python server.py")
        return 1

    print(f"{BOLD}One question, {len(keys)} modes{OFF}")
    print(f"{DIM}backend: {health.get('detail')}{OFF}")
    print(f"{DIM}question: {question}{OFF}\n")

    results = {}
    for key in keys:
        mode = modes_mod.MODES.get(key)
        if mode is None:
            print(f"  {RED}no such mode: {key}{OFF}")
            continue
        result = ask(client, args.base, question, key)
        results[key] = result

        promised = contracts_mod.describe(key)
        if result["error"]:
            verdict = f"{RED}{result['error'][:60]}{OFF}"
        elif not result["answer"].strip():
            verdict = f"{RED}BLANK — no text came back{OFF}"
        elif "does not meet what you asked for" in result["answer"]:
            verdict = f"{RED}promise not kept, and said so{OFF}"
        elif result["breaches"]:
            verdict = (f"{YELLOW}broke its promise once, rewritten "
                       f"({result['attempts']} attempts){OFF}")
        elif promised:
            verdict = f"{GREEN}kept: {'; '.join(promised)}{OFF}"
        else:
            verdict = f"{DIM}no mechanical promise to keep{OFF}"

        print(f"  {mode.icon} {BOLD}{mode.label:<10}{OFF} "
              f"{result['elapsed']:>5.1f}s  {shape(result['answer']):<44}")
        print(f"       {verdict}")
        if args.pace:
            time.sleep(args.pace)

    # Are the answers actually different from each other?
    texts = {k: " ".join(r["answer"].split()).lower()
             for k, r in results.items() if r["answer"].strip()}
    if len(texts) > 1:
        identical = [(a, b) for i, a in enumerate(texts)
                     for b in list(texts)[i + 1:] if texts[a] == texts[b]]
        lengths = sorted(len(t) for t in texts.values())
        spread = lengths[-1] / max(lengths[0], 1)
        print(f"\n  {'length spread':<20} {spread:.1f}x "
              f"({lengths[0]} to {lengths[-1]} characters)")
        if identical:
            print(f"  {RED}identical answers:{OFF} "
                  + ", ".join(f"{a}={b}" for a, b in identical[:5]))
            print(f"  {DIM}Modes that answer identically are decoration. See the"
                  f" module docstring for what causes it.{OFF}")
        elif spread < 1.5:
            print(f"  {YELLOW}Every mode answered at almost the same length.{OFF}"
                  f" {DIM}Check the shapes above — if they also look alike, the "
                  f"mode overlay is not reaching the model.{OFF}")
        else:
            print(f"  {GREEN}Every mode answered differently.{OFF}")

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"# One question, {len(results)} modes\n\n")
        fh.write(f"**Q.** {question}\n\n")
        fh.write(f"`{health.get('detail')}`\n")
        for key, result in results.items():
            mode = modes_mod.MODES[key]
            fh.write(f"\n---\n\n## {mode.icon} {mode.label}\n\n")
            fh.write(f"*{mode.blurb}*\n\n")
            promised = contracts_mod.describe(key)
            fh.write(f"Promises checked: "
                     f"{'; '.join(promised) if promised else 'none'}  \n")
            fh.write(f"{shape(result['answer'])} · {result['elapsed']:.1f}s · "
                     f"{result['attempts']} attempt(s)\n\n")
            fh.write((result["answer"].strip()
                      or f"_(no answer — {result['error'] or 'blank'})_") + "\n")
    print(f"\n  side by side → {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
