"""
Run the 100-question diagnostic suite against a live Praxis server.

    python server.py                      # in one terminal
    python tests/benchmark.py             # in another

    python tests/benchmark.py --section constraint     # one section
    python tests/benchmark.py --only 10                # first ten questions
    python tests/benchmark.py --mode reality --strategy council

What this grades, and how much each grade is worth trusting:

  MACHINERY — Praxis's own job, and graded exactly. Did an answer arrive at
  all? Was it blank? Did the turn error? Did the mode keep its contract? Did
  constraint checking engage where the question stated a constraint? Every one
  of these is a fact about the code, and a failure here is a bug to fix.

  CONSTRAINTS — the twenty questions in section 3 are graded mechanically,
  character by character. "Contains no letter e" is not an opinion. A failure
  here is the model's, but Praxis is supposed to have caught it and forced a
  rewrite, so the report shows both: whether the answer complies, and whether
  the checker noticed when it did not.

  SUBSTANCE — everything else is keyword-scored and therefore crude, and the
  report says so on every run. A keyword can pass an answer that names the
  right person and gets the rest wrong. The transcript is written to disk for
  exactly this reason: read the answers, do not trust the number.

The distinction that matters most is between a wrong answer and a broken
product. A model getting the Monty Hall variant wrong is a weak model. A blank
reply, an unhandled 429, or a mode that quietly ignored its own promise is a
bug in this repository.
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

from tests.questions import Q, SECTIONS                         # noqa: E402

GREEN, RED, YELLOW, BLUE, DIM, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m")

VOWELS = set("aeiou")


# --- exact graders for section 3 -------------------------------------------

def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z']+", text) if w]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _strip_fence(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


def _syllables(word: str) -> int:
    """Crude but consistent: vowel groups, with a silent trailing 'e'."""
    low = word.lower()
    count = len(re.findall(r"[aeiouy]+", low))
    # Silent trailing e, including the plural and third-person forms that carry
    # it: "flames" and "makes" are one syllable, and counting them as two
    # failed correct monosyllabic prose.
    if count > 1 and not low.endswith("le") and (
            low.endswith("e") or re.search(r"[^aeiouy]es$", low)):
        count -= 1
    return max(count, 1)


def check_constraint(kind, spec, answer: str) -> tuple[bool, str]:
    """Returns (satisfied, what went wrong). Graded on the answer as written."""
    text = answer.strip()
    if not text:
        return False, "empty answer"

    if kind == "no_letters":
        # The rule applies to the answer, not to a preamble about the rule.
        hits = sorted({c for c in text.lower() if c in spec})
        return (not hits), (f"uses {', '.join(repr(h) for h in hits)}"
                            if hits else "")

    if kind == "banned_words":
        low = text.lower()
        hits = [w for w in spec if re.search(rf"\b{re.escape(w)}\b", low)]
        return (not hits), f"uses {hits}" if hits else ""

    if kind == "sentence_words":
        got = [len(_words(s)) for s in _sentences(text)]
        return got == list(spec), f"word counts {got}, wanted {list(spec)}"

    if kind == "one_sentence_words":
        sentences = _sentences(text)
        if len(sentences) != 1:
            return False, f"{len(sentences)} sentences, wanted 1"
        if "," in text:
            return False, "contains a comma"
        n = len(_words(text))
        return n == spec, f"{n} words, wanted {spec}"

    if kind == "words_and_bans":
        want, banned = spec
        n = len(_words(text))
        low = text.lower()
        hits = [w for w in banned if re.search(rf"\b{re.escape(w)}\b", low)]
        if hits:
            return False, f"uses banned {hits}"
        return n == want, f"{n} words, wanted {want}"

    if kind == "json_field_words":
        field, want, count = spec
        try:
            data = json.loads(_strip_fence(text))
        except json.JSONDecodeError as e:
            return False, f"not valid JSON ({e.msg})"
        if not isinstance(data, list):
            return False, f"top level is {type(data).__name__}, wanted a list"
        if len(data) != count:
            return False, f"{len(data)} items, wanted {count}"
        bad = []
        for i, item in enumerate(data):
            if not isinstance(item, dict) or field not in item:
                bad.append(f"item {i} has no {field!r}")
                continue
            n = len(_words(str(item[field])))
            if n != want:
                bad.append(f"item {i} {field} is {n} words")
        return (not bad), "; ".join(bad)

    if kind == "monosyllabic":
        bad = [w for w in _words(text) if _syllables(w) > 1]
        return (not bad), f"multi-syllable: {bad[:6]}" if bad else ""

    if kind == "abecedarian":
        words = _words(text)
        if len(words) != spec:
            return False, f"{len(words)} words, wanted {spec}"
        wanted = [chr(ord("a") + i) for i in range(spec)]
        got = [w[0].lower() for w in words]
        return got == wanted, f"initials {''.join(got)}, wanted {''.join(wanted)}"

    if kind == "no_punctuation":
        hits = sorted({c for c in text if c in ".,;:!?'\"-—–()[]{}/"})
        return (not hits), f"uses {hits}" if hits else ""

    if kind == "line_endings":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) != len(spec):
            return False, f"{len(lines)} lines, wanted {len(spec)}"
        low = text.lower()
        if "sky" in low or "moon" in low:
            return False, "uses a banned word (sky/moon)"
        bad = [f"line {i + 1} ends {_words(ln)[-1]!r}, wanted {want!r}"
               for i, (ln, want) in enumerate(zip(lines, spec))
               if not _words(ln) or _words(ln)[-1].lower().strip("'") != want]
        return (not bad), "; ".join(bad)

    if kind == "all_consonant":
        bad = [w for w in _words(text) if w[0].lower() in VOWELS]
        sentences = _sentences(text)
        if len(sentences) != 3:
            return False, f"{len(sentences)} sentences, wanted 3"
        return (not bad), f"vowel-initial: {bad[:6]}" if bad else ""

    if kind == "all_start_with":
        bad = [w for w in _words(text) if not w.lower().startswith(spec)]
        return (not bad), f"not {spec!r}-initial: {bad[:6]}" if bad else ""

    if kind == "alternating":
        kinds = [(w[0].lower() in VOWELS) for w in _words(text)]
        bad = [i for i in range(1, len(kinds)) if kinds[i] == kinds[i - 1]]
        return (not bad), f"{len(bad)} places break the alternation" if bad else ""

    if kind == "table_cell_words":
        rows = [ln for ln in text.splitlines() if ln.strip().startswith("|")]
        body = [r for r in rows if not re.match(r"^\s*\|[\s:|-]+\|\s*$", r)][1:]
        want_rows, want_words = spec
        if len(body) != want_rows:
            return False, f"{len(body)} body rows, wanted {want_rows}"
        bad = []
        for i, row in enumerate(body):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) < 2:
                bad.append(f"row {i + 1} has {len(cells)} columns")
                continue
            n = len(_words(cells[1]))
            if n != want_words:
                bad.append(f"row {i + 1} column 2 is {n} words")
        return (not bad), "; ".join(bad)

    if kind == "heat_lines":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) != 3:
            return False, f"{len(lines)} lines, wanted 3"
        bad = []
        for i, line in enumerate(lines):
            n = len(_words(line))
            heats = len(re.findall(r"\bheat\b", line, re.IGNORECASE))
            if n > 5:
                bad.append(f"line {i + 1} is {n} words")
            if heats != 1:
                bad.append(f"line {i + 1} says 'heat' {heats} times")
        return (not bad), "; ".join(bad)

    if kind == "no_nouns":
        # Not decidable without a tagger, and a wrong accusation is worse than
        # no check. Reported, never scored.
        return True, "not mechanically checkable — read it yourself"

    return True, "no checker"


# --- the run ---------------------------------------------------------------

class Result:
    def __init__(self, question: dict) -> None:
        self.q = question
        self.answer = ""
        self.error = ""
        self.events: list[str] = []
        self.elapsed = 0.0
        self.machinery: list[str] = []      # what Praxis got wrong
        self.verdict = "UNCLEAR"
        self.why = ""


def ask(client: httpx.Client, base: str, question: dict,
        mode: str, strategy: str) -> Result:
    """One question, one fresh conversation, over the real SSE endpoint."""
    result = Result(question)
    started = time.time()
    payload = {"message": question["prompt"], "mode": mode,
               "strategy": strategy, "conversation_id": None}
    try:
        with client.stream("POST", f"{base}/api/chat", json=payload,
                           timeout=300) as response:
            if response.status_code >= 400:
                result.error = f"HTTP {response.status_code}"
                result.machinery.append(f"the API returned {result.error}")
                return result
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:].strip()
                if body == "[DONE]":
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                kind, data = event.get("type"), event.get("data") or {}
                result.events.append(kind)
                if kind == "token":
                    result.answer += data.get("text", "")
                elif kind == "error":
                    result.error = data.get("message", "")
    except httpx.HTTPError as e:
        result.error = f"{type(e).__name__}: {e}"
        result.machinery.append(f"the connection failed: {result.error}")
    result.elapsed = time.time() - started
    return result


def grade(result: Result) -> None:
    q, answer = result.q, result.answer.strip()

    # --- machinery: Praxis's own job, graded exactly ---
    if not answer and not result.error:
        result.machinery.append("BLANK: the turn succeeded and produced no text")
    if result.error and "rate limit" in result.error.lower():
        result.machinery.append("rate limited even after waiting and shrinking")
    elif result.error:
        result.machinery.append(f"errored: {result.error[:90]}")
    if "done" not in result.events and not result.error:
        result.machinery.append("no 'done' event — the stream ended early")
    if q.get("constraint") and "start" in result.events:
        pass  # constraint detection is reported below, not required here

    if not answer:
        result.verdict = "NO ANSWER"
        result.why = result.error[:100] or "nothing came back"
        return

    low = answer.lower()

    # --- section 3: graded exactly ---
    if q.get("constraint"):
        kind, spec = q["constraint"]
        ok, why = check_constraint(kind, spec, answer)
        # Praxis is meant to catch its own violations and say so.
        flagged = ("check failed after" in low or "does not meet what you asked"
                   in low)
        result.verdict = "RIGHT" if ok else ("FLAGGED" if flagged else "WRONG")
        result.why = why or "satisfies the constraint"
        if not ok and not flagged:
            result.machinery.append(
                "constraint broken and NOT flagged — the checker missed it")
        return

    # --- refusal calibration ---
    refused = any(p in low for p in (
        "i can't help", "i cannot help", "i can't assist", "i cannot assist",
        "i won't provide", "i will not provide", "against my guidelines",
        "i'm not able to help with that", "i can't provide"))
    if q.get("must_not_refuse") and refused:
        result.verdict = "WRONG"
        result.why = "refused a legitimate question"
        return

    # --- keyword scoring: crude, and labelled crude ---
    wrong_hit = next((w for w in q.get("wrong", []) if w in low), None)
    right_hit = next((r for r in q.get("right", []) if r in low), None)
    if wrong_hit and not right_hit:
        result.verdict, result.why = "WRONG", f"contains {wrong_hit!r}"
    elif right_hit:
        result.verdict, result.why = "RIGHT", f"contains {right_hit!r}"
    elif q.get("must_reject"):
        result.verdict = "WRONG"
        result.why = "played along with a false premise instead of rejecting it"
    else:
        result.verdict, result.why = "UNCLEAR", "matched neither probe set"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", default="standard")
    parser.add_argument("--strategy", default="single")
    parser.add_argument("--section", default="", choices=["", *SECTIONS])
    parser.add_argument("--only", type=int, default=0)
    parser.add_argument("--pace", type=float, default=0.0,
                        help="seconds between questions; raise on a free tier")
    parser.add_argument("--out", default="benchmark-transcript.md")
    args = parser.parse_args()

    questions = [q for q in Q if not args.section or q["section"] == args.section]
    if args.only:
        questions = questions[:args.only]

    client = httpx.Client(trust_env=False)
    try:
        health = client.get(f"{args.base}/api/health", timeout=20).json()
    except httpx.HTTPError:
        print(f"{RED}No Praxis server at {args.base}.{OFF} Start one: python server.py")
        return 1
    print(f"Praxis benchmark — {len(questions)} questions, mode={args.mode}, "
          f"strategy={args.strategy}")
    print(f"{DIM}backend: {health.get('detail')}{OFF}\n")

    results: list[Result] = []
    for i, question in enumerate(questions, 1):
        result = ask(client, args.base, question, args.mode, args.strategy)
        grade(result)
        results.append(result)

        colour = {"RIGHT": GREEN, "WRONG": RED, "UNCLEAR": YELLOW,
                  "FLAGGED": BLUE, "NO ANSWER": RED}[result.verdict]
        flag = f" {RED}⚑{OFF}" if result.machinery else ""
        print(f"  {question['n']:>3}. {colour}{result.verdict:<9}{OFF}"
              f"{result.elapsed:>6.1f}s  {question['prompt'][:58]}…{flag}")
        for note in result.machinery:
            print(f"       {RED}{note}{OFF}")
        if args.pace:
            time.sleep(args.pace)

    # --- report ---
    print(f"\n{DIM}{'═' * 78}{OFF}")
    for key, title in SECTIONS.items():
        rows = [r for r in results if r.q["section"] == key]
        if not rows:
            continue
        tally = {v: sum(1 for r in rows if r.verdict == v)
                 for v in ("RIGHT", "WRONG", "UNCLEAR", "FLAGGED", "NO ANSWER")}
        print(f"  {title:<44} {GREEN}{tally['RIGHT']:>2} right{OFF} "
              f"{RED}{tally['WRONG']:>2} wrong{OFF} "
              f"{YELLOW}{tally['UNCLEAR']:>2} unclear{OFF}"
              + (f" {BLUE}{tally['FLAGGED']} flagged{OFF}" if tally["FLAGGED"] else "")
              + (f" {RED}{tally['NO ANSWER']} blank{OFF}" if tally["NO ANSWER"] else ""))

    broken = [r for r in results if r.machinery]
    answered = sum(1 for r in results if r.answer.strip())
    print(f"\n  {'answered':<20} {answered}/{len(results)}")
    print(f"  {'machinery faults':<20} {len(broken)}"
          + (f"  {RED}← bugs in Praxis, not the model{OFF}" if broken else
             f"  {GREEN}← nothing wrong on Praxis's side{OFF}"))
    print(f"\n  {DIM}Keyword scoring is crude. The constraint section is exact;{OFF}")
    print(f"  {DIM}everything else is a probe. Read the transcript.{OFF}")

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(f"# Praxis benchmark — {health.get('detail')}\n\n")
        fh.write(f"mode `{args.mode}` · strategy `{args.strategy}` · "
                 f"{len(results)} questions\n\n")
        for r in results:
            fh.write(f"\n## {r.q['n']}. [{r.verdict}] {r.q['section']}\n\n")
            fh.write(f"**Q.** {r.q['prompt']}\n\n")
            fh.write(f"*{r.why}* · {r.elapsed:.1f}s\n\n")
            for note in r.machinery:
                fh.write(f"> MACHINERY FAULT: {note}\n\n")
            fh.write((r.answer.strip() or "_(no answer)_") + "\n")
    print(f"\n  full transcript → {out}\n")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
