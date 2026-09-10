"""
Mode contracts — holding a mode to the promise it made.

A mode is a promise. Direct mode promises under 150 words with the conclusion
first. Executive Brief promises five specific headings and a recommendation it
commits to. Reality Check promises a verdict from a fixed set. Socratic mode
promises *not* to hand over the answer.

Prompt text alone does not keep those promises. A model reads "under 150 words"
as a suggestion and writes 400. The mode then silently becomes decoration: the
label says Direct, the answer reads like Standard, and the user has no way to
tell the difference except by counting.

So each promise that can be checked mechanically, is. A broken contract feeds
the same repair loop as a broken constraint — the model is told exactly which
promise it broke, and rewrites.

Two rules govern what belongs here, because a false accusation is worse than a
missed one:

  1. Only promises the mode's own prompt actually states. Not style opinions.
  2. Only checks that cannot reasonably misfire on a good answer. "Did it end
     with a question?" is checkable. "Was it insightful?" is not.

Modes with nothing safely checkable have no contract, and that is fine.
Standard deliberately has none — it promises nothing beyond the core
identity, so there is nothing to hold it to. The other nine carry one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# Word counts are checked with a tolerance. A mode promising "under 150 words"
# is making a stylistic commitment, not a specification — failing an answer at
# 151 words would be pedantry that costs a whole regeneration.
WORD_TOLERANCE = 1.25


@dataclass
class Promise:
    """One checkable commitment a mode makes."""

    label: str                                   # shown to the user
    check: Callable[[str], bool]                 # True when kept
    repair: str                                  # told to the model when broken


@dataclass
class Breach:
    mode: str
    promise: Promise
    detail: str


def _words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _strip_markup(text: str) -> str:
    """Headings and emphasis are formatting, not content, for counting purposes."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"^[#>\-*\d.]+\s*", "", text, flags=re.MULTILINE)
    return re.sub(r"[*_`]", "", text)


def _has_heading(text: str, word: str) -> bool:
    """A heading in any of the shapes a model actually produces.

    Two accepted forms, because models vary and a false accusation costs a
    whole regeneration: the word at the start of a line (with or without #,
    ** or __), or the word emphasised anywhere — "**Situation** Sales fell."
    is a legitimate brief, just tersely formatted."""
    at_line_start = rf"(^|\n)\s*(#{{1,4}}\s*|\*\*\s*|__\s*)?{re.escape(word)}\b\s*[:*_]*"
    emphasised = rf"(\*\*|__)\s*{re.escape(word)}\b[^*_\n]{{0,12}}(\*\*|__)"
    return (re.search(at_line_start, text, re.IGNORECASE) is not None
            or re.search(emphasised, text, re.IGNORECASE) is not None)


def _ends_with_question(text: str) -> bool:
    tail = text.strip()
    if not tail:
        return False
    # Ignore a trailing offer like "Shall I go on?" — look at real sentences.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", tail) if s.strip()]
    return bool(sentences) and sentences[-1].endswith("?")


# --- the contracts ---------------------------------------------------------

CONTRACTS: dict[str, list[Promise]] = {

    "direct": [
        Promise(
            label="under 150 words",
            check=lambda t: _words(_strip_markup(t)) <= int(150 * WORD_TOLERANCE),
            repair=("Direct mode promises under 150 words. Cut it to that. "
                    "Lead with the conclusion, delete every sentence that is "
                    "not load-bearing."),
        ),
        Promise(
            label="no throat-clearing opener",
            check=lambda t: not re.match(
                r"\s*(great question|good question|certainly|sure[,!]|"
                r"absolutely|i'?d be happy to|that'?s an? (interesting|great))",
                t, re.IGNORECASE),
            repair=("Direct mode forbids preamble. Delete the opening pleasantry "
                    "and start with the answer itself."),
        ),
    ],

    "brief": [
        Promise(
            label="all five headings present",
            check=lambda t: all(_has_heading(t, w) for w in
                                ("Situation", "Problem", "Options",
                                 "Recommendation", "Next action")),
            repair=("Executive Brief promises exactly these headings: Situation, "
                    "Problem, Options, Recommendation, Next action. Include every "
                    "one, in that order."),
        ),
        Promise(
            label="under 350 words",
            check=lambda t: _words(_strip_markup(t)) <= int(350 * WORD_TOLERANCE),
            repair=("Executive Brief promises under 350 words. Cut it. If it does "
                    "not fit, you have not decided what matters."),
        ),
    ],

    "reality": [
        Promise(
            label="states a verdict",
            check=lambda t: re.search(
                r"\b(PURSUE WITH CHANGES|PURSUE|TEST FIRST|DROP IT)\b",
                t, re.IGNORECASE) is not None,
            repair=("Reality Check promises a verdict. End with exactly one of: "
                    "PURSUE / PURSUE WITH CHANGES / TEST FIRST / DROP IT, then "
                    "one paragraph defending that call."),
        ),
    ],

    "socratic": [
        Promise(
            label="asks rather than tells",
            check=_ends_with_question,
            repair=("Socratic mode promises not to hand over the answer. End with "
                    "a single question aimed at the exact gap in their reasoning. "
                    "Do not state the conclusion."),
        ),
    ],

    "exam": [
        Promise(
            label="asks a question",
            check=lambda t: "?" in t,
            repair=("Exam mode promises to test the user. Ask one question and "
                    "wait for the answer."),
        ),
    ],

    "build": [
        Promise(
            label="ships something runnable, not advice about it",
            check=lambda t: ("```" in t or re.search(
                r"(^|\n)\s*(\$|>|#)?\s*(npm|npx|pip|pipx|python|node|go |cargo|"
                r"docker|git|make|uv|bun|yarn|pnpm|curl|mkdir|cd )\b", t,
                re.IGNORECASE) is not None),
            repair=("Build mode promises artifacts, not advice: real code, real "
                    "file layouts, and the exact commands to run. Add them. An "
                    "answer with no code and no commands has not built "
                    "anything."),
        ),
    ],

    "teacher": [
        Promise(
            label="checks understanding with a real question",
            check=lambda t: "?" in t,
            repair=("Teacher mode promises to check understanding at natural "
                    "breakpoints with a real question. Ask one."),
        ),
        Promise(
            label="never assumes what the user already knows",
            check=lambda t: re.search(
                r"as you (probably|may|might|likely) know|as you'?re aware|"
                r"obviously,|of course you know", t, re.IGNORECASE) is None,
            repair=("Teacher mode forbids 'as you probably know'. If they knew, "
                    "they would not be asking. Delete the phrase and explain "
                    "the thing."),
        ),
    ],

    "research": [
        Promise(
            label="ends with what is established, contested and unknown",
            check=lambda t: re.search(
                r"establish|contested|unknown|unverified|uncertain|disputed|"
                r"in summary|synthesis|what we know", t, re.IGNORECASE)
                is not None,
            repair=("Research mode promises a closing synthesis: what is "
                    "established, what is contested, what is unknown. Add it, "
                    "and mark anything you could not verify as unverified."),
        ),
    ],

    "founder": [
        Promise(
            label="ends with something testable this week",
            check=lambda t: re.search(
                r"(this week|next step|first step|next move|start with|"
                r"cheapest (test|experiment)|test it by|validate)",
                t, re.IGNORECASE) is not None,
            repair=("Founder mode promises the one thing to do this week. Add a "
                    "concrete, cheap experiment that could falsify the idea, and "
                    "say when."),
        ),
    ],
}


def promises_for(mode: str) -> list[Promise]:
    return CONTRACTS.get((mode or "").lower(), [])


def check(mode: str, response: str) -> list[Breach]:
    """Which promises this answer broke. Empty means the mode kept its word."""
    text = (response or "").strip()
    if not text:
        return []
    breaches = []
    for promise in promises_for(mode):
        try:
            kept = promise.check(text)
        except Exception:
            # A broken check must never break the turn.
            continue
        if not kept:
            detail = promise.label
            if "words" in promise.label:
                detail = f"{promise.label} — it is {_words(_strip_markup(text))}"
            breaches.append(Breach(mode, promise, detail))
    return breaches


def repair_prompt(mode: str, breaches: list[Breach]) -> str:
    """What the model is told when it broke its own mode's promise."""
    lines = [
        f"Your answer did not honour {mode.upper()} mode, which the user "
        f"selected deliberately. Specifically:",
        "",
    ]
    for b in breaches:
        lines.append(f"- Broke: {b.detail}")
        lines.append(f"  {b.promise.repair}")
    lines += [
        "",
        "Rewrite the answer so it honours the mode. Keep the substance; change "
        "the shape. Output only the rewritten answer — do not mention this note.",
    ]
    return "\n".join(lines)


def describe(mode: str) -> list[str]:
    """The checkable promises, for display in the interface."""
    return [p.label for p in promises_for(mode)]
