"""
Don't stall — checked, not merely requested.

Three ways an assistant fails: it says something false, it gives a confident
recommendation on weak evidence, or it refuses to help because it cannot
guarantee perfection. The industry guards heavily against the first and treats
the third as a virtue, which is how you get an assistant that answers

    "I cannot confidently determine this."

and leaves the user exactly where they started. That is not caution. It is a
failure with better manners.

The identity prompt asks for the alternative — "I don't know yet, here is how
we find out" — and asking is not enough, for the same reason that asking a
model to count its own letters is not enough. A model under uncertainty
reliably reaches for the hedge and stops, because stopping is never *wrong*.
So the stop is detected and repaired through the same loop that repairs a
broken constraint: name the specific defect, ask for the specific fix.

What counts as a stall
----------------------
The answer signals it cannot settle the question — and then offers no way
forward. Both halves are required. "I'm not certain whether Postgres or MySQL
suits you better; I'd take Postgres for the JSON support, and you can settle it
in an afternoon by loading your real query mix into both" is not a stall. It is
exactly the target behaviour: honest about the edge, and still moving.

What is deliberately exempt
---------------------------
A safety refusal. If the model has declined to help with something harmful,
"add a next step" is the wrong instruction to hand it, and a repair loop that
pushes against a refusal is a loop that eventually gets one overturned. Those
answers are left alone, and the identity prompt carries the softer version of
the rule for them: offer the nearest legitimate alternative.

An empty answer is also exempt, because a different check owns that failure and
two checks reporting one fault helps nobody.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The answer is signalling that it cannot settle the question. None of these is
# a problem on its own — each is honest, and honesty is the floor here.
_STALL = re.compile(
    r"\bi (?:don'?t|do not) know\b|"
    r"\bi'?m not (?:sure|certain)\b|"
    # The adverb slot is not optional decoration — "I cannot CONFIDENTLY
    # determine this" is the exact sentence this whole module exists to catch,
    # and a pattern without the slot sails straight past it.
    r"\bi (?:can(?:no|')t|cannot|am unable to|couldn'?t) (?:\w+ly )?(?:say|tell|"
    r"determine|confirm|verify|establish|answer|know|be sure|assess|"
    r"provide (?:an|a definitive))\b|"
    r"\b(?:it'?s|it is) (?:impossible|not possible) to (?:say|tell|know|determine)\b|"
    r"\b(?:there is|there'?s) no way to (?:know|tell|say|determine)\b|"
    r"\b(?:insufficient|not enough) (?:information|data|context|detail)\b|"
    r"\bwithout more (?:information|detail|context)\b|"
    r"\bunable to (?:determine|verify|confirm|establish|answer)\b|"
    r"\b(?:this|that|it) (?:would )?depends?\b|"
    r"\bhard to say\b|\bunclear\b|\bcannot be (?:determined|verified|answered)\b",
    re.IGNORECASE)

# A way forward. Broad on purpose: an answer that genuinely keeps moving will
# almost always contain one of these, and the cost of missing a stall is zero
# while the cost of nagging a good answer is a whole regeneration.
_FORWARD = re.compile(
    r"\bnext (?:step|move)\b|\bstart (?:by|with)\b|\bbegin (?:by|with)\b|"
    r"\bi'?d (?:choose|pick|go|take|start|recommend|suggest|use|lean|say|assume|bet)\b|"
    r"\bi would (?:choose|pick|go|take|start|recommend|suggest|use|lean|assume)\b|"
    r"\b(?:my|the) (?:best guess|recommendation|suggestion|call|bet)\b|"
    r"\bif (?:i had to|you had to|forced to)\b|"
    r"\brecommend\w*\b|\bsuggest\w*\b|"
    r"\bto (?:find|check|confirm|verify|settle|resolve|narrow|decide)\b|"
    r"\byou (?:can|could|should|might want to|need to|'ll want to)\b|"
    r"\b(?:try|check|run|search|test|measure|ask|look|read|compare|confirm|"
    r"verify|start|send|open|inspect|profile|benchmark)\b|"
    r"\bin the meantime\b|\bfor now,? (?:assume|use|go|treat)\b|"
    r"\bhere'?s how\b|\bhere is how\b|\bwhat (?:would|will) settle\b|"
    r"\bdepends on\b|\bdepending on\b",
    re.IGNORECASE)

# A refusal on safety grounds, which this check must not touch.
_REFUSAL = re.compile(
    r"\bi (?:can'?t|cannot|won'?t|will not) help (?:with|you)\b|"
    r"\bi (?:can'?t|cannot|won'?t|will not) (?:assist|provide|write|create|"
    r"produce|generate)\b|"
    r"\b(?:against|violates) (?:my |our )?(?:guidelines|policy|policies)\b|"
    r"\bnot something i can help\b|\bi have to decline\b|\bi must decline\b",
    re.IGNORECASE)

REPAIR = """\
Your answer stopped at uncertainty and left the user where they started. In \
this product that is a failure, not caution — as costly as being wrong.

Keep every bit of the honesty. Do not manufacture confidence you do not have. \
Add the way forward:

- If you would lean one way, say which and why, and mark what you are unsure of.
- Say what would settle it: the specific thing to check, search, measure or try.
- If it genuinely depends on something, say what it depends on and which way \
you would go under each.
- If nothing can resolve it right now, say what you would assume in the \
meantime and why that assumption is safe.

Output only the rewritten answer. Do not mention this note."""


@dataclass
class Stall:
    """Where the answer stopped, and what it was missing."""

    quote: str          # the sentence that stalled, for the repair message
    detail: str


def check(response: str) -> list[Stall]:
    """Empty when the answer keeps the user moving. At most one stall."""
    text = (response or "").strip()
    if not text:
        return []                       # the empty-answer check owns this
    if _REFUSAL.search(text):
        return []                       # never argue with a refusal

    hit = _STALL.search(text)
    if not hit:
        return []
    if _FORWARD.search(text):
        return []                       # hedged, but still moving

    # Quote the sentence it stopped on, so the repair is about something the
    # model can see rather than a general complaint about its attitude.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    quote = next((s.strip() for s in sentences if _STALL.search(s)), text[:120])
    return [Stall(quote=quote[:160],
                  detail="stopped at uncertainty without a next step")]


def repair_prompt(stalls: list[Stall]) -> str:
    if not stalls:
        return REPAIR
    return f'You wrote: "{stalls[0].quote}"\n\n{REPAIR}'
