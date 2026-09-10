"""
Mechanical constraint checking.

The failure this exists to fix: asked for four sentences with no letter E or S,
a model produced 15 Es, 16 Ss, and invented words like "trnspnds" trying. Asked
for descriptions of exactly eight words, it produced seven, seven and six.

No amount of prompt text fixes that. Generation is token-level — a token is
commonly several characters chosen by a model with no per-letter counting step —
so "be careful to obey the constraint" is an instruction the model cannot
actually execute. It will comply approximately and report full compliance,
which is the worst of both.

Code can check it in microseconds. So: detect the constraint in the request,
check the answer against it, and hand the model its specific violations to fix.
That converts an impossible task into an easy one — the model no longer has to
count, it only has to repair a named defect.

Detection is deliberately conservative. A missed constraint costs nothing (the
answer is no worse than before), a falsely-detected one wastes a generation and
confuses the model. So each pattern requires explicit constraint language.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

MAX_LETTER_EXAMPLES = 6


@dataclass
class Constraint:
    kind: str                    # forbidden_letters | word_count | sentence_count | json
    spec: dict
    description: str             # human-readable, shown to the model on retry


@dataclass
class Violation:
    constraint: Constraint
    detail: str                  # exactly what went wrong, specifically


# --- detection -------------------------------------------------------------

# "do NOT use the letters 'e' or 's'" / "without using the letter e" /
# "avoid the letter E" / "no letter e anywhere"
_LETTER_RE = re.compile(
    r"(?:do\s+not|don't|never|without|avoid|exclude|no)\s+"
    r"(?:use\s+|using\s+|contain\s+|containing\s+)?"
    r"(?:the\s+)?letters?\s+"
    r"(?P<body>[^.\n;]{1,80})",
    re.IGNORECASE,
)

# "exactly 8 words" / "in exactly 40 words" / "at most 20 words" /
# "no more than 20 words" / "20 words or fewer"
_WORD_EXACT_RE = re.compile(r"exactly\s+(?P<n>\d{1,4})\s+words?", re.IGNORECASE)
_WORD_MAX_RE = re.compile(
    r"(?:at\s+most|no\s+more\s+than|maximum\s+of|under|fewer\s+than|less\s+than)\s+"
    r"(?P<n>\d{1,4})\s+words?|(?P<n2>\d{1,4})\s+words?\s+or\s+(?:fewer|less)",
    re.IGNORECASE,
)

# "a 4-sentence summary" / "exactly 3 sentences" / "in 5 sentences"
_SENTENCE_RE = re.compile(
    r"(?:exactly\s+)?(?P<n>\d{1,3})[-\s]sentences?\b|"
    r"in\s+(?P<n2>\d{1,3})\s+sentences?\b",
    re.IGNORECASE,
)

_JSON_RE = re.compile(
    r"(?:valid\s+json|json\s+(?:object|array)\b|"
    r"(?:format|output|return|respond|reply|give|provide|produce)"
    r"[^.\n]{0,40}?\bjson\b)(?P<shape>[^.\n]{0,40})",
    re.IGNORECASE,
)

# Marks a count as applying per item rather than to the whole answer, and
# captures WHICH field it applies to: "limit every DESCRIPTION to exactly 8
# words" constrains description, not every string in the document.
_PER_ITEM_RE = re.compile(r"\b(each|every|per)\b", re.IGNORECASE)
_PER_FIELD_RE = re.compile(
    r"\b(?:each|every|per)\s+(?P<field>[a-z_][a-z0-9_]{2,30})\s+"
    r"(?:to|must\s+be|should\s+be|is)?\s*"
    r"(?:exactly|at\s+most|no\s+more\s+than|under)?\s*\d{1,4}\s+words?",
    re.IGNORECASE,
)


def _letters_from(body: str) -> list[str]:
    """Pull the actual letters out of 'e' or 's' / E and S / "e", "s"."""
    quoted = re.findall(r"['\"‘’“”]\s*([a-z])\s*['\"‘’“”]", body, re.IGNORECASE)
    if quoted:
        return sorted({c.lower() for c in quoted})
    # Unquoted: standalone single letters, e.g. "letters E or S anywhere"
    loose = re.findall(r"\b([a-z])\b", body, re.IGNORECASE)
    # Drop English words that are single letters and rarely the target.
    loose = [c for c in loose if c.lower() != "a"] or loose
    return sorted({c.lower() for c in loose})


def detect(request: str) -> list[Constraint]:
    """Find explicit, checkable constraints in a user's message."""
    found: list[Constraint] = []
    if not request:
        return found

    match = _LETTER_RE.search(request)
    if match:
        letters = _letters_from(match.group("body"))
        if letters:
            shown = ", ".join(f"'{c}'" for c in letters)
            found.append(Constraint(
                "forbidden_letters", {"letters": letters},
                f"must not contain the letter(s) {shown}"))

    per_item = bool(_PER_ITEM_RE.search(request))
    field_match = _PER_FIELD_RE.search(request)
    field = field_match.group("field").lower() if field_match else None

    match = _WORD_EXACT_RE.search(request)
    if match:
        n = int(match.group("n"))
        found.append(Constraint(
            "word_count", {"exact": n, "per_item": per_item, "field": field},
            f"{'each ' + field if field else 'each item' if per_item else 'the response'}"
            f" must be exactly {n} words"))
    else:
        match = _WORD_MAX_RE.search(request)
        if match:
            n = int(match.group("n") or match.group("n2"))
            found.append(Constraint(
                "word_count", {"max": n, "per_item": per_item, "field": field},
                f"{'each ' + field if field else 'each item' if per_item else 'the response'}"
                f" must be at most {n} words"))

    match = _SENTENCE_RE.search(request)
    if match:
        n = int(match.group("n") or match.group("n2"))
        if 1 <= n <= 100:
            found.append(Constraint(
                "sentence_count", {"exact": n},
                f"the response must be exactly {n} sentences"))

    match = _JSON_RE.search(request)
    if match:
        shape = match.group("shape").lower()
        want = "array" if "array" in shape or "list" in shape else (
            "object" if "object" in shape else "any")
        found.append(Constraint(
            "json", {"shape": want},
            "the response must be valid JSON"
            + (f" (a single {want})" if want != "any" else "")))

    return found


# --- verification ----------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return fenced[0].strip() if fenced else text.strip()


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _json_strings(value, field: str | None = None) -> list[str]:
    """String leaves from a parsed JSON structure.

    When `field` is given, only values under that key are returned — "every
    description must be 8 words" constrains description, not every string in
    the document. Without that scoping the checker flags correct fields."""
    out: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if field is not None:
                if k.lower() == field and isinstance(v, str):
                    out.append(v)
                else:
                    out.extend(_json_strings(v, field))
            else:
                out.extend(_json_strings(v, field))
    elif isinstance(value, list):
        for v in value:
            out.extend(_json_strings(v, field))
    elif isinstance(value, str) and field is None:
        out.append(value)
    return out


def verify(response: str, constraints: list[Constraint]) -> list[Violation]:
    """Check a response. Returns only real, specific violations."""
    violations: list[Violation] = []
    if not response.strip():
        return violations

    parsed = None
    for c in constraints:
        if c.kind == "json":
            body = _strip_code_fences(response)
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as e:
                violations.append(Violation(
                    c, f"the response is not valid JSON ({e.msg} at line {e.lineno})"))
                continue
            want = c.spec.get("shape", "any")
            actual = "array" if isinstance(parsed, list) else (
                "object" if isinstance(parsed, dict) else "scalar")
            if want != "any" and actual != want:
                violations.append(Violation(
                    c, f"the JSON is a {actual}, but a single {want} was asked for"))

    for c in constraints:
        if c.kind == "forbidden_letters":
            hits = {}
            for letter in c.spec["letters"]:
                count = response.lower().count(letter)
                if count:
                    hits[letter] = count
            if hits:
                examples = []
                for letter in hits:
                    for word in re.findall(r"[A-Za-z']+", response):
                        if letter in word.lower():
                            examples.append(word)
                            break
                summary = ", ".join(f"'{k}' appears {v}x" for k, v in hits.items())
                sample = ", ".join(sorted(set(examples))[:MAX_LETTER_EXAMPLES])
                violations.append(Violation(
                    c, f"{summary}. Offending words include: {sample}"))

        elif c.kind == "word_count":
            targets: list[tuple[str, str]] = []
            if c.spec.get("per_item") and parsed is not None:
                field = c.spec.get("field")
                label = field or "item"
                targets = [(f"{label} {i + 1}", s) for i, s in
                           enumerate(_json_strings(parsed, field))]
            elif not c.spec.get("per_item"):
                targets = [("the response", _strip_code_fences(response))]

            bad = []
            for label, text in targets:
                n = len(_words(text))
                if "exact" in c.spec and n != c.spec["exact"]:
                    bad.append(f"{label} has {n} (needs {c.spec['exact']}): \"{text}\"")
                elif "max" in c.spec and n > c.spec["max"]:
                    bad.append(f"{label} has {n} (max {c.spec['max']})")
            if bad:
                violations.append(Violation(c, "; ".join(bad[:8])))

        elif c.kind == "sentence_count":
            n = len(_sentences(_strip_code_fences(response)))
            if n != c.spec["exact"]:
                violations.append(Violation(
                    c, f"the response has {n} sentences, not {c.spec['exact']}"))

    return violations


def correction_prompt(violations: list[Violation]) -> str:
    """The message handed back to the model. Names each defect precisely, so the
    model repairs a specific fault rather than re-attempting an impossible
    counting task from scratch."""
    lines = [
        "Your previous answer broke constraints the user set. A checker ran over "
        "it and found these specific failures:",
        "",
    ]
    for v in violations:
        lines.append(f"- {v.constraint.description.capitalize()}. {v.detail}")
    lines += [
        "",
        "Rewrite the answer so it satisfies every constraint. Rules for the rewrite:",
        "- Use only real words. Do not delete letters from words to dodge a letter "
        "rule — 'trnspnds' is not a word and does not count as a fix.",
        "- If a constraint genuinely cannot be met with real language, say so "
        "plainly and explain why, rather than shipping something that violates it.",
        "- Output only the corrected answer. Do not mention this correction step.",
    ]
    return "\n".join(lines)
