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
# "sentences must contain exactly 10 words, 8 words, 6 words, and 4 words, in
# that order" / "the first sentence has 4 words, the second has 5 words ...".
# A run of counts, one per sentence — not one count over the whole answer,
# which is how this was read before and why a correct answer was rejected.
_SENTENCE_SERIES_CUE_RE = re.compile(
    r"in\s+that\s+order|\b(?:first|second|third|fourth|fifth)\s+"
    r"(?:sentence|line)\b|\bthe\s+second\s+has\b", re.IGNORECASE)
_COUNT_RUN_RE = re.compile(r"(?P<n>\d{1,3})\s*words?", re.IGNORECASE)

# "Column 2 must contain exactly 5 words in Row 1 and exactly 5 words in Row 2"
# — a count per cell of one column. Suppressing the whole-response count for
# these (below) stopped the false accusation; this makes the check real.
_TABLE_CELL_RE = re.compile(
    r"column\s+(?P<col>\d+)\s+(?:must\s+)?(?:contain|have|be)\s+"
    r"(?:exactly\s+)?(?P<n>\d{1,3})\s+words?", re.IGNORECASE)

# A word count tied to a cell, column, row or line rather than to the answer.
_SCOPED_COUNT_RE = re.compile(
    r"\b(?:column|columns|row|rows|cell|cells|per\s+line|each\s+line|"
    r"every\s+line)\b", re.IGNORECASE)

_PER_ITEM_RE = re.compile(
    r"\b(?:each|every|per)\b|\bfor\s+all\b|\ball\s+\d+\s+items?\b|"
    r"\b(?:both|all)\s+(?:rows|items|entries|objects|elements)\b",
    re.IGNORECASE)
_PER_FIELD_RE = re.compile(
    r"\b(?:each|every|per)\s+(?P<field>[a-z_][a-z0-9_]{2,30})\s+"
    r"(?:to|must\s+be|should\s+be|is)?\s*"
    r"(?:exactly|at\s+most|no\s+more\s+than|under)?\s*\d{1,4}\s+words?",
    re.IGNORECASE,
)


# "no punctuation marks whatsoever" / "do not use any punctuation"
_NO_PUNCT_RE = re.compile(
    r"(?:no|without|not use|avoid|do not use|don'?t use)\s+(?:any\s+)?punctuation",
    re.IGNORECASE)

# "every word must start with a consonant" / "no word may begin with a vowel"
_ALL_CONSONANT_RE = re.compile(
    r"(?:every|each|all)\s+(?:single\s+)?words?\s+(?:in [^.]{0,40})?"
    r"must\s+(?:start|begin)\s+with\s+a\s+consonant", re.IGNORECASE)

# "every word begins with the letter P" / "all words start with 'p'"
_ALL_INITIAL_RE = re.compile(
    r"(?:every|each|all)\s+(?:single\s+)?words?\b[^.]{0,60}?"
    r"(?:start|begin)s?\s+with\s+(?:the\s+letter\s+)?[\"'“‘]?"
    r"(?P<letter>[a-z])[\"'”’]?\b", re.IGNORECASE)

# "alternate between words that start with a vowel and words ... consonant"
_ALTERNATE_RE = re.compile(
    r"alternat\w*\s+between\s+words[^.]{0,80}?vowel[^.]{0,80}?consonant",
    re.IGNORECASE)

# "first letter of each consecutive word must follow the alphabetical sequence"
_ABECEDARIAN_RE = re.compile(
    r"first\s+letter\s+of\s+each[^.]{0,60}?word[^.]{0,80}?"
    r"(?:alphabetical|a-b-c|sequence\s+a)", re.IGNORECASE)

# 'Line 1 must end with "star", Line 2 with "far", Line 3 with "night"'.
# The verb is stated once and elided thereafter, which is how people actually
# write this — requiring "must end" on every clause found only the first line.
_LINE_END_RE = re.compile(
    r"line\s*(?P<n>\d+)\s*(?:must\s+|should\s+)?(?:end(?:s|ing)?\s*)?"
    r"with\s*[\"'“‘](?P<word>[\w'-]+)[\"'”’]", re.IGNORECASE)

# "do NOT use the words 'cold', 'snow'" / "you cannot use the keywords WHERE,
# AND, or JOIN" / "without using the words X, Y or Z".
_BANNED_RE = re.compile(
    r"(?:without\s+using|do\s*not\s+use|don'?t\s+use|cannot\s+use|can'?t\s+use|"
    r"avoid\s+using|never\s+use|must\s+not\s+use|no\s+use\s+of)\s+"
    r"(?:any\s+of\s+)?(?:the\s+)?(?:words?|terms?|keywords?|phrases?)\b"
    r"(?P<body>[^.?!\n]{0,200})", re.IGNORECASE)

# Tokens inside a ban list: quoted, or BARE CAPS like SQL keywords.
_BANNED_TOKEN_RE = re.compile(
    r"[\"'“‘](?P<quoted>[\w'-]+)[\"'”’]|\b(?P<caps>[A-Z]{2,})\b")

# 'the word "heat" must appear exactly once per line'
_PER_LINE_WORD_RE = re.compile(
    r"(?:the\s+)?word\s+[\"'“‘](?P<word>[\w'-]+)[\"'”’]\s*"
    r"must\s+appear\s+exactly\s+(?P<n>\d+|once|twice)\s+(?:time\s+)?per\s+line",
    re.IGNORECASE)

# "no line can exceed 5 total words" / "each line at most 5 words"
_LINE_WORD_MAX_RE = re.compile(
    r"(?:no|each|every)\s+line\s+(?:can\s+)?(?:not\s+)?"
    r"(?:exceed|be\s+(?:longer|more)\s+than|contain\s+more\s+than|"
    r"have\s+more\s+than)\s+(?P<n>\d{1,3})\s*(?:total\s+)?words?",
    re.IGNORECASE)


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

    # A series of per-sentence counts must be read before the single-count
    # patterns, which would otherwise grab the first number in the run and
    # apply it to the entire response.
    series: list[int] = []
    if re.search(r"\bsentences?\b", request, re.IGNORECASE) and \
            _SENTENCE_SERIES_CUE_RE.search(request):
        numbers = [int(m.group("n")) for m in _COUNT_RUN_RE.finditer(request)]
        if len(numbers) >= 2 and all(1 <= n <= 200 for n in numbers):
            series = numbers
            found.append(Constraint(
                "sentence_word_counts", {"counts": series},
                "the sentences must have exactly "
                + ", ".join(str(n) for n in series) + " words, in that order"))

    match = _WORD_EXACT_RE.search(request)
    if match and (series or _SCOPED_COUNT_RE.search(request)):
        # Either the count belongs to a series already captured above, or it
        # is scoped to a cell — "Column 2 must contain exactly 5 words in Row
        # 1". Checking either against the whole response fails every correct
        # answer, and this module would rather miss a constraint than invent a
        # violation.
        match = None
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

    # --- structural constraints -------------------------------------------
    # Every one of these was added because a benchmark run caught the model
    # breaking it and this checker saying nothing. An undetected constraint is
    # not a neutral outcome: the answer is presented as compliant when it is
    # not, which is the failure this whole module exists to prevent.

    match = _BANNED_RE.search(request)
    if match:
        tokens = []
        for hit in _BANNED_TOKEN_RE.finditer(match.group("body")):
            word = (hit.group("quoted") or hit.group("caps") or "").lower()
            if word and word not in tokens:
                tokens.append(word)
        if tokens:
            found.append(Constraint(
                "banned_words", {"words": tokens},
                "the response must not use the word(s) "
                + ", ".join(f"'{w}'" for w in tokens)))

    match = _TABLE_CELL_RE.search(request)
    if match:
        found.append(Constraint(
            "table_cell_words",
            {"column": int(match.group("col")), "words": int(match.group("n"))},
            f"column {match.group('col')} of the table must be exactly "
            f"{match.group('n')} words in every row"))

    if _NO_PUNCT_RE.search(request):
        found.append(Constraint(
            "no_punctuation", {},
            "the response must contain no punctuation marks at all"))

    if _ALL_CONSONANT_RE.search(request):
        found.append(Constraint(
            "word_initial", {"class": "consonant"},
            "every word must start with a consonant"))
    else:
        match = _ALL_INITIAL_RE.search(request)
        if match:
            letter = match.group("letter").lower()
            found.append(Constraint(
                "word_initial", {"letter": letter},
                f"every word must begin with the letter '{letter}'"))

    if _ALTERNATE_RE.search(request):
        found.append(Constraint(
            "alternating_initials", {},
            "words must alternate between vowel-initial and consonant-initial"))

    if _ABECEDARIAN_RE.search(request):
        found.append(Constraint(
            "abecedarian", {},
            "consecutive words must start with A, B, C, D … in order"))

    endings = _LINE_END_RE.findall(request)
    if len(endings) >= 2:
        wanted = [w.lower() for _, w in sorted(endings, key=lambda p: int(p[0]))]
        found.append(Constraint(
            "line_endings", {"words": wanted},
            "the lines must end with: " + ", ".join(f"'{w}'" for w in wanted)))

    match = _PER_LINE_WORD_RE.search(request)
    if match:
        raw = match.group("n").lower()
        times = {"once": 1, "twice": 2}.get(raw, 0) or int(raw or 1)
        found.append(Constraint(
            "word_per_line", {"word": match.group("word").lower(), "times": times},
            f"the word '{match.group('word')}' must appear exactly {times} "
            f"time(s) on every line"))

    match = _LINE_WORD_MAX_RE.search(request)
    if match:
        found.append(Constraint(
            "line_word_max", {"max": int(match.group("n"))},
            f"no line may exceed {match.group('n')} words"))

    return found


# --- verification ----------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return fenced[0].strip() if fenced else text.strip()


def _words(text: str) -> list[str]:
    """Whitespace-separated tokens that contain at least one letter or digit.

    The bare whitespace split counted a markdown bullet "-" as a word, so a
    correctly written list where every word began with P was told that three
    of its words did not. A false accusation costs a whole regeneration and
    teaches the model to distrust a checker that is usually right."""
    return [w for w in re.split(r"\s+", text.strip())
            if w and re.search(r"[A-Za-z0-9]", w)]


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


_VOWELS = frozenset("aeiou")
# Deliberately excludes markdown structure (#, *, |, -) so a bulleted answer is
# not accused of punctuation it did not choose. Sentence punctuation only.
_PUNCTUATION = frozenset(".,;:!?'\"()[]{}")


def _lines(text: str) -> list[str]:
    """Non-empty lines, with list bullets and numbering stripped.

    A model asked for "3 lines" writes a bulleted list about half the time.
    Counting the bullet as content would fail a correct answer."""
    out = []
    for raw in _strip_code_fences(text).splitlines():
        line = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", raw).strip()
        if line:
            out.append(line)
    return out


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
    # A per-item count implies structured output even when the request never
    # said the word "JSON" — "every description must be 8 words" only makes
    # sense over items. Try parsing regardless, or the check silently no-ops.
    if any(c.kind == "word_count" and c.spec.get("per_item") for c in constraints):
        try:
            parsed = json.loads(_strip_code_fences(response))
        except json.JSONDecodeError:
            parsed = None

    for c in constraints:
        if c.kind == "json":
            body = _strip_code_fences(response)
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as e:
                parsed = None
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

        elif c.kind == "sentence_word_counts":
            sentences = _sentences(_strip_code_fences(response))
            wanted = c.spec["counts"]
            if len(sentences) != len(wanted):
                violations.append(Violation(
                    c, f"the response has {len(sentences)} sentences, "
                       f"not {len(wanted)}"))
            else:
                bad = [f"sentence {i + 1} has {len(_words(sent))} "
                       f"(needs {want})"
                       for i, (sent, want) in enumerate(zip(sentences, wanted))
                       if len(_words(sent)) != want]
                if bad:
                    violations.append(Violation(c, "; ".join(bad)))

        elif c.kind == "table_cell_words":
            rows = [ln.strip() for ln in _strip_code_fences(response).splitlines()
                    if ln.strip().startswith("|")]
            body = [r for r in rows
                    if not re.fullmatch(r"\|[\s:|-]+\|", r)][1:]
            if not body:
                violations.append(Violation(
                    c, "the response contains no markdown table rows"))
            else:
                index, want = c.spec["column"] - 1, c.spec["words"]
                bad = []
                for i, row in enumerate(body):
                    cells = [cell.strip() for cell in row.strip("|").split("|")]
                    if index >= len(cells):
                        bad.append(f"row {i + 1} has only {len(cells)} columns")
                        continue
                    n = len(_words(cells[index]))
                    if n != want:
                        bad.append(f"row {i + 1} has {n} (needs {want}): "
                                   f"\"{cells[index]}\"")
                if bad:
                    violations.append(Violation(c, "; ".join(bad[:8])))

        elif c.kind == "banned_words":
            low = response.lower()
            hits = [w for w in c.spec["words"]
                    if re.search(rf"\b{re.escape(w)}\b", low)]
            if hits:
                violations.append(Violation(
                    c, "the response uses " + ", ".join(f"'{h}'" for h in hits)))

        elif c.kind == "no_punctuation":
            hits = sorted({ch for ch in response if ch in _PUNCTUATION})
            if hits:
                violations.append(Violation(
                    c, "the response uses " + ", ".join(f"'{h}'" for h in hits)))

        elif c.kind == "word_initial":
            words = _words(_strip_code_fences(response))
            if c.spec.get("class") == "consonant":
                bad = [w for w in words if w[0].lower() in _VOWELS]
                if bad:
                    violations.append(Violation(
                        c, f"{len(bad)} words start with a vowel: "
                           f"{', '.join(bad[:MAX_LETTER_EXAMPLES])}"))
            else:
                letter = c.spec["letter"]
                bad = [w for w in words if not w.lower().startswith(letter)]
                if bad:
                    violations.append(Violation(
                        c, f"{len(bad)} words do not start with '{letter}': "
                           f"{', '.join(bad[:MAX_LETTER_EXAMPLES])}"))

        elif c.kind == "alternating_initials":
            words = _words(_strip_code_fences(response))
            kinds = [w[0].lower() in _VOWELS for w in words]
            breaks = [i for i in range(1, len(kinds)) if kinds[i] == kinds[i - 1]]
            if breaks:
                sample = ", ".join(f"'{words[i - 1]} {words[i]}'"
                                   for i in breaks[:MAX_LETTER_EXAMPLES])
                violations.append(Violation(
                    c, f"the alternation breaks at {len(breaks)} places: {sample}"))

        elif c.kind == "abecedarian":
            words = _words(_strip_code_fences(response))
            wanted = [chr(ord("a") + i) for i in range(len(words))]
            got = [w[0].lower() for w in words]
            if got != wanted:
                first = next((i for i, (a, b) in enumerate(zip(got, wanted))
                              if a != b), min(len(got), len(wanted)))
                violations.append(Violation(
                    c, f"word {first + 1} starts with "
                       f"'{got[first] if first < len(got) else '?'}', not "
                       f"'{wanted[first] if first < len(wanted) else '?'}' "
                       f"(initials so far: {''.join(got[:12])})"))

        elif c.kind == "line_endings":
            lines = _lines(response)
            wanted = c.spec["words"]
            if len(lines) != len(wanted):
                violations.append(Violation(
                    c, f"the response has {len(lines)} lines, not {len(wanted)}"))
            else:
                bad = []
                for i, (line, want) in enumerate(zip(lines, wanted)):
                    words = _words(line)
                    got = words[-1].lower().strip("'") if words else ""
                    if got != want:
                        bad.append(f"line {i + 1} ends '{got}', needs '{want}'")
                if bad:
                    violations.append(Violation(c, "; ".join(bad)))

        elif c.kind == "word_per_line":
            word, times = c.spec["word"], c.spec["times"]
            bad = []
            for i, line in enumerate(_lines(response)):
                n = len(re.findall(rf"\b{re.escape(word)}\b", line, re.IGNORECASE))
                if n != times:
                    bad.append(f"line {i + 1} has it {n}x")
            if bad:
                violations.append(Violation(
                    c, f"'{word}' must appear {times}x per line — "
                       + "; ".join(bad[:8])))

        elif c.kind == "line_word_max":
            bad = [f"line {i + 1} has {len(_words(line))}"
                   for i, line in enumerate(_lines(response))
                   if len(_words(line)) > c.spec["max"]]
            if bad:
                violations.append(Violation(
                    c, f"max {c.spec['max']} words per line — " + "; ".join(bad[:8])))

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
