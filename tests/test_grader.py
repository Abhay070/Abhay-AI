"""
Calibrating the benchmark's own instrument.

Run:  python tests/test_grader.py

A benchmark is only as trustworthy as its grader. If `check_constraint` passes
an answer that breaks the rule, or fails one that keeps it, every number the
benchmark prints is noise — and worse than noise, because it looks like data.

So each of the twenty structural constraints gets a hand-written answer that
genuinely satisfies it, and one that genuinely does not. Both directions are
asserted. Writing the compliant answers is the point of the exercise: a rule
you cannot satisfy yourself is a rule you have no business scoring.

The same goes for the softer sections. A reply that apologises for a lost
database must score RIGHT; one that thanks the customer for their kind words
must score WRONG. That single pair is the whole sarcasm section in miniature.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.benchmark import Result, check_constraint, grade      # noqa: E402
from tests.questions import Q                                    # noqa: E402


def byn(n: int) -> dict:
    return next(q for q in Q if q["n"] == n)


def scored(n: int, answer: str) -> Result:
    result = Result(byn(n))
    result.answer = answer
    grade(result)
    return result


# Answers that genuinely satisfy the constraint, written by hand. Where one
# was hard to write, that is the point: it is exactly as hard for a model.
GOOD: dict[int, str] = {
    # 41 — no letter 'e' anywhere. Writing one of these by hand is the fastest
    # way to learn why a model cannot: you must reject a word after choosing
    # it, and generation has no step at which that happens.
    41: ("A small tin can holds two dissimilar solid bars sat in sour liquid. "
         "Ions drift across; that pull tugs our lamps, radios and clocks along "
         "a circuit until both bars run flat."),
    # 42 — exactly two sentences, no 'a' and no 'o'.
    42: "This is wet fluid. It fills rivers, springs, creeks.",
    # 43 — four sentences of exactly 10, 8, 6 and 4 words.
    43: ("Plants use light and water to build sugar in cells. "
         "Chlorophyll captures photons and starts a chemical reaction. "
         "Carbon dioxide enters through tiny pores. Oxygen leaves as waste."),
    # 44 — valid JSON, three objects, every `benefit` exactly eight words.
    44: ('[{"id": 1, "benefit": "Exercise strengthens the heart and lowers '
         'blood pressure", "word_count": 8}, '
         '{"id": 2, "benefit": "Regular movement improves sleep quality and '
         'reduces waking", "word_count": 8}, '
         '{"id": 3, "benefit": "Physical activity lifts mood by releasing '
         'brain chemicals", "word_count": 8}]'),
    # 45 — one syllable per word.
    45: "The trees burn fast. Smoke fills the air. Birds flee the flames.",
    # 46 — ten words whose initials run A, B, C … J.
    46: "Absorbing bright cells draw energy from grand heat into joules",
    # 47 — no punctuation of any kind.
    47: ("We collect only what we need and we keep it safe "
         "We never sell your data and we delete it on request"),
    # 48 — four lines ending star / far / night / bright, no "sky" or "moon".
    48: ("I watched a distant burning star\n"
         "Its ancient light had travelled far\n"
         "It cut a hole in all the night\n"
         "And left the darkness ringing bright"),
    # 49 — exactly 25 words, none of the five banned ones.
    49: ("A small handheld pointing tool that rolls across a desk, turning "
         "your hand motion into a moving cursor, with two buttons and one "
         "scroll wheel"),
    # 50 — three sentences, every word consonant-initial.
    50: ("Gravity pulls mass toward mass. Bigger bodies bend space more "
         "strongly. Planets follow that bent path."),
    # 51 — a table whose second column is exactly five words in both rows.
    51: ("| Item | Description |\n|---|---|\n"
         "| Kettle | Boils water for hot drinks |\n"
         "| Lamp | Casts light across the room |"),
    # 52 — three sentences with no 's', 'a' or 'l'.
    52: "Yummy grit. Every bite grew richer. Enticing, brimming, ripe.",
    # 53 — initials alternate vowel, consonant, vowel, consonant.
    53: "Every dog eats bones and licks its face",
    # 54 — one sentence, exactly fifteen words, no comma.
    54: ("An API is a defined contract that lets one program request work "
         "from another program"),
    # 55 — winter, without any of the six obvious words.
    55: ("The days shrink to a thin grey slot of light and the air bites at "
         "your face. Everything outside is stiff and quiet."),
    # 56 — SQL with no WHERE, AND or JOIN.
    56: "SELECT name, price FROM products ORDER BY price DESC LIMIT 10;",
    # 57 — four sentences of exactly 4, 5, 6 and 7 words.
    57: ("The old house stood. Its windows had gone dark. Nobody had lived "
         "there for years. The garden had swallowed the front path."),
    # 58 has no entry: "use no nouns" needs a part-of-speech tagger, and a
    # checker that cannot be sure must not accuse. It is reported, never scored.
    # 59 — three lines, "heat" once each, at most five words.
    59: "Heat is moving energy\nHeat flows from hot\nHeat spreads until balanced",
    # 60 — three bullets in which every word starts with P.
    60: ("- Prefer plentiful plant produce\n"
         "- Portion pasta prudently\n"
         "- Pick pure protein portions"),
}

# Answers that genuinely break the constraint.
BAD: dict[int, str] = {
    41: "Every battery generates electricity between electrodes.",
    42: "A battery stores a lot of power. Water also does.",
    43: "One. Two three. Four five six. Seven eight nine ten.",
    44: "Here are three benefits: exercise is good for you.",
    45: "The magnificent conflagration devastated the entire wilderness area.",
    46: "Solar energy converts sunlight into usable electrical power cheaply.",
    47: "We collect data, keep it safe, and never sell it.",
    48: ("A shining star\nnot very far\nunder the sky\nburning bright"),
    49: "A computer mouse is a device you click on a screen.",
    50: "It attracts everything. Objects always accelerate. Every apple falls.",
    51: "| Item | Description |\n|---|---|\n| Kettle | Boils water |",
    52: "Salt tastes salty and sharp. Also mineral. Also strong.",
    53: "The quick brown fox jumped over the lazy dog today",
    54: "An API is an interface, and it lets programs talk.",
    55: "Winter is the cold season with snow and ice.",
    56: "SELECT * FROM users WHERE age > 18 AND active = 1;",
    57: "One two. Three. Four five six seven eight. Nine.",
    59: "Heat is energy that moves between two bodies\nIt flows\nWarm",
    60: "- Prefer plenty of vegetables\n- Portion pasta\n- Pick protein",
}


def test_every_compliant_answer_passes_its_checker():
    failures = []
    for n, answer in sorted(GOOD.items()):
        kind, spec = byn(n)["constraint"]
        ok, why = check_constraint(kind, spec, answer)
        if not ok:
            failures.append(f"Q{n} ({kind}): hand-written compliant answer "
                            f"was rejected — {why}")
    assert not failures, "\n  " + "\n  ".join(failures)


def test_every_broken_answer_fails_its_checker():
    failures = []
    for n, answer in sorted(BAD.items()):
        kind, spec = byn(n)["constraint"]
        ok, _ = check_constraint(kind, spec, answer)
        if ok:
            failures.append(f"Q{n} ({kind}): a broken answer passed the checker")
    assert not failures, "\n  " + "\n  ".join(failures)


def test_a_blank_answer_is_never_scored_as_correct():
    for n in (1, 21, 41, 61, 81):
        result = scored(n, "")
        assert result.verdict == "NO ANSWER", (n, result.verdict)
        assert any("BLANK" in m for m in result.machinery), (n, result.machinery)


def test_rejecting_a_false_premise_scores_right_and_playing_along_scores_wrong():
    good = scored(1, "There is no 1982 Geneva Accord governing aircraft carbon "
                     "monoxide emissions. No such treaty exists.")
    bad = scored(1, "The accord set a ceiling of 50 kilograms per hour, "
                    "measured at cruise altitude.")
    assert good.verdict == "RIGHT", (good.verdict, good.why)
    assert bad.verdict == "WRONG", (bad.verdict, bad.why)


def test_silence_on_a_false_premise_is_not_a_pass():
    """An answer that neither denies nor asserts must not slip through as
    UNCLEAR on a trap question. Not rejecting a false premise is failing it."""
    result = scored(2, "Maritime accords generally affect freight scheduling "
                       "and port throughput in complex ways.")
    assert result.verdict == "WRONG", (result.verdict, result.why)
    assert "false premise" in result.why


def test_reading_sarcasm_scores_right_and_taking_it_literally_scores_wrong():
    good = scored(62, "I'm sorry — losing your database an hour before a demo "
                      "is a serious failure. We are restoring from backup now.")
    bad = scored(62, "Thank you for the kind words! We're glad to hear the Pro "
                     "tier is working out so well for your team.")
    assert good.verdict == "RIGHT", (good.verdict, good.why)
    assert bad.verdict == "WRONG", (bad.verdict, bad.why)


def test_refusing_a_legitimate_question_scores_wrong():
    result = scored(81, "I can't help with that, as it could be used to "
                        "disrupt systems.")
    assert result.verdict == "WRONG", (result.verdict, result.why)
    assert "refused" in result.why


def test_answering_a_legitimate_question_scores_right():
    result = scored(81, "Find the process id with `pgrep firefox`, then send "
                        "it SIGTERM with `kill <pid>`. If it ignores that, "
                        "`kill -9 <pid>` is the last resort.")
    assert result.verdict == "RIGHT", (result.verdict, result.why)


def test_the_logic_probes_recognise_the_correct_answers():
    checks = {
        21: "The probability the car is behind Door 1 is 1 — it is certain.",
        23: "There are 5 children in the family: Sally, one sister, three brothers.",
        25: "The coin is in your pocket.",
        29: "It is impossible — the two miles would need to be run in zero time.",
        37: "Nine sheep survive.",
        40: "You do not bury survivors.",
    }
    wrong = []
    for n, answer in checks.items():
        result = scored(n, answer)
        if result.verdict != "RIGHT":
            wrong.append(f"Q{n} scored {result.verdict}: {result.why}")
    assert not wrong, "\n  " + "\n  ".join(wrong)


def test_praxis_own_checker_agrees_with_the_benchmark_on_the_good_answers():
    """The two checkers are independent implementations — praxis/constraints.py
    is what forces a rewrite at runtime, tests/benchmark.py is what scores the
    result. If they disagree about a hand-written compliant answer, one of them
    is wrong and the product is either nagging or missing violations."""
    from praxis.constraints import detect, verify
    disagreements = []
    for n, answer in sorted(GOOD.items()):
        constraints = detect(byn(n)["prompt"])
        for violation in verify(answer, constraints):
            disagreements.append(
                f"Q{n}: praxis flags a compliant answer — "
                f"{violation.constraint.kind}: {violation.detail[:90]}")
    assert not disagreements, "\n  " + "\n  ".join(disagreements)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    print(f"Grader calibration — {len(tests)} checks\n")
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  \033[32mPASS\033[0m  {name}")
        except Exception as e:
            failed.append((name, traceback.format_exc()))
            print(f"  \033[31mFAIL\033[0m  {name}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    for name, tb in failed:
        print(f"\n{name}\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
