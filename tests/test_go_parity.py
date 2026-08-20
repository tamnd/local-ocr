"""The Python rules against the Go rules, case by case.

The fixtures in `tests/fixtures/` are not written by hand. `cmd/ocrfixtures` in
`tamnd/bourbaki-solver` calls `ocr.Validate`, `textguard.Check`,
`textguard.Normalise` and `mathtex.Split` and records what they return, so every
expectation in this file is the original's own answer and not somebody's reading
of the original.

When a rule changes over there, the fixture changes and these tests fail. That
is the mechanism working, and the correct response is to look at the Go diff
rather than to loosen an assertion here.

Regenerate with:

    go run ./cmd/ocrfixtures -out ../local-ocr/tests/fixtures
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_ocr.rules import mathtex, textguard
from local_ocr.rules.validate import Confidence, Expect, Grammar, validate

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str, key: str) -> list[dict]:
    body = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    cases = body[key]
    assert cases, f"{name} has no cases in it, which is not a passing test"
    return cases


def ident(case: dict) -> str:
    return case["name"]


VALIDATE = load("validate.json", "validate")
LEAKS = load("leaks.json", "leaks")
NORMALISE = load("normalise.json", "normalise")


def expect_of(raw: dict) -> Expect:
    return Expect(
        book=raw["book"],
        pdf_page=raw["pdf_page"],
        blank=raw["blank"],
        sparse=raw["sparse"],
        grammar=Grammar(raw["grammar"]),
        chapter=raw["chapter"],
        page=raw["page"],
        confidence=Confidence(raw["confidence"]),
        has_head=raw["has_head"],
    )


@pytest.mark.parametrize("case", VALIDATE, ids=ident)
def test_the_eight_rules_return_what_the_go_rules_return(case: dict) -> None:
    """Rule, detail and line, not merely accepted or rejected.

    The detail is what the failures report prints and the line is what the
    repair prompt names, so a rule that rejects the right page for the wrong
    stated reason still turns a targeted fix into a re-read.
    """
    got = validate(case["text"], expect_of(case["expect"]))
    want = case["problems"]
    assert [str(p.rule) for p in got] == [w["rule"] for w in want], case["why"]
    assert [p.detail for p in got] == [w["detail"] for w in want]
    assert [p.line for p in got] == [w["line"] for w in want]


@pytest.mark.parametrize("case", LEAKS, ids=ident)
def test_the_leak_check_finds_what_the_go_one_finds(case: dict) -> None:
    got = textguard.check(case["text"])
    want = case["leaks"]
    assert [leak.kind for leak in got] == [w["kind"] for w in want], case["why"]
    assert [leak.detail for leak in got] == [w["detail"] for w in want]
    assert [leak.line for leak in got] == [w["line"] for w in want]


@pytest.mark.parametrize("case", NORMALISE, ids=ident)
def test_normalisation_produces_the_same_text(case: dict) -> None:
    assert textguard.normalise(case["in"]) == case["normalise"], case["why"]


@pytest.mark.parametrize("case", NORMALISE, ids=ident)
def test_prose_normalisation_produces_the_same_text(case: dict) -> None:
    assert textguard.normalise_prose(case["in"]) == case["normalise_prose"], case["why"]


@pytest.mark.parametrize("case", NORMALISE, ids=ident)
def test_the_delimiter_rewrite_agrees_and_counts_the_same(case: dict) -> None:
    text, n = textguard.dollars(case["in"])
    assert text == case["dollars"], case["why"]
    assert n == case["dollars_count"]


@pytest.mark.parametrize("case", NORMALISE, ids=ident)
def test_the_star_rewrite_agrees_and_counts_the_same(case: dict) -> None:
    text, n = textguard.stars(case["in"])
    assert text == case["stars"], case["why"]
    assert n == case["stars_count"]


@pytest.mark.parametrize("case", NORMALISE, ids=ident)
def test_stripping_a_wrapper_agrees(case: dict) -> None:
    assert textguard.strip(case["in"]) == case["strip"], case["why"]


@pytest.mark.parametrize("case", NORMALISE, ids=ident)
def test_normalisation_is_idempotent_wherever_the_go_side_says_it_is(case: dict) -> None:
    """`bourbaki fix dollars` is run over the corpus as often as anyone likes."""
    once = textguard.normalise(case["in"])
    assert (textguard.normalise(once) == once) == case["normalise_is_idempotent"]


@pytest.mark.parametrize("case", NORMALISE, ids=ident)
def test_the_math_spans_are_in_the_same_places(case: dict) -> None:
    """Down to the rune offsets, because `stars` decides by them.

    The same asterisk is a binary law inside a span and Bourbaki's forward
    reference mark outside one. A splitter that is off by a rune rewrites the
    wrong glyph, and nothing on the rendered page looks wrong afterwards.
    """
    spans, unclosed = mathtex.split(case["normalise"])
    want = case["spans"]
    assert [s.text for s in spans] == [w["text"] for w in want], case["why"]
    assert [s.display for s in spans] == [w["display"] for w in want]
    assert [s.line for s in spans] == [w["line"] for w in want]
    assert [(s.start, s.end) for s in spans] == [(w["start"], w["end"]) for w in want]
    if case["unclosed"] is None:
        assert unclosed is None
    else:
        assert unclosed is not None
        assert unclosed.text == case["unclosed"]["text"]
        assert unclosed.line == case["unclosed"]["line"]
        assert unclosed.start == case["unclosed"]["start"]


def test_the_fixtures_cover_every_rule() -> None:
    """A parity suite that never exercises a rule is a parity suite with a hole.

    Rule 7 is deliberately absent: it delegates to a TeX installation through an
    interface, so a fixture for it would pin the stub and not the rule.
    """
    fired = {p["rule"] for case in VALIDATE for p in case["problems"]}
    assert fired == {"short", "math", "leak", "head", "illegible", "label", "exercise"}


def test_the_fixtures_cover_every_kind_of_leak() -> None:
    found = {leak["kind"] for case in LEAKS for leak in case["leaks"]}
    assert found == {"empty", "refusal", "no-image", "prompt", "thinking", "meta", "markup"}
