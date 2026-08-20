"""The eight rules that decide whether what came back is a transcription.

A transcription of `ocr/validate.go` in `tamnd/bourbaki-solver`. The decision is
the expensive half. On the fleet a call costs about 151 seconds, and an accepted
page that is wrong is not caught by anything downstream until a person reads it.
On a local GPU the call is cheaper and the argument is unchanged, because the
cost that matters is the wrong page sitting in the corpus and not the seconds.

Everything here rejects cheaply and says which rule rejected, because the retry
that follows is chosen from the reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from local_ocr.rules import textguard
from local_ocr.rules.goquote import quote

MIN_CHARS = 200
"""The shortest a real page can be.

The thinnest page of either volume that carries text is the seven line close of
chapter III, about 400 characters. Two hundred is under half of that and still
well above what a truncated answer or a one line apology comes to.
"""

MAX_ILLEGIBLE = 2
"""How many unreadable spots a page may carry and still be accepted.

Up to two is a damaged scan, which these are. More than that is a model that
gave up, and re-reading at 600 dpi is worth the call.
"""

ILLEGIBLE = "⟪illegible⟫"
"""What the prompt asks for in place of a guess."""

SPARSE_INK = 0.01
"""The ink ratio below which a page is too thin to have a length expected of it.

A full page of these volumes runs about 6.5 percent; the title pages and the
publisher's device come in under half a percent.
"""


class Rule(StrEnum):
    """The check that rejected a page. The retry policy switches on it."""

    SHORT = "short"  # 1, empty or under MIN_CHARS
    MATH = "math"  # 2, unbalanced $ or $$
    LEAK = "leak"  # 3, refusal, narration or the prompt back
    HEAD = "head"  # 4, no plausible running head on the first line
    ILLEGIBLE = "illegible"  # 5, too many unreadable spots
    LABEL = "label"  # 6, the page label contradicts the page map
    LATEX = "latex"  # 7, the LaTeX does not compile
    EXERCISE = "exercise"  # 8, an exercise below the exercises head is a heading


class Grammar(StrEnum):
    """How a volume prints its page number."""

    HEAD_LABEL = "head-label"
    FOOT_NUMBER = "foot-number"
    HEAD_NUMBER = "head-number"


class Confidence(StrEnum):
    """Where the page map's number came from."""

    FROM_HEAD = "head"
    FROM_FOOT = "foot"
    INTERPOLATED = "interpolated"
    UNKNOWN = "none"

    @property
    def printed(self) -> bool:
        """Whether the number was read off the page rather than worked out."""
        return self in (Confidence.FROM_HEAD, Confidence.FROM_FOOT)


@dataclass(frozen=True)
class Problem:
    """One reason a page was not accepted."""

    rule: Rule
    detail: str
    line: int = 0

    def __str__(self) -> str:
        if self.line > 0:
            return f"{self.rule}: {self.detail} (line {self.line})"
        return f"{self.rule}: {self.detail}"


@dataclass(frozen=True)
class Expect:
    """What the rest of the pipeline already knows about a page.

    Which is what makes rules 4 and 6 possible at all.
    """

    book: str = ""
    pdf_page: int = 0
    blank: bool = False
    sparse: bool = False
    """A page with ink on it but very little: a title page, a part title, the
    publisher's device. Rule 1 does not run on one.

    Without this the rule fires on pages that are short because the book is
    short there. Measured: alg-iv-vii page 3 is the Springer knight and nothing
    else at 0.47 percent ink. Each would burn three calls and then be filed as a
    defect that is not one. A volume that was never rendered leaves this false
    and the rule runs, which is the safe way round.
    """
    grammar: Grammar = Grammar.HEAD_LABEL
    chapter: str = ""
    page: int = 0
    """What the page map says, and 0 when it does not know."""
    confidence: Confidence = Confidence.UNKNOWN
    """An interpolated number is a guess, and a guess must not reject a page
    that was read correctly, so rule 6 only runs on a printed one."""
    has_head: bool = False
    """False for the pages that print no running head at all: the first page of
    a chapter, a part title, the pages of the front matter."""


def validate(text: str, expect: Expect, latex: object | None = None) -> list[Problem]:
    """Apply the rules in order and return everything that failed.

    All of them run rather than stopping at the first. A page that is short and
    has no running head is a different failure from one that is only short: the
    first is a truncated answer and the second is a page whose head the scan
    ate, and the retry differs.

    `latex` is rule 7 and is opt in, because it needs a TeX installation and
    costs a subprocess per page. It is anything with a `check(fragment)` that
    raises on a fragment that does not compile.
    """
    problems: list[Problem] = []

    # Rule 3 first, even though it is numbered third. A refusal is short and has
    # no running head and no mathematics, so checking it first means the report
    # says what actually happened rather than listing four symptoms of it.
    for leak in textguard.check(text):
        problems.append(Problem(Rule.LEAK, f"{leak.kind}: {leak.detail}", leak.line))
    if problems:
        return problems

    body = text.strip()

    # Rule 1.
    if not expect.blank and not expect.sparse and len(body) < MIN_CHARS:
        problems.append(Problem(Rule.SHORT, f"{len(body)} characters, want at least {MIN_CHARS}"))

    # Rule 2.
    problem = _check_math(body)
    if problem is not None:
        problems.append(problem)

    # Rule 5.
    spots = body.count(ILLEGIBLE)
    if spots > MAX_ILLEGIBLE:
        problems.append(
            Problem(
                Rule.ILLEGIBLE,
                f"{spots} unreadable spots, want at most {MAX_ILLEGIBLE}",
            )
        )

    # Rules 4 and 6 both read the first line.
    head = _first_line(body)
    if expect.has_head:
        problem = _check_head(head, expect)
        if problem is not None:
            problems.append(problem)
    problem = _check_label(head, expect)
    if problem is not None:
        problems.append(problem)

    # Rule 8.
    problem = _check_after_exercises(body)
    if problem is not None:
        problems.append(problem)

    # Rule 7.
    if latex is not None:
        try:
            latex.check(body)  # type: ignore[attr-defined]
        except Exception as err:
            problems.append(Problem(Rule.LATEX, str(err)))
    return problems


def ok(text: str, expect: Expect, latex: object | None = None) -> bool:
    """Whether a page is accepted."""
    return not validate(text, expect, latex)


def _check_math(text: str) -> Problem | None:
    """Rule 2, counted twice.

    The delimiters are counted over the page and then again paragraph by
    paragraph, because the first count on its own is too weak to catch what the
    fleet actually returns. The paragraph check runs first even though the page
    count is the cruder and more obvious of the two, because both fire on a
    single unclosed formula and only one of them can say which line it is on.
    The repair prompt names the line, so a problem with no line in it turns a
    targeted fix into a re-read.
    """
    problem = _check_inline_runs(text)
    if problem is not None:
        return problem
    return _check_math_totals(text)


def _check_math_totals(text: str) -> Problem | None:
    """Count the delimiters over the whole page.

    The two forms have to be counted together and in one pass, because `$$` is
    two dollars and a naive count of `$` says every display page is unbalanced.
    Escaped dollars are literal and do not count, and Bourbaki uses them: the
    volumes print prices in the historical notes.
    """
    inline = display = 0
    runes = list(text)
    i = 0
    while i < len(runes):
        if runes[i] == "\\":
            i += 2  # whatever follows a backslash is literal, dollar included
            continue
        if runes[i] != "$":
            i += 1
            continue
        if i + 1 < len(runes) and runes[i + 1] == "$":
            display += 1
            i += 2
            continue
        inline += 1
        i += 1
    if display % 2 != 0:
        return Problem(Rule.MATH, f"{display} display delimiters, an odd number")
    if inline % 2 != 0:
        return Problem(Rule.MATH, f"{inline} inline delimiters, an odd number")
    return None


def _check_inline_runs(text: str) -> Problem | None:
    """Take the same count paragraph by paragraph.

    Parity over the whole page passes a page with two unclosed dollars as
    readily as one with none, and that is not a corner case. Page 53 of Algebra
    I came back with "and all $x\\in E." at the end of one paragraph and
    "$\\sup$ and $\\inf." at the end of another, both missing their closing
    dollar, and the two odd counts added to an even one. The page was accepted
    with two broken formulae on it and no flag.

    An inline formula does not run across a blank line anywhere in these
    volumes, so a dollar still open when a paragraph ends is an unclosed one.

    Dollars inside a display block are not inline delimiters and are not
    counted, which is what the display parity is tracked for here: a `$$` block
    that spans a blank line leaves the count odd, and while it is odd the rule
    stands down.
    """
    open_line = display = 0
    for i, raw in enumerate(text.split("\n")):
        if not raw.strip():
            if open_line > 0 and display % 2 == 0:
                return _unclosed(open_line)
            continue
        runes = list(raw)
        j = 0
        while j < len(runes):
            if runes[j] == "\\":
                j += 2
                continue
            if runes[j] != "$":
                j += 1
                continue
            if j + 1 < len(runes) and runes[j + 1] == "$":
                display += 1
                j += 2
                continue
            j += 1
            if display % 2 != 0:
                continue  # inside a display block, not an inline delimiter
            open_line = 0 if open_line > 0 else i + 1
    if open_line > 0 and display % 2 == 0:
        return _unclosed(open_line)
    return None


def _unclosed(line: int) -> Problem:
    return Problem(
        Rule.MATH,
        "an inline $ opened on this line is never closed before the paragraph ends",
        line,
    )


# The head a volume prints over the exercises of a chapter, English and French,
# at whatever level the reading gave it.
_EXERCISE_HEAD = re.compile(r"#+\s*(EXERCISES|EXERCICES)\s*", re.A)

# A no. or a statement set as a heading, which is what the body of a section is
# written with and what nothing below the exercises head can be. The section
# head itself is left out of it: a chapter gathers the exercises of all its
# sections under one head and divides them by section, so "## § 3" below the
# exercises head is the printing and not a defect.
_NUMBERED_HEAD = re.compile(r"#{3,}\s", re.A)


def _check_after_exercises(text: str) -> Problem | None:
    """Rule 8.

    Nothing of a section comes after its exercises, so a no. or a statement
    heading below the exercises head is the reading of the page and not the
    page. Page 289 of Theory of Sets prints EXERCISES, then § 1, then "1. Let S
    be the set of signs P, X, ..." as an ordinary paragraph, and the reading
    came back with that paragraph as a heading. Every other exercise of the
    volume, on 32 other pages, came back as the prose it is, including the one
    on the very next page, which opens the same way word for word.

    It only sees the page it is given, and a volume prints EXERCISES once over a
    block that runs for pages. An exercise set as a heading on the second page
    of such a block goes past this rule, and the assembler still catches it.
    """
    after = False
    for i, line in enumerate(text.split("\n")):
        if _EXERCISE_HEAD.fullmatch(line.strip()):
            after = True
            continue
        if after and _NUMBERED_HEAD.match(line):
            return Problem(
                Rule.EXERCISE,
                "an exercise below the exercises head is set as a heading: " + quote(_clip(line)),
                i + 1,
            )
    return None


def _check_head(head: str, expect: Expect) -> Problem | None:
    """Rule 4.

    What counts as a plausible head depends on how the volume prints one. A
    volume with head labels has to show one. A volume that prints its number at
    the foot has only the chapter name or the section locator up there, and the
    most that can be asked is that the first line is a head and not the first
    sentence of a paragraph: short, and not ending in a full stop.
    """
    if not head.strip():
        return Problem(
            Rule.HEAD,
            "the first line is empty, the page map says this page has a running head",
            1,
        )
    if expect.grammar == Grammar.HEAD_LABEL:
        if parse_page_label(head) is None:
            return Problem(Rule.HEAD, f"no page label in the first line: {quote(_clip(head))}", 1)
        return None
    # FootNumber. A running head here is a chapter name in capitals, a section
    # locator, or a bare number. A line of prose is none of those.
    if parse_section_locator(head) is not None:
        return None
    if _looks_like_head(head):
        return None
    return Problem(
        Rule.HEAD,
        f"the first line reads as prose, not a running head: {quote(_clip(head))}",
        1,
    )


def _looks_like_head(line: str) -> bool:
    """Deliberately loose.

    Rejecting a good page costs a whole read and lands it in the failures
    report; letting a doubtful one through costs a line in the audit, which a
    person reads anyway.
    """
    line = line.strip()
    if len(line) > 90:
        return False  # a running head is not a paragraph
    if line.endswith(".") and not line.endswith("no."):
        return False  # a sentence, not a head
    letters = upper = 0
    for ch in line:
        if "a" <= ch <= "z":
            letters += 1
        elif "A" <= ch <= "Z":
            letters += 1
            upper += 1
    if letters == 0:
        return True  # a bare number or a locator
    # Bourbaki sets its running heads in capitals. Half is enough, because the
    # small capitals of a chapter title come back from OCR mixed.
    return upper * 2 >= letters


def _check_label(head: str, expect: Expect) -> Problem | None:
    """Rule 6.

    It only runs where a page label is printed and where the page map read its
    number off a page rather than working it out, because two guesses
    disagreeing says nothing. One page of slack is allowed: the head of a verso
    and the head of the facing recto differ by one, and a scan that is off by a
    page in its own numbering is a known feature of these files.
    """
    if expect.grammar != Grammar.HEAD_LABEL or expect.page == 0 or not expect.confidence.printed:
        return None
    label = parse_page_label(head)
    if label is None:
        return None  # rule 4 has already said so, do not say it twice
    if expect.chapter and label.chapter != expect.chapter:
        return Problem(
            Rule.LABEL,
            f"the page says chapter {label.chapter}, the page map says {expect.chapter}",
            1,
        )
    diff = label.page - expect.page
    if diff > 1 or diff < -1:
        return Problem(
            Rule.LABEL,
            f"the page says {label.page}, the page map says {expect.page}",
            1,
        )
    return None


@dataclass(frozen=True)
class PageLabel:
    """A Bourbaki page reference such as "A VIII.13".

    The Book's letter, the chapter in Roman, and the page within that chapter.
    Bourbaki cross references are page based, so this is a primary key and not
    decoration.
    """

    book: str
    chapter: str
    page: int


# The volumes print the same label several ways, so the separators are
# deliberately loose: "A VIII.13", "A.IV.3", "A. IV. 2", "A.V . 36". A comma is
# not accepted between the Book and the chapter, because that is how a prose
# cross reference reads and those are parsed elsewhere.
_PAGE_LABEL = re.compile(r"\b([A-Z]{1,3})[.\s]\s*([IVXLCDM]+)\s*[.,]\s*(\d{1,4})\b", re.A)


def parse_page_label(text: str) -> PageLabel | None:
    """Find a page label anywhere in text, as it appears in a running head."""
    match = _PAGE_LABEL.search(text)
    if match is None:
        return None
    return PageLabel(match.group(1), match.group(2), int(match.group(3)))


@dataclass(frozen=True)
class SectionLocator:
    """The other kind of running head, "§ 6.5", meaning § 6 no. 5.

    The 1998 printing of chapters I to III carries no page label at all, only
    the chapter numeral on one side and this on the other, so for that volume it
    is the only anchor the page map has.
    """

    section: int
    subsec: int = 0
    """0 when the head prints only the section."""


_SECTION_LOCATOR = re.compile(r"§\s*(\d{1,2})(?:\s*\.\s*(\d{1,2}))?", re.A)


def parse_section_locator(text: str) -> SectionLocator | None:
    """Find a section locator anywhere in text."""
    match = _SECTION_LOCATOR.search(text)
    if match is None:
        return None
    section = int(match.group(1))
    if section == 0:
        return None
    return SectionLocator(section, int(match.group(2)) if match.group(2) else 0)


def _first_line(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip()
    return ""


def _clip(s: str) -> str:
    return s if len(s) <= 60 else s[:60] + "…"


def reasons(problems: list[Problem]) -> str:
    """Join the problems into the one line that goes in the queue history."""
    return "; ".join(str(p) for p in problems)


def rules_of(problems: list[Problem]) -> list[Rule]:
    """The distinct rules that rejected a page, which is what a retry reads."""
    seen: set[Rule] = set()
    out: list[Rule] = []
    for problem in problems:
        if problem.rule not in seen:
            seen.add(problem.rule)
            out.append(problem.rule)
    return out
