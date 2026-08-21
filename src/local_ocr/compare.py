"""What two readers disagree about, in a form somebody can act on.

M6 puts a second reader behind the first, and the whole value of that turns on
one question: what does a disagreement between two models tell you that the
eight acceptance rules do not already tell you.

The rules are good at the failures that leave a mark. A reading with an unclosed
dollar, no running head, three hundred characters where the page has three
thousand, an inline span that runs half a page: all of those are caught without
a second opinion, and M5 measured them catching six per cent of a real volume.

They are blind to the failure that matters most here, which is a reading that is
well formed and says something else. `\\aleph_0` read as `\\aleph_1` passes every
rule. So does a lost subscript, a `\\subset` read as `\\subseteq`, an exercise
numbered 7 that the page numbers 9. Those are the errors that get into the
corpus and stay there, and the only cheap detector for them is a second reader
that was trained by different people on different data and therefore does not
make the same mistake in the same place.

That last clause is why the referee is a different base family and not a second
sample from the same weights. Two Qwen2.5-VL derivatives agree on their shared
inheritance, including where it is wrong, and an agreement between them is worth
much less than it looks.

## What is compared, and why in this order

Structure first, then formulas, then prose, because that is increasing cost and
decreasing severity.

A structural difference is nearly free to find and is the most likely to be a
real defect: if one reader sees four headings and the other sees three, one of
them has swallowed a heading into a paragraph, and no character metric will say
so clearly. Exercise numbering is in here for the same reason, and it is the one
structural signal in this corpus with an external check available, because
exercise numbers are consecutive and a gap is a fact rather than an opinion.

Formulas next, by CDM, which compares what a formula prints rather than how it
is spelled, so `\\frac` against `\\dfrac` is not a disagreement and a lost
subscript is.

Prose last, sentence by sentence, because prose disagreements are mostly
typography and mostly not worth a person's time, and rolling them in earlier
would bury the two categories that are. Sentences rather than lines is the one
thing here that was got wrong first and then fixed, and the reason is in
`prose` below: comparing physical lines measures where each reader chose to
wrap, which is not a fact about the page.

## What this module does not do

It does not decide who is right. Everything here produces a `Difference` with
both readings in it and a severity, and the adjudicator in `second.py` is what
spends money to resolve one. Keeping those apart matters because the comparison
is deterministic and free and the adjudication is neither, so the comparison can
be tested exhaustively on fixtures and the adjudication cannot.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import StrEnum

from local_ocr.metrics import cdm, cer
from local_ocr.rules import textguard
from local_ocr.rules.mathtex import split
from local_ocr.rules.validate import looks_like_head

# A CDM score below this is a formula the two readers do not agree prints the
# same thing. 0.99 and not 1.0 because the metric lays glyphs out in floating
# point and two renderings of the same formula can differ in the last place.
# §05 uses the same threshold for the same reason.
AGREE = 0.99

# Prose sentences closer than this after normalisation are the same sentence
# with different typography. Expressed as a fraction of the longer one, so that
# one wrong character in a ten character sentence is a disagreement and one
# wrong character in a two hundred character sentence is not.
PROSE_SLACK = 0.02


class Where(StrEnum):
    """Which of the three comparisons produced a difference."""

    STRUCTURE = "structure"
    FORMULA = "formula"
    PROSE = "prose"


class Severity(StrEnum):
    """How much a difference is worth spending on.

    Three levels and not a score, because the adjudicator spends in whole
    units: it either sends a crop or it does not. A number would invite a
    threshold, and a threshold here would be tuned on whichever volume was
    being read that week.
    """

    HIGH = "high"
    """A reader lost or invented something. Worth a round trip on its own."""

    MEDIUM = "medium"
    """The two readings differ in what they say. Worth a round trip in turn."""

    LOW = "low"
    """Typography. Worth recording and not worth asking about."""


@dataclass(frozen=True)
class Difference:
    """One thing the two readers do not agree about."""

    where: Where
    what: str
    """`heading count`, `span 3`, `prose sentence 12`. Enough to find it again."""
    first: str
    second: str
    severity: Severity
    why: str
    score: float | None = None
    """The CDM score, for a formula difference, and None otherwise."""

    def line(self) -> str:
        return f"{self.where} {self.what}: {self.severity}, {self.why}"


@dataclass(frozen=True)
class Structure:
    """The shape of a reading, with the words taken out.

    Everything here is countable and none of it depends on the two readers
    having spelled anything the same way, which is what makes a structural
    difference cheap to trust. Two readings of the same page have the same
    number of paragraphs whatever they disagree about inside them, unless one of
    them has genuinely lost or merged something.
    """

    head: str
    headings: tuple[str, ...]
    paragraphs: int
    exercises: tuple[int, ...]
    displays: int
    inlines: int


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(\S.*?)\s*$", re.M)

# An exercise opens a paragraph with its number and a stop. Anchored to the
# start of a line because a bare "7." mid paragraph is a section reference and
# not an exercise, and the corpus is full of those.
_EXERCISE = re.compile(r"^\s{0,3}(\d{1,3})\s*[.)]\s+\S", re.M)


def structure(text: str) -> Structure:
    """Read the shape off a reading.

    Runs on the normalised body, so that a reader which writes curly quotes and
    one which writes straight ones are not recorded as differing in structure.
    """
    body = textguard.normalise(textguard.strip(text))
    lines = body.splitlines()
    head = ""
    for line in lines:
        if line.strip():
            head = line.strip() if looks_like_head(line.strip()) else ""
            break
    spans, _ = split(body)
    return Structure(
        head=head,
        headings=tuple(m.group(1) for m in _HEADING.finditer(body)),
        paragraphs=sum(1 for block in re.split(r"\n\s*\n", body) if block.strip()),
        exercises=tuple(int(m.group(1)) for m in _EXERCISE.finditer(body)),
        displays=sum(1 for s in spans if s.display),
        inlines=sum(1 for s in spans if not s.display),
    )


def gaps(numbers: tuple[int, ...]) -> list[int]:
    """The exercise numbers a run of them skips.

    Exercises are consecutive in this corpus, which is the one place a single
    reading can be checked against something other than another reading. A
    reading that goes 5, 6, 8 has lost exercise 7 or misread its number, and
    that is knowable without a referee at all.

    Only the ascending run from the first number is considered. A page whose
    exercise list restarts, which happens where a chapter's exercises are split
    by section, would otherwise report every number of the second run as a gap.
    """
    if len(numbers) < 2:
        return []
    run = [numbers[0]]
    for value in numbers[1:]:
        if value <= run[-1]:
            break
        run.append(value)
    missing = []
    for expected in range(run[0], run[-1]):
        if expected not in run:
            missing.append(expected)
    return missing


def structural(first: str, second: str) -> list[Difference]:
    """Compare the shapes of two readings.

    The severities are not uniform and the reason is that the four signals are
    not equally trustworthy. A disagreement about the running head or about how
    many headings there are means one reader lost a line of the page, which is
    always a defect. A disagreement about paragraph count is usually one reader
    breaking a paragraph where the other did not, which is a defect nobody
    should pay a round trip for. So heads and headings are high, exercise
    numbering is high because it can be checked, and paragraphs are low.
    """
    left, right = structure(first), structure(second)
    out: list[Difference] = []

    if left.head != right.head:
        # One of the two is empty in nearly every case, which is the head pass
        # having fired on one reading and not the other. That is worth knowing
        # about: it means the two readers do not agree the page prints a head.
        out.append(
            Difference(
                Where.STRUCTURE,
                "running head",
                left.head,
                right.head,
                Severity.HIGH,
                "the two readings do not print the same running head",
            )
        )

    if len(left.headings) != len(right.headings):
        out.append(
            Difference(
                Where.STRUCTURE,
                "heading count",
                str(len(left.headings)),
                str(len(right.headings)),
                Severity.HIGH,
                f"{len(left.headings)} headings against {len(right.headings)}, "
                "so one reading has swallowed a heading or invented one",
            )
        )
    else:
        for index, (one, other) in enumerate(zip(left.headings, right.headings, strict=True)):
            if textguard.normalise_prose(one) == textguard.normalise_prose(other):
                continue
            out.append(
                Difference(
                    Where.STRUCTURE,
                    f"heading {index + 1}",
                    one,
                    other,
                    Severity.MEDIUM,
                    "the same heading read two ways",
                )
            )

    if left.exercises != right.exercises:
        out.append(
            Difference(
                Where.STRUCTURE,
                "exercise numbering",
                ", ".join(str(n) for n in left.exercises),
                ", ".join(str(n) for n in right.exercises),
                Severity.HIGH,
                "the two readings number the exercises differently",
            )
        )
    for name, numbers in (("first", left.exercises), ("second", right.exercises)):
        missing = gaps(numbers)
        if missing:
            out.append(
                Difference(
                    Where.STRUCTURE,
                    "exercise gap",
                    ", ".join(str(n) for n in left.exercises),
                    ", ".join(str(n) for n in right.exercises),
                    Severity.HIGH,
                    f"the {name} reading skips exercise " + ", ".join(str(n) for n in missing),
                )
            )

    if (left.displays, left.inlines) != (right.displays, right.inlines):
        out.append(
            Difference(
                Where.STRUCTURE,
                "span count",
                f"{left.displays} display, {left.inlines} inline",
                f"{right.displays} display, {right.inlines} inline",
                Severity.HIGH,
                "the two readings do not find the same mathematics on the page",
            )
        )

    if left.paragraphs != right.paragraphs:
        out.append(
            Difference(
                Where.STRUCTURE,
                "paragraph count",
                str(left.paragraphs),
                str(right.paragraphs),
                Severity.LOW,
                f"{left.paragraphs} paragraphs against {right.paragraphs}",
            )
        )
    return out


def formulas(first: str, second: str) -> list[Difference]:
    """Compare the mathematics of two readings, span by span, by CDM.

    Pairing is by position, which `compare_pages` does and which is the right
    thing here for the reason given there: pairing by similarity would match
    every formula to whichever one it most resembles and never report a lost
    display. When the two readings disagree about how many spans there are, the
    structural comparison has already said so at high severity, and the unpaired
    spans below carry the detail.
    """
    body_a = textguard.normalise(textguard.strip(first))
    body_b = textguard.normalise(textguard.strip(second))
    out: list[Difference] = []
    for span in cdm.compare_pages(body_a, body_b).spans:
        what = f"span {span.index}"
        if span.score is None:
            if span.note.startswith("unpaired"):
                out.append(
                    Difference(
                        Where.FORMULA,
                        what,
                        span.reference,
                        span.read,
                        Severity.HIGH,
                        "one reading has this formula and the other does not",
                    )
                )
                continue
            if span.note.startswith("one side"):
                # One reading produced LaTeX the layout engine can set and the
                # other did not. That is not proof of an error, because mathtext
                # is a subset of LaTeX and Bourbaki uses things outside it, but
                # it is worth a look: the commonest cause is one reader having
                # invented a macro.
                out.append(
                    Difference(
                        Where.FORMULA,
                        what,
                        span.reference,
                        span.read,
                        Severity.MEDIUM,
                        span.note,
                    )
                )
                continue
            # Neither side renders, so there is nothing to compare and no
            # evidence either way. Recorded at low severity so it appears in the
            # sidecar and costs nothing.
            if span.reference.strip() != span.read.strip():
                out.append(
                    Difference(
                        Where.FORMULA,
                        what,
                        span.reference,
                        span.read,
                        Severity.LOW,
                        span.note,
                    )
                )
            continue
        if span.score >= AGREE:
            continue
        out.append(
            Difference(
                Where.FORMULA,
                what,
                span.reference,
                span.read,
                Severity.HIGH if span.score < 0.9 else Severity.MEDIUM,
                f"CDM {span.score:.3f}",
                score=span.score,
            )
        )
    return out


def prose(first: str, second: str) -> list[Difference]:
    """Compare the words of two readings, sentence by sentence, mathematics out.

    Sentences and not physical lines, and this was worth getting wrong once to
    learn. Comparing physical lines makes the comparison depend on where each
    reader chose to wrap, and two readers never choose the same places: one
    keeps the scan's line breaks, another reflows the paragraph, a third breaks
    where the display interrupted the sentence. Diffing those produces a page of
    differences that are all the same sentence, and worse, it produces them at
    medium severity, so the adjudicator spends its whole budget on typography
    before it reaches the formula that changed.

    Where the paragraphs break is a real difference and it is already reported,
    by `structural`, at low severity, which is what it is worth. So this can
    flatten the whitespace entirely and lose nothing.

    Sentences and not the whole prose as one string, because the output is read
    by a person and by the adjudicator, and both need a unit small enough to
    point at. A sentence is that unit.

    Severity is low unless the two sentences are far enough apart to be saying
    different things, and the cut is proportional: one wrong character in ten is
    a different word, one wrong character in two hundred is a hyphen.
    `PROSE_SLACK` is that fraction.
    """
    left = sentences(textguard.normalise(textguard.strip(first)))
    right = sentences(textguard.normalise(textguard.strip(second)))
    out: list[Difference] = []
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        was = " ".join(left[i1:i2])
        now = " ".join(right[j1:j2])
        if textguard.normalise_prose(was) == textguard.normalise_prose(now):
            continue
        edits = cer.distance(was, now)
        longer = max(len(was), len(now), 1)
        severity = Severity.LOW if edits <= longer * PROSE_SLACK else Severity.MEDIUM
        out.append(
            Difference(
                Where.PROSE,
                f"prose sentence {i1 + 1}",
                was,
                now,
                severity,
                f"{tag}, {edits} edits over {longer} characters",
            )
        )
    return out


# A sentence ends at a stop, a question mark or a colon followed by space and a
# capital or a digit. The colon is in there because Bourbaki uses it to open a
# definition as often as a full stop, and the capital is what keeps "Ch. I" and
# "cf. Exercise 3" from being cut in half. It is not a linguistic sentence
# splitter and does not need to be: what it has to do is produce the same units
# from two readings of the same page, and it does that as long as it is
# deterministic and does not depend on line breaks.
_SENTENCE = re.compile(r"(?<=[.?:])\s+(?=[A-Z0-9(\\])")


def sentences(text: str) -> list[str]:
    """The prose of a reading, flattened and cut into sentences."""
    flat = " ".join(cer.prose_of(text).split())
    return [part.strip() for part in _SENTENCE.split(flat) if part.strip()]


@dataclass
class Comparison:
    """Everything two readings of one page disagree about."""

    differences: list[Difference] = field(default_factory=list)

    @property
    def agreed(self) -> bool:
        """Whether anything worth a round trip came up.

        Low severity differences do not count against agreement. Two readings
        that differ only in where the paragraphs break and how the quotes curl
        are the same reading, and treating them as a disagreement would send
        every page in the corpus to the adjudicator.
        """
        return not self.worth_asking

    @property
    def worth_asking(self) -> list[Difference]:
        """The differences an adjudicator should spend on, worst first."""
        order = {Severity.HIGH: 0, Severity.MEDIUM: 1}
        got = [d for d in self.differences if d.severity is not Severity.LOW]
        return sorted(got, key=lambda d: (order[d.severity], d.where, d.what))

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for d in self.differences:
            out[d.severity.value] += 1
        return out


def compare(first: str, second: str) -> Comparison:
    """Two readings of one page in, everything they disagree about out.

    Structure, then formulas, then prose, which is the order they are worth
    reading in and, not by coincidence, increasing cost to compute.
    """
    out = Comparison()
    out.differences.extend(structural(first, second))
    out.differences.extend(formulas(first, second))
    out.differences.extend(prose(first, second))
    return out
