"""The eight house rules, each measured on its own.

The prompt in `prompt/ocr_bourbaki.md` is fourteen hundred tokens of rules about
how these particular volumes are set, and §05 makes the point that every one of
them is machine checkable. That is the metric no leaderboard can give and this
corpus can, and the reason to report the eight separately rather than as one
number is that they are separately fixable. "The output looks wrong" is not a
work list. "The running head is right on 99 per cent of pages and the dangerous
bend on 40" is.

Two things about how applicability is decided.

A rule counts against a page only when the reference page gives it something to
do. A page with no footnote on it cannot get the footnote rule wrong, and
counting it as a pass would put the rate wherever the corpus happens to sit
rather than where the model does. So each rule says what makes it apply, and it
reads the reference to decide, never the reading being judged. A model cannot
raise its own score by leaving the hard thing out: dropping the dangerous bend
makes the rule apply and fail, not become irrelevant.

Every rule is written so that the reference obeys itself. A check that fails on
the reference is not measuring the model, it is measuring the corpus, and it
subtracts from the rate of every reader equally including a perfect one. So the
rules are comparisons and not absolutes: no more blackboard bold than the
reference has, no more wrapped lines, the same footnote marks. `tests/` asserts
this over the whole of `golden-dev` on the real corpus, and it caught two rules
that were absolute the first time this was run.

These run on the reading as it arrived, before `textguard.normalise` has been
near it. Two of the rules are ones the normaliser repairs on its way past, the
ring and the fenced star, so a check that ran after it can never fire on either.
That is not a theory: the first negative control run of this harness gave one
page in five the wrong ring on purpose and the rings rule reported 100 per cent.
`evaluate.judge` keeps the two texts apart for this reason.

Where a rule can be measured either by structure or by wording, it is measured
by structure. The heading check compares how many headings there are and at what
depth, and not what they say, because what they say is what CER measures and a
title with one letter misread should count once against the model rather than
twice.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from local_ocr.rules import textguard
from local_ocr.rules.validate import ILLEGIBLE, MAX_ILLEGIBLE, parse_page_label

_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$", re.M | re.A)

# Both languages, because three of the six tier B volumes are French and a rule
# that only knows the English word would report those volumes as having no
# statement heads at all rather than as failing on them.
_HEAD_WORDS = (
    "Definition",
    "Proposition",
    "Theorem",
    "Lemma",
    "Corollary",
    "Remark",
    "Scholium",
    "Example",
    "Théorème",
    "Lemme",
    "Corollaire",
    "Remarque",
    "Scholie",
    "Exemple",
    "Définition",
)
_WORDS = "|".join(sorted(set(_HEAD_WORDS), key=len, reverse=True))

# Bold, then the spaced em rule, which is how the volumes print it.
_BOLD_HEAD = re.compile(rf"^\*\*({_WORDS})\b[^*]*\*\*\s*—", re.M)

# The same head with nothing around it, which is what a model that ignored the
# rule writes. Anchored at the start of a line and followed by a number or the
# rule, so an ordinary sentence beginning with the word Remark does not count.
_PLAIN_HEAD = re.compile(rf"^({_WORDS})\b\s*(\d+\s*)?\.\s*(—|-|–)", re.M)  # noqa: RUF001

_BEND = "☡"

# \mathbb in any form, which is the thing the rule forbids outright. The bare
# form without braces is here because textguard rewrites it and a reading judged
# before normalisation still has it.
_BLACKBOARD = re.compile(r"\\mathbb\b", re.A)
_RING = re.compile(r"\\math(?:bf|bb)\s*\{?\s*[ZQRCNFP]", re.A)

# What a model writes when it decides to admit defeat in its own words instead
# of the corpus's. Each of these has been seen in the fleet logs.
_OTHER_PLACEHOLDERS = (
    "[illegible]",
    "[unreadable]",
    "[unclear]",
    "[illisible]",
    "(illegible)",
    "[...]",
    "[text]",
    "???",
)

_FOOTNOTE_MARK = re.compile(r"\[\^(\w+)\]", re.A)
_FOOTNOTE_NOTE = re.compile(r"^\[\^(\w+)\]:", re.M | re.A)


def headings(text: str) -> list[tuple[int, str]]:
    return [(len(m.group(1)), m.group(2).strip()) for m in _HEADING.finditer(text)]


def wrapped_lines(text: str) -> list[int]:
    """Lines that look like a printed paragraph broken at the column width.

    A paragraph is one line in this corpus, so a line that stops without
    finishing its sentence and is followed by a line carrying on in lower case
    is a line the model wrapped. The length floor keeps a genuine short line, a
    heading or a list item from being read as a wrap.
    """
    lines = text.split("\n")
    out: list[int] = []
    for i, line in enumerate(lines[:-1]):
        body = line.rstrip()
        nxt = lines[i + 1].lstrip()
        if len(body) < 40 or body.startswith("#") or body.endswith("$$"):
            continue
        if not nxt or not nxt[:1].islower():
            continue
        if body.endswith((".", "!", "?", ":", ";", "$")):
            continue
        out.append(i + 1)
    return out


def footnotes(text: str) -> tuple[set[str], set[str]]:
    """The marks used in the body and the notes defined at the foot."""
    notes = set(_FOOTNOTE_NOTE.findall(text))
    marks = set(_FOOTNOTE_MARK.findall(text)) - notes
    return marks, notes


def _bends_alone(text: str) -> bool:
    return all(line.strip() == _BEND for line in text.split("\n") if _BEND in line)


@dataclass(frozen=True)
class Check:
    """One house rule, when it applies and whether it was obeyed."""

    name: str
    what: str
    applies: Callable[[str], bool]
    """Reads the reference page only. See the module docstring."""
    obeyed: Callable[[str, str], bool]


CHECKS: tuple[Check, ...] = (
    Check(
        "headings",
        "Headings carry hashes, because the structure of the Book is read out of them.",
        lambda ref: bool(headings(ref)),
        lambda ref, read: [d for d, _ in headings(ref)] == [d for d, _ in headings(read)],
    ),
    Check(
        "paragraphs",
        "A printed paragraph is one line, whatever the column width did to it.",
        # A page with no long line has no paragraph on it to break, so the rule
        # has nothing to say about it either way.
        lambda ref: any(len(line) >= 200 for line in ref.split("\n")),
        lambda ref, read: len(wrapped_lines(read)) <= len(wrapped_lines(ref)),
    ),
    Check(
        "running head",
        "The first line is the running head, because the page map depends on it.",
        lambda ref: parse_page_label(_first(ref)) is not None,
        lambda ref, read: parse_page_label(_first(read)) == parse_page_label(_first(ref)),
    ),
    Check(
        "rings",
        "The rings are \\mathbf{Z} and never \\mathbb{Z}.",
        lambda ref: bool(_RING.search(ref)),
        lambda ref, read: len(_BLACKBOARD.findall(read)) <= len(_BLACKBOARD.findall(ref)),
    ),
    Check(
        "statement heads",
        "A statement head is bold and keeps the spaced em rule.",
        lambda ref: bool(_BOLD_HEAD.search(ref)),
        lambda ref, read: (
            len(_BOLD_HEAD.findall(read)) >= len(_BOLD_HEAD.findall(ref))
            and len(_PLAIN_HEAD.findall(read)) <= len(_PLAIN_HEAD.findall(ref))
        ),
    ),
    Check(
        "dangerous bend",
        "The dangerous bend is the sign alone on its line.",
        lambda ref: _BEND in ref,
        lambda ref, read: (
            read.count(_BEND) == ref.count(_BEND) and (_bends_alone(read) or not _bends_alone(ref))
        ),
    ),
    Check(
        "forward references",
        "A forward reference passage is fenced with \\* and never a bare asterisk.",
        lambda ref: textguard.STAR in ref,
        # stars() returns what it had to change. No more changes than the
        # reference needed means the reading wrote the fence the way the rule
        # asks for it.
        lambda ref, read: (
            read.count(textguard.STAR) == ref.count(textguard.STAR)
            and textguard.stars(read)[1] <= textguard.stars(ref)[1]
        ),
    ),
    Check(
        "footnotes",
        "Footnotes are Markdown footnotes, and every mark has its note.",
        lambda ref: bool(_FOOTNOTE_MARK.search(ref)),
        lambda ref, read: _footnotes_match(ref, read),
    ),
    Check(
        "illegible",
        "What cannot be read is the illegible mark and never a guess.",
        # Applies everywhere, because the failure it catches is a model
        # inventing its own way of saying it could not read something, and that
        # can happen on any page.
        lambda ref: True,
        lambda ref, read: _illegible_ok(ref, read),
    ),
)


def _first(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip()
    return ""


def _footnotes_match(reference: str, read: str) -> bool:
    """The same marks, and no mark left without its note that the reference has.

    The second half is against the reference rather than absolute because a note
    can be printed on the facing page. The corpus has pages that carry a mark
    whose text is somewhere else, and a reading that reproduces that faithfully
    has done the right thing.
    """
    ref_marks, ref_notes = footnotes(reference)
    marks, notes = footnotes(read)
    return marks == ref_marks and notes == ref_notes


def _illegible_ok(reference: str, read: str) -> bool:
    """Half of the rule, and the half a machine can see.

    A machine can check that the model admitted defeat in the corpus's words
    rather than inventing its own, and that it did not admit defeat so often
    that the page is a stub. It cannot check the other half, that the model said
    so instead of guessing, because a guess looks exactly like a reading. That
    half is what CDM against a second reader is for, and it is why §05 ends
    where it does.
    """
    lowered = read.lower()
    if any(word in lowered for word in _OTHER_PLACEHOLDERS):
        return False
    return read.count(ILLEGIBLE) <= max(MAX_ILLEGIBLE, reference.count(ILLEGIBLE))


@dataclass
class Count:
    name: str
    what: str
    applicable: int = 0
    obeyed: int = 0

    @property
    def rate(self) -> float | None:
        """None, and not 1.0, when no page gave the rule anything to do.

        A rule that never applied has no rate, and printing 100 per cent for it
        would read as evidence about the model when it is evidence about the
        sample.
        """
        if self.applicable == 0:
            return None
        return self.obeyed / self.applicable

    def line(self) -> str:
        if self.rate is None:
            return f"{self.name}: did not apply on any page"
        return f"{self.name}: {self.rate:.1%} of {self.applicable} pages"


@dataclass
class Conformance:
    """The eight rates, accumulated over a set of pages."""

    counts: dict[str, Count] = field(default_factory=dict)
    pages: int = 0

    def observe(self, reference: str, read: str) -> list[str]:
        """Judge one page, and return the rules it broke."""
        self.pages += 1
        broke: list[str] = []
        for check in CHECKS:
            count = self.counts.setdefault(check.name, Count(check.name, check.what))
            if not check.applies(reference):
                continue
            count.applicable += 1
            if check.obeyed(reference, read):
                count.obeyed += 1
            else:
                broke.append(check.name)
        return broke

    def rows(self) -> list[Count]:
        """In the order the rules are written, which is the order §05 lists them."""
        return [self.counts[check.name] for check in CHECKS if check.name in self.counts]
