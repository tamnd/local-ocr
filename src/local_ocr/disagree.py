"""Every disagreement gets adjudicated before it counts.

§05 makes this a rule rather than a nicety, and the reason is that tier B is not
perfect. `extract` has defects the corpus already knows about: page 448 of
`alg-viii-fr` is the standing example, a matrix whose entries carry scripts of
their own that defeated the grid assembler for months. A model that reads that
page correctly disagrees with the reference, and a harness that scores the
disagreement against the model has the sign backwards.

So this produces a work list and not a score. Each item is one of

    variation    the two readings say the same thing in different characters
    extraction   the reference is the one that is wrong
    model        the reading is the one that is wrong
    undecided    a person has to look

and the classifier's job is to make the undecided pile small, not to make it
empty. Three of the four verdicts are given automatically and the fourth is
given by whoever reads the residue. Nothing here ever decides "model" from
similarity alone, because that is the verdict with a cost attached: it is the
one that goes on a training curriculum.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from local_ocr import corpus as corpuslib
from local_ocr.evaluate import expect_from
from local_ocr.metrics import cdm, cer
from local_ocr.rules import textguard
from local_ocr.rules.validate import validate


class Kind(StrEnum):
    VARIATION = "variation"
    EXTRACTION = "extraction"
    MODEL = "model"
    UNDECIDED = "undecided"


@dataclass(frozen=True)
class Item:
    """One difference between two readings of one page."""

    page: str
    where: str
    """`prose line 12` or `span 3`, so the item can be found again."""
    reference: str
    read: str
    kind: Kind
    why: str

    def line(self) -> str:
        return f"{self.page} {self.where}: {self.kind}, {self.why}"


@dataclass
class Work:
    """The list, and what it adds up to."""

    items: list[Item] = field(default_factory=list)

    def of(self, kind: Kind) -> list[Item]:
        return [item for item in self.items if item.kind is kind]

    def counts(self) -> dict[str, int]:
        return {kind.value: len(self.of(kind)) for kind in Kind}

    def to_markdown(self) -> str:
        counts = self.counts()
        out = [
            "# Disagreements",
            "",
            ", ".join(f"{n} {name}" for name, n in counts.items()),
            "",
            "Variation and extraction items are classified automatically and need "
            "no reading. The undecided pile is the work.",
            "",
        ]
        for kind in (Kind.UNDECIDED, Kind.MODEL, Kind.EXTRACTION):
            items = self.of(kind)
            if not items:
                continue
            out.append(f"## {kind.value} ({len(items)})")
            out.append("")
            for item in items:
                out.append(f"### {item.page} {item.where}")
                out.append("")
                out.append(f"{item.why}")
                out.append("")
                out.append(f"reference: `{_clip(item.reference)}`")
                out.append("")
                out.append(f"read: `{_clip(item.read)}`")
                out.append("")
        return "\n".join(out) + "\n"


def _clip(text: str, width: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def classify(
    page: corpuslib.Page,
    read: str,
    *,
    drift: Iterable[str] = (),
) -> list[Item]:
    """Adjudicate one page as far as a machine honestly can.

    `drift` is the set of page ids that `extract drift` or `bourbaki audit` has
    already named. A disagreement on one of those pages is presumed to be the
    extraction's fault, because the corpus said so before this model existed and
    letting the model's output decide would be circular.
    """
    reference = textguard.normalise(page.body)
    reading = textguard.normalise(textguard.strip(read))
    known_bad = page.id in set(drift)

    # The reference is checked against the same eight rules the reading is. A
    # tier B page that fails them is an extraction that went wrong, and it is the
    # one automatic signal that points at the corpus rather than at the model.
    expect = expect_from(page)
    reference_broken = bool(validate(reference, expect))
    reading_broken = bool(validate(reading, expect))

    items: list[Item] = []
    for span in cdm.compare_pages(reference, reading).spans:
        if span.score == 1.0:
            continue
        where = f"span {span.index}"
        if span.score is None:
            items.append(
                Item(
                    page.id,
                    where,
                    span.reference,
                    span.read,
                    _verdict(known_bad, reference_broken, reading_broken),
                    span.note,
                )
            )
            continue
        if span.reference.strip() == span.read.strip():
            continue
        items.append(
            Item(
                page.id,
                where,
                span.reference,
                span.read,
                _verdict(known_bad, reference_broken, reading_broken),
                f"CDM {span.score:.3f}",
            )
        )

    items.extend(
        _prose_items(page.id, reference, reading, known_bad, reference_broken, reading_broken)
    )
    return items


def _verdict(known_bad: bool, reference_broken: bool, reading_broken: bool) -> Kind:
    """The three automatic verdicts, in the order they take precedence.

    A page the corpus has already flagged goes to extraction whatever else is
    true of it. Failing that, a reference that cannot pass the acceptance rules
    while the reading can is the corpus being wrong. Failing that, a reading that
    cannot pass them while the reference can is the model being wrong. Anything
    else is a person's job.
    """
    if known_bad:
        return Kind.EXTRACTION
    if reference_broken and not reading_broken:
        return Kind.EXTRACTION
    if reading_broken and not reference_broken:
        return Kind.MODEL
    return Kind.UNDECIDED


def _prose_items(
    page_id: str,
    reference: str,
    reading: str,
    known_bad: bool,
    reference_broken: bool,
    reading_broken: bool,
) -> list[Item]:
    """The prose, line by line, mathematics taken out first.

    Line by line and not character by character, because the output of this is
    read by a person and a list of two hundred single character edits is not
    something a person can act on. A line that changed is one item, and the item
    carries both versions of it.
    """
    left = _lines(reference)
    right = _lines(reading)
    out: list[Item] = []
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        was = "\n".join(left[i1:i2])
        now = "\n".join(right[j1:j2])
        if textguard.normalise_prose(was) == textguard.normalise_prose(now):
            # The same sentence written with different typography, which the
            # corpus normalises away before anybody reads it.
            out.append(
                Item(
                    page_id,
                    f"prose line {i1 + 1}",
                    was,
                    now,
                    Kind.VARIATION,
                    "identical after normalisation",
                )
            )
            continue
        out.append(
            Item(
                page_id,
                f"prose line {i1 + 1}",
                was,
                now,
                _verdict(known_bad, reference_broken, reading_broken),
                f"{tag}, {cer.distance(was, now)} edits",
            )
        )
    return out


def _lines(text: str) -> list[str]:
    return [line for line in cer.prose_of(text).split("\n") if line.strip()]
