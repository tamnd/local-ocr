"""What the prompt did not fix, and whether that is convention or capability.

    local-ocr residual --set golden-dev --readings base=out/base --readings head3=out/head3

Section 08 will not let a fine tune start on a hunch. Its precondition is one
sentence: a `golden-dev` report exists, the house rule conformance rates are
broken out individually, and the residual failures are characterised. This is
that characterisation, and the reason it is a module and not an afternoon in a
shell is that the answer decides what a LoRA is trained on.

## Why two readings and not one

A single report says what is wrong. It cannot say what is fixable by asking,
which is the only question section 08 needs answered, because a failure the
prompt can still move is a failure that does not need a GPU.

So this takes the prompt revisions and reports the spread. reader-a on
golden-dev has four of them, and the spread separates them cleanly:

  running head    4.8% -> 73.8% -> 76.8% -> 80.4%
  acceptance     11.5% -> 78.0% -> 80.5% -> 83.5%
  headings        0.0% -> 0.0%  -> 0.0%  -> 0.0%
  footnotes       0.0% -> 0.0%  -> 0.0%  -> 0.0%
  prose CER       7.63% -> 7.58% -> 7.52% -> 7.68%
  formula mean   0.7847 -> 0.7814 -> 0.7811 -> 0.7841

Three prompt revisions bought seventy five points of running head and seventy
two points of acceptance, and bought nothing at all anywhere else. That is not a
prompt that needs a fourth revision. That is a channel that has been used up on
the rules it can reach.

## Convention or capability, decided by a test and not by an opinion

Section 08 draws the line and this draws it the same way, mechanically, one
failing page at a time.

**Convention** is the reading holding the content and not the corpus's markup
for it. The model read the words and wrote them its own way. `Définition 1. —`
where the corpus writes `**Définition 1.**`, `(1)` where the corpus writes
`[^1]`, a line of running prose where the corpus writes `### 1. Modules
semi-simples`. Nothing was lost off the page. This is what a style LoRA is for
and it is the cheap half of section 08.

**Capability** is the content not being there. The six point page number at the
outer margin that came back as nothing, the display that has no counterpart on
the other side of the alignment. No amount of instruction recovers a thing the
reader did not see, and section 08 says to attempt this only if the style LoRA
lands first.

The test per rule is whichever one can be written down. For headings and
statement heads it is whether the head's own text survives somewhere in the
reading. For the running head it is whether the words survive without the page
label, which is the margin failure, or whether the whole line is gone. For
footnotes it is whether the note text is there under some other mark. Where no
such test exists the rule is reported without an attribution rather than
guessed at, and `forward references` has never applied to a golden-dev page at
all.

## What it found, and the one number that is not a rule

reader-a's fourth prompt revision on golden-dev, 200 pages:

  rule              obeyed        failures     convention  capability
  headings            0 of 60       60             56          4
  statement heads    34 of 84       50             50          0
  running head      135 of 168      33              7         26
  paragraphs        170 of 184      14              conventions by construction
  footnotes           0 of 7         7              5          2
  rings              83 of 84        1              conventions by construction
  dangerous bend      0 of 1         1              conventions by construction
  illegible         200 of 200       0

118 convention against 32 capability, over the rules where the two can be told
apart. That is the answer section 08 wanted, and it says a style LoRA is the
right instrument and the capability LoRA is not needed yet. Headings and
statement heads carry 106 of the 118 on their own, so the answer does not turn
on any of the smaller rows.

Not one of the 200 readings contains a single Markdown heading, and 60 of those
pages have one to write. That is the largest single convention failure in the
corpus and it is invisible in every character rate, because a hash is one
character. It is also load bearing: the structure of the Book is read out of the
headings, and the corpus is currently being built without it.

The running head's largest bucket is not something the prompt can reach, though
the first version of this module said it was. 17 of its 33 failures are pages
where the head's own words are nowhere in the reading at all, and the head pass
runs on every page by default, so those are pages a second dedicated look at the
top strip also came back empty on. 9 more dropped the page label off the end of
an otherwise correct head, which is the six point figure in the outer margin and
is not going to be talked into existence. Only 2 are the head present but out of
place, which is the one shape the head pass is built to repair.

The formula number is the one line here that is not a house rule, and it is
counted separately for a reason. 2730 of golden-dev's 10730 spans are on one
side of the alignment and not the other. That is not a formula written in the
wrong notation and it is not a formula read badly. It is a formula that is not
there, or one the reader invented, and it is the largest capability failure on
the set.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_ocr import corpus as corpuslib
from local_ocr import evaluate
from local_ocr.metrics import cdm, conformance
from local_ocr.rules import textguard
from local_ocr.rules.validate import parse_page_label

CONVENTION = "convention"
CAPABILITY = "capability"

_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_NOTE = re.compile(r"^\[\^\w+\]:\s*(.+)$", re.M)
_LABEL_TAIL = re.compile(r"\s*[A-Z]+\s+[IVXLCDM]+\.\d+\s*$")
_WORD = re.compile(r"\w+", re.UNICODE)


def _bare(text: str) -> str:
    """The text with everything that is markup or spacing taken out.

    Both sides of every attribution test go through here, because the whole
    question is whether the words survived, and a word is no less present for
    having lost its asterisks or gained a line break.
    """
    return "".join(_WORD.findall(text)).casefold()


def _survived(needle: str, haystack: str) -> bool:
    """Whether a head opens a line of the reading, markup aside.

    A head is a thing that starts something, and that is true of it whether or
    not the reader marked it up. `**Lemme 5.**` unbolded is still the first
    thing on its line. So the test is that the head opens a line and not that
    it appears anywhere, and the difference is not cosmetic, because a short
    head is a short string and a short string turns up inside a page of prose
    for reasons that have nothing to do with the page.

    The three candidates, over golden-dev's 235 heads, each asked once about
    the page it belongs to and once about a page drawn at random:

      anywhere in the page              231 of 235 found, 8.51 % false
      anywhere, and at least 8 folded   191 of 235 found, 2.98 % false
      opens a line                      220 of 235 found, 2.55 % false

    The length floor was the first attempt and it is the worst of the three. It
    buys its way out of coincidence by refusing to answer about short heads at
    all, and golden-dev has 50 heads that fold to under eight characters, so it
    was calling forty real ones missing to avoid seven false ones. Opening a
    line is both the more accurate test and the one with no constant in it.
    """
    key = _bare(needle)
    if not key:
        return False
    return any(_bare(line).startswith(key) for line in haystack.split("\n"))


def _present(needle: str, haystack: str) -> bool:
    """Whether a passage is anywhere in the reading, markup aside.

    For footnotes rather than heads, because a note is not a thing that opens a
    line. A reader that kept the note may have set it at the foot on its own
    line or run it into the body, and both mean the words survived. Coincidence
    is not a worry here the way it is for heads, since the passage compared is
    up to sixty folded characters of a sentence.
    """
    key = _bare(needle)
    return bool(key) and key in _bare(haystack)


@dataclass
class Verdict:
    """One house rule, over one set of readings, and what its failures were."""

    rule: str
    applicable: int
    obeyed: int
    convention: int = 0
    capability: int = 0
    unattributed: int = 0
    notes: dict[str, int] = field(default_factory=dict)

    @property
    def failures(self) -> int:
        return self.applicable - self.obeyed

    @property
    def rate(self) -> float | None:
        return None if not self.applicable else self.obeyed / self.applicable


def _attribute_headings(ref: str, read: str) -> tuple[str, str]:
    """A heading is convention when its words are in the reading without the hashes."""
    wanted = [title for _, title in conformance.headings(ref)]
    if wanted and all(_survived(title, read) for title in wanted):
        return CONVENTION, "the heading is there as running text, without its hashes"
    if any(_survived(title, read) for title in wanted):
        return CONVENTION, "some of the headings are there as running text"
    return CAPABILITY, "the heading's own words are not in the reading"


def _attribute_statement_heads(ref: str, read: str) -> tuple[str, str]:
    wanted = _BOLD.findall(ref)
    if wanted and all(_survived(head, read) for head in wanted):
        return CONVENTION, "the head is there unbolded"
    if any(_survived(head, read) for head in wanted):
        return CONVENTION, "some of the heads are there unbolded"
    return CAPABILITY, "the head's own words are not in the reading"


def _attribute_running_head(ref: str, read: str) -> tuple[str, str]:
    """Four failures wearing one name, and they do not share a fix.

    The page label is set at the outer margin in type several sizes down from
    the body, so a reading that kept the words of the head and dropped `A
    VIII.138` off the end did not misunderstand an instruction. It did not see
    the number. That is the same failure the Kvant rotated set measures and it
    is the one entry here that a prompt was never going to reach.

    A head that is a bare label with no words is a convention. There is nothing
    on that line to make it look like a heading, so the reader starts at the
    section title below it instead, and being told is enough to fix that.

    A head somewhere in the reading but not on its first line is a convention
    too, and it is the one with a fix that is neither prompt nor weights: the
    head pass exists to put it back and did not.

    A head with no trace of it anywhere is capability, and the first version of
    this function got that wrong. It called the case `the head is not the first
    line of the reading` without ever checking whether the head was on any line,
    which put 19 golden-dev pages in a bucket named for a defect they do not
    have. They are pages the reader never transcribed the top strip of at all.
    """
    # conformance_reference is the one text with the running head put back, so
    # its first line is the head and no lookup is needed to find it.
    words = _LABEL_TAIL.sub("", _first(ref)).strip()
    got = _first(read)
    if parse_page_label(got) is not None:
        return CONVENTION, "a page label, but not this page's"
    if words and _bare(got) == _bare(words):
        return CAPABILITY, "the words of the head, without the page label set in the margin"
    if not words:
        return CONVENTION, "the head is a bare label and the reading started at the section title"
    if _survived(words, read):
        return CONVENTION, "the head is in the reading but not on its first line"
    return CAPABILITY, "the head's own words are nowhere in the reading"


def _attribute_footnotes(ref: str, read: str) -> tuple[str, str]:
    """A footnote is convention when the note is there under some other mark."""
    notes = _NOTE.findall(ref)
    if notes and all(_present(note[:60], read) for note in notes):
        return CONVENTION, "the note is there, under the printed mark rather than a Markdown one"
    if any(_present(note[:60], read) for note in notes):
        return CONVENTION, "some of the notes are there under the printed mark"
    return CAPABILITY, "the note's own words are not in the reading"


ATTRIBUTORS = {
    "headings": _attribute_headings,
    "statement heads": _attribute_statement_heads,
    "running head": _attribute_running_head,
    "footnotes": _attribute_footnotes,
}
"""The rules a failure can be attributed for, and nothing else is guessed at.

`paragraphs`, `rings`, `illegible` and `dangerous bend` are absent on purpose.
All four are conventions by construction, since each is about how something
already on the page is written down, so an attribution column for them would be
a constant dressed up as a measurement.
"""


def _first(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line.strip()
    return ""


@dataclass
class Formulas:
    """The formula side, which is one number and not a rule."""

    scored: int = 0
    exact: int = 0
    unpaired: int = 0
    total: int = 0
    mean: float = 0.0


@dataclass
class Reading:
    """One prompt revision, judged."""

    name: str
    pages: int = 0
    verdicts: list[Verdict] = field(default_factory=list)
    formulas: Formulas = field(default_factory=Formulas)

    def find(self, rule: str) -> Verdict | None:
        return next((v for v in self.verdicts if v.rule == rule), None)


def judge(name: str, pages: Sequence[corpuslib.Page], readings: Path) -> Reading:
    """Every house rule over one directory of readings, with failures attributed."""
    out = Reading(name)
    order = [check.name for check in conformance.CHECKS]
    verdicts = {rule: Verdict(rule, 0, 0) for rule in order}
    scores: list[float] = []
    for page in pages:
        found = evaluate.find_reading(readings, page.id)
        if found is None:
            continue
        out.pages += 1
        ref = evaluate.conformance_reference(page)
        text = found.read_text()
        for check in conformance.CHECKS:
            if not check.applies(ref):
                continue
            verdict = verdicts[check.name]
            verdict.applicable += 1
            if check.obeyed(ref, text):
                verdict.obeyed += 1
                continue
            attribute = ATTRIBUTORS.get(check.name)
            if attribute is None:
                verdict.unattributed += 1
                continue
            which, note = attribute(ref, text)
            setattr(verdict, which, getattr(verdict, which) + 1)
            verdict.notes[note] = verdict.notes.get(note, 0) + 1

        body = evaluate.without_head(page, textguard.normalise(textguard.strip(text)))
        report = cdm.compare_pages(textguard.normalise(page.body), body)
        out.formulas.total += len(report.spans)
        out.formulas.unpaired += report.unpaired
        for span in report.scored:
            scores.append(span.score or 0.0)
            out.formulas.exact += (span.score or 0.0) >= 0.99
    out.formulas.scored = len(scores)
    out.formulas.mean = sum(scores) / len(scores) if scores else 0.0
    out.verdicts = [verdicts[rule] for rule in order]
    return out


def moved(readings: Sequence[Reading], rule: str) -> float | None:
    """How far the prompt moved one rule, over every revision given.

    The spread and not the last value, because the question is whether asking
    still works. A rule sitting at the same rate across four revisions of the
    prompt is a rule the prompt cannot reach, whatever that rate happens to be.
    """
    rates = [
        v.rate for v in (r.find(rule) for r in readings) if v is not None and v.rate is not None
    ]
    if len(rates) < 2:
        return None
    return max(rates) - min(rates)


# A rule whose rate moved less than this across every revision of the prompt is
# reported as one the prompt cannot reach. One point, because the two groups on
# golden-dev are not close: the running head moved 75.6 points and everything
# else moved between 0.0 and 2.4, so anywhere in that gap picks the same rules
# and there is nothing to tune.
STUCK = 0.01


def table(readings: Sequence[Reading]) -> str:
    """The conformance rates side by side, and what the prompt bought."""
    names = [r.name for r in readings]
    out = ["| Rule | " + " | ".join(names) + " | Moved by | Reachable by prompt |"]
    out.append("| --- | " + " | ".join("---" for _ in names) + " | --- | --- |")
    for check in conformance.CHECKS:
        cells = []
        for reading in readings:
            verdict = reading.find(check.name)
            if verdict is None or verdict.rate is None:
                cells.append("did not apply")
            else:
                cells.append(f"{verdict.rate:.1%} of {verdict.applicable}")
        spread = moved(readings, check.name)
        if spread is None:
            cells.append("n/a")
            cells.append("n/a")
        else:
            cells.append(f"{spread:.1%}")
            cells.append("no" if spread < STUCK else "yes")
        out.append(f"| {check.name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def attribution(reading: Reading) -> str:
    """Every failing rule, split into convention and capability."""
    out = [
        "| Rule | Failures | Convention | Capability | Not attributed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for verdict in reading.verdicts:
        if not verdict.failures:
            continue
        out.append(
            f"| {verdict.rule} | {verdict.failures} of {verdict.applicable} | "
            f"{verdict.convention} | {verdict.capability} | {verdict.unattributed} |"
        )
    return "\n".join(out) + "\n"


def report(readings: Sequence[Reading], set_name: str) -> str:
    """The whole thing, as the Markdown section 08 asks for."""
    last = readings[-1]
    lines = [
        f"# Residual failures on {set_name}",
        "",
        f"{last.pages} pages, {len(readings)} revisions of the prompt.",
        "",
        "## What the prompt reached and what it did not",
        "",
        table(readings),
        f"## Where {last.name}'s failures come from",
        "",
        attribution(last),
    ]
    for verdict in last.verdicts:
        if not verdict.notes:
            continue
        lines.append(f"### {verdict.rule}")
        lines.append("")
        for note, count in sorted(verdict.notes.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {count} pages: {note}")
        lines.append("")
    formulas = last.formulas
    exact = f"{formulas.exact}"
    if formulas.scored:
        exact += f" ({formulas.exact / formulas.scored:.1%})"
    lines.extend(
        [
            "## Formulas",
            "",
            "| Spans | Scored | At or above 0.99 | Mean | On one side only |",
            "| --- | --- | --- | --- | --- |",
            f"| {formulas.total} | {formulas.scored} | {exact} "
            f"| {formulas.mean:.4f} | {formulas.unpaired} |",
            "",
        ]
    )
    return "\n".join(lines)
