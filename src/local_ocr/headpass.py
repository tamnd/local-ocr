"""A second, tiny look at the top of the page, for the line every reader drops.

The eight acceptance rules on the Go side decide whether a reading enters the
corpus, and on the first bake off run one rule rejected 177 of 200 pages on its
own: the running head. Rule 4 asks that the first line of the reading be the
line the volume prints across the top of the page, because that line carries the
chapter and the page label and is what a page is filed under.

Every reader measured here drops it, and not by accident. A document model is
trained to strip headers and footers, that being what everybody else wants, and
prompting barely moved it: reader-a was asked for the head in the first line of
a 1400 token prompt and produced one on 4.8 per cent of the pages that print
one, reader-d under a four word prompt managed 0.6 per cent.

So it is asked for separately. A page whose first line does not read as a head
is sent back with a crop of its top band and a one sentence instruction, and the
answer goes on the front of the reading. The crop is about a tenth of the page,
so the second look costs a fraction of the first, and only the pages that need
it pay for it.

This is a repair and it is honest about being one. It never invents a head, and
if the second look fails or comes back with something that is not a head then
the reading is passed through as it was. A page that arrives here unreadable
leaves here unreadable.

## The half a head

There is a second way to lose the line, and it costs about as much as losing it
whole. On the volumes that print a page label in the running head, the reader
often brings back the title and not the label: `ANNEAUX` where the page prints
`ANNEAUX A I.109`, `EXTENSIONS RADICIELLES` where it prints the label too. That
line reads as a head, so the gate above is happy with it, and rule 4 on the Go
side is not: a `head-label` volume has to show a label. The page is read again,
the reader drops the label again, and after three attempts the page is dead.

Counted over the raw readings on disk, 71 pages of four volumes are exactly
this, every one of them read three times: 32 in `alg-i-iii-fr`, 29 in
`ac-viii-ix-fr`, 9 in `alg-iv-vii-fr` and 1 in `alg-iv-vii`. None of them was
ever asked about, because none of them looked wrong from here.

So the wrapper watches the volume rather than being told about it. A batch is
one volume, the pages go past one at a time, and a volume that prints a label
prints it on nearly every body page. After eight pages, if most of them opened
with a label, the wrapper starts asking about the ones that do not. The strip
answer is only used when it is the page's own head with the label put back, so
a strip that answers something else changes nothing.

That is the one place this module edits a line rather than prepending one, and
it is the same line read off the same pixels at ten times the size. The guard is
`completes`: the strip has to carry a label, the page's line has to carry none,
and the page's line has to be contained in the strip's.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from local_ocr.batch import Reader, Refused
from local_ocr.rules.validate import (
    LONGEST_HEAD,
    looks_like_head,
    parse_page_label,
    parse_section_locator,
)

# The top band of the page, as a fraction of its height. Bourbaki prints the
# running head about 6 per cent down on a body page, and a scan is not square
# with the platen, so 12 leaves room for a page that sits crooked without
# pulling in the first line of the body and inviting the model to transcribe
# that instead.
BAND = 0.12

# The page number is called out because the first version of this prompt did
# not call it out and the model dropped it on almost every page that prints
# one. Asked for "the running head, exactly as printed" it answered TABLE DES
# MATIERES where the page prints 496 TABLE DES MATIERES, which is the same
# habit that loses the head in the first place: a document model is trained to
# treat a folio as furniture. The folio is half of what the head is for here,
# because it is what the page is filed under.
PROMPT = (
    "This image is the strip across the very top of a printed page. "
    "Reply with the running head printed there, exactly as printed, on one line, "
    "and nothing else. Include the page number if one is printed in the strip, "
    "on the side it is printed on, so that a strip printing "
    "18 ALGEBRAIC STRUCTURES Ch. I is answered with the 18 and not without it. "
    "Reply NONE if the strip holds no running head."
)

# What a reader says when there is nothing up there. Kept small, because the
# test below is what a head has to look like and this only keeps an obvious
# refusal out of the reading.
NOTHING = ("none", "none.", "no running head", "n/a", "-")

# A running head is a line. Longer than this and the model has transcribed the
# first paragraph of the body, which is the failure this module exists to undo,
# so it is dropped rather than prepended. The acceptance rule's number, because
# a head this module is happy with and a head the rule rejects is a page read
# twice for nothing.
LONGEST = LONGEST_HEAD


def reads_as_head(line: str) -> bool:
    """Whether a line is already a running head, and so not worth asking about.

    Three tests, and the order matters only in that the first one is the one
    that was missing. `parse_page_label` and `parse_section_locator` both search
    the line rather than matching it, which is right where they are used to pull
    a locator out of a citation and wrong here, because it makes any paragraph
    that happens to cite `A VIII.144` or `§ 5` answer yes to `is this line a
    running head`. On the 200 golden-dev readings that was not a corner case:

      nine readings open with a paragraph of body text that answers yes,
      the longest of them 1425 characters,
      against a longest genuinely printed head on the same set of 64.

    So the length gate comes first and the rest follow it. 90 rather than 64 is
    the same number `usable` uses and it leaves a printing whose heads run long
    more room than these do.

    A line with no letter and no digit is not a head either. `\\[` and `\\(` are
    what a reader writes when the page opens on a display, they carry no
    letters, and the capitals test reads a letterless line as a bare folio and
    waves it through. Two golden-dev pages are exactly that.
    """
    line = line.strip()
    if not line or len(line) > LONGEST:
        return False
    if not any(ch.isalnum() for ch in line):
        return False
    if parse_page_label(line) is not None or parse_section_locator(line) is not None:
        return True
    return looks_like_head(line)


def missing(text: str) -> bool:
    """Whether the reading needs a head put on it.

    Deliberately close to the test the acceptance rule applies, so the pass
    fires on the pages the rule would reject and on very few others. Both sides
    of the paragraph defect were fixed together, here and in the rule, so the
    two agree on the length gate and on the letterless line.

    Where they still differ this one is stricter, and in the direction that is
    cheap to be wrong in: the cost of asking about a page that did not need it
    is one crop of a tenth of a page, and the cost of not asking is a page that
    enters the corpus with a paragraph where its head should be.

    It cannot be the rule itself for a second reason: the rule knows from the
    page map whether a page prints a head at all, and the reader does not have
    the page map. A page that prints no head gets one asked for, the strip comes
    back NONE, and nothing happens.
    """
    for line in text.splitlines():
        first = line.strip()
        if not first:
            continue
        return not reads_as_head(first)
    return True  # an empty reading, which has other problems


def usable(answer: str) -> str | None:
    """The head out of what the second look said, or None."""
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[0].strip("`").strip()
    if not line or line.lower() in NOTHING:
        return None
    return line if reads_as_head(line) else None


# How many pages go past before the wrapper will believe anything about what
# this volume prints, and what share of them has to carry a page label.
#
# Eight and three fifths. A batch is sixteen pages at the smallest, so eight is
# half of the smallest thing this ever runs on and the belief is formed early
# enough to be worth having. Three fifths rather than a bare majority because
# the front matter of a volume prints no label at all and a batch that opens on
# the front matter would otherwise spend its first pages arguing with itself.
# On alg-i-iii-fr, which is the volume this was measured on, 396 of the 480
# pages read carry a label on every attempt, so the real share is 82 per cent
# and neither number is close to the edge.
LEARN = 8
SHARE = 0.6


def labelled(text: str) -> bool:
    """Whether the reading opens with a line that carries a page label."""
    return parse_page_label(_first(text)) is not None


def completes(head: str, text: str) -> bool:
    """Whether the strip answer is the page's own head with its label put back.

    Three tests, and all three matter. The strip has to carry a label, or there
    is nothing to put back. The page's line must not already carry one, or there
    was nothing wrong with it. And the page's line has to be contained in the
    strip's, on the letters and digits alone, because that containment is what
    says the two are the same head rather than two different readings of the
    top of the page, and a strip answer that is not the page's head is a strip
    answer this module will not put in place of one.
    """
    first = _first(text)
    if parse_page_label(head) is None or parse_page_label(first) is not None:
        return False
    key = _key(first)
    return bool(key) and key in _key(head)


def _first(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _key(line: str) -> str:
    return "".join(ch for ch in line.casefold() if ch.isalnum())


def _replace_first(head: str, text: str) -> str:
    """The reading with its first line swapped for the one off the strip."""
    lines = text.splitlines(keepends=True)
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        end = "\n" if line.endswith("\n") else ""
        lines[number] = head + end
        return "".join(lines)
    return text


def _same(head: str, text: str) -> bool:
    """Whether the strip handed back the line the reading already opens with.

    Compared on the letters and digits alone, because the two came out of two
    different requests and the spacing and the punctuation around a page label
    are the first things to differ between them.
    """
    key = _key(head)
    return bool(key) and key == _key(_first(text))


def band(image: Path, out: Path, fraction: float = BAND) -> Path:
    """The top strip of a page, written as a PNG.

    Pillow rather than a second pdftoppm run: the page image is what the fleet
    sent, and the crop has to come from that exact image rather than from a
    second rasterisation that might differ in a way nobody would check.
    """
    from PIL import Image

    with Image.open(image) as page:
        height = max(1, int(page.height * fraction))
        strip = page.crop((0, 0, page.width, height))
        buffer = io.BytesIO()
        strip.save(buffer, format="PNG")
    out.write_bytes(buffer.getvalue())
    return out


@dataclass
class HeadPass:
    """A reader wrapped in the second look.

    It is a Reader itself, so the batch does not know it is there and neither
    does anything else. That is the point: the repair is part of what reading a
    page means here, not a stage somebody has to remember to run.
    """

    inner: Reader
    fraction: float = BAND
    prompt: str = PROMPT
    asked: int = field(default=0, init=False)
    fixed: int = field(default=0, init=False)
    completed: int = field(default=0, init=False)
    """Pages whose head was there but had lost its label. See the module note."""

    seen: int = field(default=0, init=False)
    labels: int = field(default=0, init=False)
    """What this batch has shown about the volume: how many readings went past
    and how many of them opened with a page label."""
    _strip: dict[Path, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    """What the second look cost, kept under the page it was taken for.

    The strip is written to a temporary directory that is gone by the time
    anybody asks, so its counts have to be moved to the page's key here or they
    are lost. A page read on 90 of 122 pages is not a rounding error.
    """

    def wants_label(self) -> bool:
        """Whether this batch has shown that its volume prints a page label."""
        return self.seen >= LEARN and self.labels >= self.seen * SHARE

    async def read(self, image: Path, prompt: str) -> str:
        text = await self.inner.read(image, prompt)
        # Counted before the decision, so the page being judged is part of the
        # evidence about its own volume. A label-less page is a vote against the
        # volume printing labels and it should be allowed to cast it.
        self.seen += 1
        if labelled(text):
            self.labels += 1
        gone = missing(text)
        if not gone and not (self.wants_label() and not labelled(text)):
            return text
        self.asked += 1
        with TemporaryDirectory(prefix="local-ocr-head-") as scratch:
            strip = band(image, Path(scratch) / "head.png", self.fraction)
            try:
                answer = await self.inner.read(strip, self.prompt)
                self._keep(image, strip)
            except Refused:
                # The page itself was read. A failed second look leaves a page
                # without a head, which the acceptance rules will reject and
                # somebody will read again, and that is a better outcome than
                # throwing away a reading that exists.
                return text
        head = usable(answer)
        if head is None:
            return text
        if not gone:
            # The page has a head and it is short of its label. Only the strip's
            # own reading of that same head goes in its place.
            if not completes(head, text):
                return text
            self.completed += 1
            return _replace_first(head, text)
        if _same(head, text):
            # The gate thought the page had no head and the strip disagreed by
            # handing back the line the page already opens with. Prepending it
            # would give the page two heads, which is a worse reading than the
            # one that arrived, so the reading is passed through and the ask is
            # counted rather than the fix.
            return text
        self.fixed += 1
        return f"{head}\n\n{text.lstrip()}"

    def _keep(self, image: Path, strip: Path) -> None:
        got = self._ask(strip)
        if got is not None:
            self._strip[image] = got

    def _ask(self, image: Path) -> tuple[int, int] | None:
        ask = getattr(self.inner, "usage", None)
        if not callable(ask):
            return None
        try:
            got = ask(image)
        except Exception:
            return None
        return got if isinstance(got, tuple) and len(got) == 2 else None

    def usage(self, image: Path) -> tuple[int, int] | None:
        """The page, plus the strip when a second look was taken for it.

        One number for what reading this page cost, because from outside this
        wrapper reading the page is one thing. Splitting the head pass out would
        need its own field in the sidecar and nobody has asked a question that
        wants it separated.
        """
        page = self._ask(image)
        strip = self._strip.pop(image, None)
        if page is None:
            return strip
        if strip is None:
            return page
        return page[0] + strip[0], page[1] + strip[1]
