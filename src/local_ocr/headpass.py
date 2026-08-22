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

## The fragment

There is a third way to lose the line and it is the largest of the three. The
reader brings back one end of the head and nothing else: `§ 2` where the page
prints `§ 2 EXERCICES TG I.91`, `N° 2` where it prints `N° 2 LIMITES TG I.47`.
That is not a head at all, it is a piece of one, but it answers yes to every
test above. `parse_section_locator` finds the section marker, so the gate thinks
the page has its head, and the volume rule above is the only thing that could
ask about it.

The volume rule cannot. It learns inside one batch, a batch is about fifteen
pages, and it needs eight before it will believe anything, so it opens two
thirds of the way through the batch at the earliest. Worse, on the pages this
defect lands on it never opens at all: the loss is systematic down one side of
the exercise pages, so half the batch comes back with no page label, the share
sits at 0.5625 against a threshold of 0.6, and the pages that need the second
look are the votes keeping it shut. Counted over the batch directories on disk,
323 readings open with a fragment and the volume rule asked about 13 of them.

So a fragment is recognised for what it is, without reference to the volume. A
line that reads as a head, carries no page label and holds no word at all is not
a head any volume prints; a running head has a title in it. Those pages get the
second look whatever the volume has shown, and the strip's answer goes in place
of the fragment when it extends it, which is `extends` below.

Asked about all 189 distinct fragment pages on disk, the strip answered with a
usable head on every one and 173 of them extended the fragment. The other 16 are
passed through: the strip read the top of the page as something with no piece of
the fragment in it, and this module does not guess between two readings.

## The body heading

The fourth way is the reader answering with a line that really is printed on the
page, in capitals, the right length, carrying a section marker, and is not the
running head. On a page that opens a section the body's own heading is set a
third of the way down, `§ 5. APPLICATIONS OUVERTES ET APPLICATIONS FERMEES`, and
that is what comes back. The head is printed above it and is lost.

Nothing in this module could tell the two apart, for the same reason the
fragment defeated it: `parse_section_locator` searches the line rather than
matching it. The full stop after the section number does tell them apart. The
volumes print `§ 5` in the running head and `§ 5.` at the top of the body, and
across the 4520 pages read so far 42 first lines open with a section number, a
full stop and a title. Not one carries a page label and not one is a running
head.

So those pages are treated as having no head at all, which is what they have, and
the strip's answer goes on the front of the reading rather than in place of the
line. The body heading stays where it belongs, in the body. Asked about all 42
against the live reader, 33 came back with the page's own running head and 9 came
back NONE. All nine are `ac-x-fr`, whose section opening pages print no head, so
nothing happens to them and nothing should.

`usable` asks the same question of the strip's answer, because a strip that read
past the top band brings back the body heading too, and the answer to that is not
to put it on the front of the page.

## The volume

Three of the four repairs above work without asking what volume the page is
from, and that was not the plan, it was the only way past a rule that does not
hold. The volume rule learns from fifteen contiguous pages of one volume, spends
the first eight refusing to answer, and then answers from a sample nobody would
accept. `completes` is the one repair that cannot route around it, because
whether a head with a title and no label is a defect really does depend on the
volume, and on `top-v-x-fr` it depends on the page: `CHAPITRE V` and `NOTE
HISTORIQUE` print no label and are right, `§ 2 MESURE DES GRANDEURS` prints one
and lost it.

So the rule is not guessed when it can be told. `head_label` on the wrapper
carries what the caller knows, and the caller does know: the corpus records the
grammar of every volume and a batch is one volume. Unset means guess, which is
what a run started by hand still does.

The guess itself is fixed as far as it can be from here, by taking the vote away
from the pages that lost their head. A reading with no head, a fragment, or the
body's section heading is not evidence about what a volume prints across the top
of a page. Letting those pages vote is what pins the share at 0.5625 on the
exercise batches: the loss runs down one side of the page, so the pages needing
the second look are the votes keeping it shut. Over the 118 batches of a
`head-label` volume on disk, abstention opens 19 that stay shut today and closes
none, and the pages asked about go from 22 to 30 of the 280 that want it.

That is still 30 of 280, which is the measurement that says the guess is not
worth improving further. Told rather than guessed, all 280 are asked. Guessing
open instead was measured too and rejected: it asks about 1643 pages on the
volumes that print no label, and on a sample of 40 of them the strip changed
nothing at all, because `completes` wants a label in the answer and those
volumes print none. Zero harm and 1643 crops is not a trade, it is a waste.

## The prompt coming back

A prompt that shows the reader an example is a prompt the reader can hand back.
Across the 4520 pages read so far, 17 readings on 15 different volumes open with
`ALGEBRAIC STRUCTURES Ch. I`, which is the first example out of this module's
prompt, on Commutative Algebra, Lie, Integration, Topology and the Historical
Note, where those words are printed nowhere. Two pages of `alg-iv-vii` go
further and carry the whole production OCR prompt as their body, so the echo is
not something this module invented, but this module hands it a line to sit on.

Refused in `usable`, where an answer that is an example is treated as no answer,
which leaves the page alone rather than putting somebody else's title on it. The
test is the example with its numbers taken out, compared whole. That is exactly
the form the echo arrives in and it is not a form a real page produces: the 47
pages of `alg-i-iii` that really do print `ALGEBRAIC STRUCTURES` never print the
` Ch. I`, and a strip that prints the example for real prints the folio with it.
Nothing on disk matches an example entire.

Why the reader does it is not settled. Asked again for the strips of five of the
17 at 150 and 300 dpi, the live reader answered NONE every time, and the images
those batches used are rendered and deleted, so the ones that produced it are
gone. The guard is worth having either way, because it costs a page nothing to
be left alone.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from local_ocr.batch import Reader
from local_ocr.rules.validate import (
    LONGEST_HEAD,
    looks_like_head,
    page_label_span,
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
#
# Calling out the page number was not enough, because a Bourbaki head has three
# parts and not two. The volumes print "N° 4  EXTENSIONS TRANSCENDANTES  A
# V.107" on the odd page and "A V.130  CORPS COMMUTATIFS  § 16" on the even
# one, and asked for the head with its page number the model returned the
# middle part alone on the odd pages: EXTENSIONS TRANSCENDANTES, no number at
# either end. On the alg-iv-vii-fr window of 22 August that cost 62 of 118
# readings their page label, and on a head-label volume a head with no label is
# a page rule 4 rejects.
#
# So the instruction now names both ends and gives an example of each shape. Run
# side by side against the old wording over 26 live page strips, 10 off
# alg-iv-vii-fr and 16 off alg-iv-vii, the strips that came back carrying a page
# label went from 15 to 22. Counting only the 24 pages that print a label at all,
# from 15 to 22 again, and no strip that answered with a label under the old
# wording lost it under the new one.
# The two heads the prompt shows the reader. Named rather than written into the
# wording, because `echoed` has to know what they are and a second copy of them
# would drift away from the first one the next time the prompt is reworded.
EXAMPLES = ("18 ALGEBRAIC STRUCTURES Ch. I", "N 2 EXTENSIONS GALOISIENNES A V.57")

PROMPT = (
    "This image is the strip across the very top of a printed page. "
    "Reply with the running head printed there, exactly as printed, on one line, "
    "and nothing else. Transcribe the whole line from its left edge to its right edge, "
    "including whatever is printed at either end: a page number, a page reference such as "
    "A V.113, a section marker such as N degree 2 or a section sign and its number. "
    f"A strip printing {EXAMPLES[0]} is answered with the 18 and not without it, "
    f"and a strip printing {EXAMPLES[1]} is answered with all three parts "
    "and not with the middle part on its own. "
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


# The section heading printed in the body of a page that opens a section, which
# is the one line in the volume that looks more like a running head than the
# running head does. `§ 5. APPLICATIONS OUVERTES ET APPLICATIONS FERMEES` is in
# capitals, is the right length, and carries a section marker that
# `parse_section_locator` finds, so every test in this module says it is the head
# and the reader that returned it instead of the head gets away with it.
#
# The full stop after the number is what separates the two. The volumes print
# `§ 5` in the running head and `§ 5.` at the top of the body, and across the
# 4520 pages read so far 42 first lines open with a section number, a full stop
# and then a title. Not one of them carries a page label and not one of them is a
# running head: they are section openings, plus three lines of a table of
# contents with the dot leaders still attached.
#
# Something has to follow the stop. `§ 5.` on its own is a piece of a head with
# the stop misread onto it, and that is a fragment, which is repaired by putting
# the rest of the head in its place rather than by putting a head in front of it.
HEADING = re.compile(r"^\s*§\s*\d+\s*\.\s*\S")


def heading(line: str) -> bool:
    """Whether a line is the body's section heading rather than the page's head.

    Asked of the page's first line, so the head gets put in front of it rather
    than nothing happening, and asked of the strip's answer, so a strip that read
    too far down the page does not have its answer put on the front of one.

    Not folded into `reads_as_head`, which is the question of whether the
    acceptance rules will take the line. They will, and that is the defect. What
    this says is narrower: whatever else it is, this line is not the line printed
    across the top of the page.
    """
    return HEADING.match(line) is not None


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
        # The body heading first, because it passes the test below. See
        # `heading`: 42 pages open with one and the head is printed above it.
        return heading(first) or not reads_as_head(first)
    return True  # an empty reading, which has other problems


def usable(answer: str) -> str | None:
    """The head out of what the second look said, or None."""
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[0].strip("`").strip()
    if not line or line.lower() in NOTHING:
        return None
    if heading(line):
        # The strip read past the top band into the body of the page, and the
        # body heading is not the head however much it looks like one.
        return None
    if echoed(line):
        # The reader handed back the example out of the prompt instead of NONE.
        return None
    return line if reads_as_head(line) else None


def echoed(line: str) -> bool:
    """Whether a line is one of the prompt's own examples handed back as an answer.

    Matched against the examples with their numbers taken out, because that is
    the form the reader hands back. A near blank strip sometimes comes back as
    `ALGEBRAIC STRUCTURES Ch. I` rather than NONE, and the 18 is dropped on the
    way, which is the tell. The strip that really prints that head prints the
    folio with it.

    Compared whole and not by containment, because both examples are real
    running heads on the volumes they were taken off. 47 pages of `alg-i-iii`
    open with `ALGEBRAIC STRUCTURES`, some of them with the volume numeral in
    front, and every one of them is a page that says so. Not one carries the
    ` Ch. I` that the prompt's example carries. Across the 4520 pages read so
    far nothing matches an example entire and 17 match one stripped, on volumes
    of Commutative Algebra, Lie, Integration, Topology and the Historical Note
    where the words appear nowhere in the book.
    """
    key = _key(line)
    return any(key == _key(re.sub(r"\d+", "", example)) for example in EXAMPLES)


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


# A word, for the purpose of deciding that a line has a title in it. Three
# letters rather than one because a Bourbaki head prints ordinals and section
# markers around the title, N° and Ch and no, and none of those is the title. Of
# the 189 fragment pages on disk the longest run of letters in any of them is
# two, and the shortest title word in a head read off the same volumes is four,
# so nothing on this corpus sits near the boundary.
#
# It matches letters and not \\w, because \\w takes digits and underscores and a
# bare folio is exactly what must not count as a word here.
WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def fragment(line: str) -> bool:
    """Whether a line is a piece of a running head rather than one.

    See the module note. A head has a title in it and this line has no word at
    all, so whatever it is, no volume prints it across the top of a page on its
    own. The page label is excluded because a line carrying one has already been
    read well enough to file the page under, and the rest of the head is what
    `completes` is for.

    Deliberately not part of `reads_as_head`. That test answers whether the
    acceptance rules will take the line, and a fragment is a line they take and
    should not; changing it there would reject the page instead of repairing it,
    which is the wrong direction. The whole point of this module is that a page
    with a bad first line is worth a second look and not a rejection.
    """
    line = line.strip()
    return reads_as_head(line) and parse_page_label(line) is None and WORD.search(line) is None


def extends(head: str, text: str) -> bool:
    """Whether the strip's answer is the page's fragment with the rest put back.

    The fragment has to be one end of the strip's head, on the letters and
    digits alone. One end rather than anywhere in it, because that is where the
    reader keeps the piece from: the head prints its title between a section
    marker and a page label, and what survives is the marker or the label, never
    the middle. Measured over the 173 pages this accepts, every one of them has
    the fragment at an end, so the tighter test costs nothing and it refuses a
    class the looser one would take: a fragment of one digit is inside almost
    any head, including one belonging to a different section.

    Over the head with its page label cut out, for the reason `completes` gives:
    a one character key against a head that carries `A I.119` is satisfied by the
    digits of the label and says nothing about whether the two agree.

    A strip that hands back the fragment and no more extends nothing, so the
    length test is not decoration.
    """
    key = _key(_first(text))
    whole = _key(_unlabelled(head))
    if not key or len(whole) <= len(key):
        return False
    return whole.startswith(key) or whole.endswith(key)


def completes(head: str, text: str) -> bool:
    """Whether the strip answer is the page's own head with its label put back.

    Three tests, and all three matter. The strip has to carry a label, or there
    is nothing to put back. The page's line must not already carry one, or there
    was nothing wrong with it. And the page's line has to be contained in the
    strip's, on the letters and digits alone, because that containment is what
    says the two are the same head rather than two different readings of the
    top of the page, and a strip answer that is not the page's head is a strip
    answer this module will not put in place of one.

    The containment runs over the head with its page label cut out. That matters
    more than it sounds. Of the 206 readings on the three head-label volumes
    whose first line opens with a section sign, 183 carry an alphanumeric key of
    one or two characters, almost always a single digit, so a containment test
    that runs over the whole head is satisfied by the digits of the label itself
    and says nothing. Pairing every one of those short lines against every full
    head read on the same volume, 49695 wrong pairings in all, containment over
    the whole head accepts 25.6 per cent of them and containment over the head
    without its label accepts 11.3 per cent.

    Nothing here checks the label the strip hands back, and nothing here can.
    The point of the pass is to recover a label the page did not give up, so
    there is no second copy to check it against. What catches a wrong one is
    rule 6, which compares the label to the page map whenever the map's
    confidence is printed, and on every head-label volume it is.

    Containment rather than a prefix or a suffix test because the fragment the
    reader keeps sits at either end. On the body pages of ac-viii-ix-fr the head
    reads "AC VIII.14 DIMENSION § 2" and the reader keeps the tail. On the
    exercise pages it reads "§ 4 EXERCICES AC VIII.93" and the reader keeps the
    head. Both were read off the same volume.
    """
    first = _first(text)
    if parse_page_label(head) is None or parse_page_label(first) is not None:
        return False
    key = _key(first)
    return bool(key) and key in _key(_unlabelled(head))


def _first(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def _key(line: str) -> str:
    return "".join(ch for ch in line.casefold() if ch.isalnum())


def _unlabelled(head: str) -> str:
    """The head with its page label taken out, so its digits stop counting."""
    span = page_label_span(head)
    if span is None:
        return head
    start, end = span
    return f"{head[:start]} {head[end:]}"


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
    head_label: bool | None = None
    """Whether this batch's volume prints a page label, when the caller knows.

    None means guess, which is what happens when nobody said. See `wants_label`.
    """

    asked: int = field(default=0, init=False)
    fixed: int = field(default=0, init=False)
    completed: int = field(default=0, init=False)
    """Pages whose head was there but had lost its label. See the module note."""

    mended: int = field(default=0, init=False)
    """Pages that came back with a piece of the head. See the module note."""

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
        """Whether the volume this batch belongs to prints a page label.

        Told when the caller knows, and guessed from the batch when it does not.
        The guess is the weakest thing in this module and the module note says
        why: it is a fact about a volume being learned from fifteen contiguous
        pages of it, so it spends the first eight of them refusing to answer and
        the rest answering from a sample nobody would accept.

        The pages that lost their head do not vote. A reading with no head, or
        with a piece of one, or with the body's section heading on the front, is
        not evidence about what the volume prints; it is evidence that the
        reader dropped it. Letting those pages vote is what pins the share at
        0.5625 on the exercise batches, where the loss is systematic down one
        side of the page, so the pages that need the second look are the votes
        keeping it shut. Counted over the 118 batches of a `head-label` volume
        on disk, abstention opens 19 that stay shut today and closes none.
        """
        if self.head_label is not None:
            return self.head_label
        return self.seen >= LEARN and self.labels >= self.seen * SHARE

    async def read(self, image: Path, prompt: str) -> str:
        text = await self.inner.read(image, prompt)
        gone = missing(text)
        # A fragment is asked about whatever the volume has shown, because it is
        # not a head on any volume. See the module note: the volume rule cannot
        # reach these pages and on the batches they fall in it never opens.
        piece = fragment(_first(text))
        # Counted after the two tests above and before the decision, so the page
        # being judged is part of the evidence about its own volume but only if
        # it is evidence. See `wants_label`: a reading that lost its head tells
        # you nothing about what the volume prints across the top of a page.
        if not gone and not piece:
            self.seen += 1
            if labelled(text):
                self.labels += 1
        if not gone and not piece and not (self.wants_label() and not labelled(text)):
            return text
        self.asked += 1
        with TemporaryDirectory(prefix="local-ocr-head-") as scratch:
            try:
                strip = band(image, Path(scratch) / "head.png", self.fraction)
                answer = await self.inner.read(strip, self.prompt)
                self._keep(image, strip)
            except Exception:
                # The page itself was read. A failed second look leaves a page
                # without a head, which the acceptance rules will reject and
                # somebody will read again, and that is a better outcome than
                # throwing away a reading that exists.
                #
                # Everything and not just Refused, and the crop inside the guard
                # rather than in front of it. The crop opens the page image a
                # second time and it can fail on its own: an image Pillow will
                # not decode raised straight out of here and the batch turned a
                # page it had read into a refused one. Which pages those are is
                # the worst part of it. A page image truncated by an interrupted
                # copy is exactly the page whose reading is worth keeping,
                # because the reading is the only thing left that came off it.
                return text
        head = usable(answer)
        if head is None:
            return text
        if not gone:
            # The page has a head and it is short of something. Only the strip's
            # own reading of that same head goes in its place.
            #
            # Which guard applies is decided by what is wrong with the page and
            # not by which test happens to pass. The two overlap: on a head-label
            # volume a fragment carries no label either, so `completes` accepts
            # 103 of the 189 fragment pages on disk and would count them as lost
            # labels. They are not. The line is a piece of a head, the repair is
            # the fragment one, and the counters are the only account anybody
            # gets of what the second look did.
            if piece:
                if not extends(head, text):
                    return text
                self.mended += 1
                return _replace_first(head, text)
            if completes(head, text):
                self.completed += 1
                return _replace_first(head, text)
            return text
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
