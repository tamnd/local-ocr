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

This is a repair and it is honest about being one. It never rewrites a line the
reader produced, it only prepends one that is missing, and if the second look
fails or comes back with something that is not a head then the reading is passed
through as it was. A page that arrives here unreadable leaves here unreadable.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from local_ocr.batch import Reader, Refused
from local_ocr.rules.validate import looks_like_head, parse_page_label, parse_section_locator

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
# so it is dropped rather than prepended.
LONGEST = 90


def missing(text: str) -> bool:
    """Whether the reading needs a head put on it.

    Deliberately the same loose test the acceptance rule applies, so the pass
    fires on the pages the rule would reject and on no others. It cannot be the
    rule itself: the rule knows from the page map whether a page prints a head
    at all, and the reader does not have the page map. A page that prints no
    head gets one asked for, the strip comes back NONE, and nothing happens.
    """
    for line in text.splitlines():
        first = line.strip()
        if not first:
            continue
        if parse_page_label(first) is not None:
            return False
        if parse_section_locator(first) is not None:
            return False
        return not looks_like_head(first)
    return True  # an empty reading, which has other problems


def usable(answer: str) -> str | None:
    """The head out of what the second look said, or None."""
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[0].strip("`").strip()
    if not line or line.lower() in NOTHING:
        return None
    if len(line) > LONGEST:
        return None
    if parse_page_label(line) is not None or parse_section_locator(line) is not None:
        return line
    return line if looks_like_head(line) else None


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
    _strip: dict[Path, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    """What the second look cost, kept under the page it was taken for.

    The strip is written to a temporary directory that is gone by the time
    anybody asks, so its counts have to be moved to the page's key here or they
    are lost. A page read on 90 of 122 pages is not a rounding error.
    """

    async def read(self, image: Path, prompt: str) -> str:
        text = await self.inner.read(image, prompt)
        if not missing(text):
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
