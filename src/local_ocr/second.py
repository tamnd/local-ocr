"""Two readers, and what to do when they disagree.

This is M6. `HeadPass` fixed the defect the acceptance rules could see; this one
is aimed at the defect they cannot, which is a reading that passes every rule and
says something else than the page does. §05 calls that the only failure mode
worth building a second pipeline for, and it is right: a page with an unclosed
dollar gets rejected and read again, and a page where `\\aleph_0` became
`\\aleph_1` goes into the corpus and stays there.

## The shape

Read the page with A. Read it again with B. Compare, in `compare.py`, which is
deterministic and free. If nothing above low severity came up, write A's reading
and note in the sidecar that B agreed, which is worth more than it sounds: an
agreement between two unrelated model families is real evidence, and the sidecar
is where that evidence lives.

If something did come up, adjudicate in cost order, and stop when the budget runs
out rather than when the list does.

## The ladder, and why it is in this order

    crop        send the disputed region back, cropped
    reread      read the whole page again, cropped harder or sampled differently
    illegible   mark it, flag it, stop

Crop first because it is both the cheapest and the best. A crop of a quarter of
the page is a quarter of the vision tokens, so it costs less than a page read,
and it is also a *higher resolution* look at the disputed region, because a
vision encoder downsamples whatever it is given to a fixed budget. A formula that
occupies two per cent of a page gets two per cent of the token budget; the same
formula cropped gets all of it. That is the single most useful fact in this
module and it is why "crop to the formula specialist" and "re read at higher
dpi" are not two different rungs, as §05 assumed when it listed them. They are
the same rung, and cropping is how you get the dpi.

Re read second, and it is a genuine second rung rather than a repeat, because it
crops harder. If a crop of the region did not settle it, a tighter crop is more
resolution again, and past a point the model is being shown the formula at native
scan resolution, which is as much as the image has.

Illegible last. `⟪illegible⟫` is a real marker in this corpus with a rule
counting it, and the rule allows two per page. Marking is not failure: a page
that admits it could not read one symbol is worth more than a page that guessed,
because the first can be found again and the second cannot.

## The budget

Per page, in adjudications and not in tokens, because a token budget invites the
question of what a crop costs and the answer varies with the page. Three is the
default and the reasoning is arithmetic: a page read costs about eight seconds
at 300 dpi in the M5 measurements, a crop costs perhaps a third of that, and the
referee's page read is a whole page. So a page that goes all the way through the
ladder costs roughly 8 for A, 8 for B, and 3 crops at 3, which is about 25
seconds against 8, three times a plain read. At 436 pages an hour that is 145,
which is still comfortably above the 150 threshold in §00 only if the referee
runs on the pages that need it rather than on all of them, and that is the
`--second-when` knob below.

## What this refuses to do

It does not merge the two readings. If B wins a disagreement, B's whole reading
is written, not A's reading with B's formula spliced in. Splicing produces a page
that neither model ever produced and that neither can be held to, and the first
time it goes wrong nobody will be able to say which reader wrote the sentence.
The sidecar records both readings' hashes, so the losing reading is recoverable.
"""

from __future__ import annotations

import io
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from local_ocr.batch import Reader, Refused
from local_ocr.compare import Comparison, Difference, Severity, Where, compare
from local_ocr.sidecar import Adjudication, Read, Record, as_adjudication, digest, text_digest

BUDGET = 3
"""Adjudications a page may pay for. See the module docstring for the arithmetic."""

# The vertical slice of the page a crop takes, as a fraction of its height, at
# each rung. The first is a third of the page, which is enough to hold a display
# and the sentence either side of it; the second is a ninth, which is a formula
# and nothing else. Both are centred on where the difference was found.
CROPS = (0.34, 0.12)


class Step(StrEnum):
    CROP = "crop"
    REREAD = "reread"
    ILLEGIBLE = "illegible"
    BUDGET = "budget"
    """Not a rung. Recorded when the page ran out before this difference."""


class Winner(StrEnum):
    FIRST = "first"
    SECOND = "second"
    NEITHER = "neither"
    UNKNOWN = ""


# Asked of the adjudicating reader about a crop. Short on purpose. The long
# fleet prompt describes a whole page and its furniture, and none of that is
# true of a strip holding one display, so asking with it invites the model to
# invent a running head and a paragraph to go with the formula.
CROP_PROMPT = (
    "This image is a strip cut out of a printed page of a mathematics book. "
    "Transcribe exactly what it says, with the mathematics in LaTeX between "
    "dollar signs, and nothing else. Do not add a heading, a page number or any "
    "commentary. If part of it is genuinely unreadable, write ⟪illegible⟫ in "
    "that place and transcribe the rest."
)


def where_of(difference: Difference, text: str) -> float:
    """Roughly how far down the page a difference is, as a fraction of height.

    Text position is not image position and this does not pretend otherwise. It
    assumes the reading runs down the page in order, which for a single column
    book is true, and finds the line the difference sits on. Bourbaki is single
    column throughout, which is what makes the assumption safe here and would
    make it unsafe somewhere else.

    Returns 0.5 when it cannot tell, which crops the middle, which is where most
    of a page is.
    """
    lines = text.splitlines()
    if not lines:
        return 0.5
    needle = (difference.first or difference.second).strip().splitlines()
    if not needle or not needle[0]:
        return 0.5
    target = needle[0][:40]
    for index, line in enumerate(lines):
        if target and target in line:
            return min(1.0, max(0.0, (index + 0.5) / len(lines)))
    return 0.5


def crop(image: Path, out: Path, centre: float, height: float) -> Path:
    """A horizontal band of the page, written as a PNG.

    Full width, because the disputed thing is a line or a display and both run
    the width of the column, and because a horizontal cut cannot lose the left
    half of an equation the way a rectangle can.
    """
    from PIL import Image

    with Image.open(image) as page:
        band = max(1, int(page.height * height))
        top = int(page.height * centre) - band // 2
        top = max(0, min(page.height - band, top))
        strip = page.crop((0, top, page.width, top + band))
        buffer = io.BytesIO()
        strip.save(buffer, format="PNG")
    out.write_bytes(buffer.getvalue())
    return out


def settles(answer: str, difference: Difference) -> Winner:
    """Which reading the crop agrees with, if either.

    Substring containment after whitespace collapse, and nothing cleverer. A
    crop read is a short piece of text and the two candidates are short pieces
    of text, and a similarity score here would need a threshold that would be
    tuned on whatever page was in front of whoever tuned it.

    Both matching means the crop contains both, which happens when one reading
    is a prefix of the other, and that is not a verdict. Neither matching is the
    common case and is also not a verdict: it means the adjudicator read the
    strip differently again, and a third opinion that agrees with nobody is
    evidence that the region is hard, not evidence about who is right.
    """
    flat = " ".join(answer.split())
    first = " ".join(difference.first.split())
    second = " ".join(difference.second.split())
    if not flat:
        return Winner.UNKNOWN
    hit_first = bool(first) and first in flat
    hit_second = bool(second) and second in flat
    if hit_first and not hit_second:
        return Winner.FIRST
    if hit_second and not hit_first:
        return Winner.SECOND
    return Winner.UNKNOWN


def mark_illegible(text: str, difference: Difference) -> str:
    """Replace the disputed text with the marker, once.

    Only for a formula or a prose line, and only when the losing text is
    actually in the reading, because a structural difference is a count and
    there is nothing on the page to replace.
    """
    from local_ocr.rules.validate import ILLEGIBLE

    if difference.where is Where.STRUCTURE:
        return text
    target = difference.first.strip()
    if not target or target not in text:
        return text
    return text.replace(target, ILLEGIBLE, 1)


@dataclass
class Outcome:
    """What the second reader did to one page."""

    text: str
    record: Record


@dataclass
class SecondPass:
    """A primary reader with a referee behind it.

    A `Reader` itself, so `batch.run` drives it without knowing, exactly as with
    `HeadPass`. The two compose: the usual arrangement is `SecondPass(HeadPass(a),
    HeadPass(b))`, so each reader gets its head repaired before the two are
    compared, and a difference in running head means the two readers genuinely
    disagree about the top of the page rather than one of them having been
    wrapped and the other not.
    """

    first: Reader
    second: Reader | None = None
    """The referee. None is supported and logged once, per M6."""

    budget: int = BUDGET
    adjudicator: Reader | None = None
    """Who reads the crops. Defaults to the referee, which is the right default:
    the referee is the smaller model and a crop is a small job."""

    names: tuple[str, str] = ("first", "second")
    """What to call the two readers in the sidecar."""

    second_prompt: str = ""
    """What to ask the referee, when it is not what the primary was asked.

    Empty means ask both the same thing, which is the tidy case and is not the
    common one. The fleet prompt is eight kilobytes of Bourbaki convention
    written for a general instruction following model, and a purpose built OCR
    model does not read it as instructions: MinerU2.5 answered 114 of 124 pages
    with letter spaced text inside an `array` environment until it hit the token
    limit, and DeepSeek-OCR wants the two words its card documents and nothing
    else. Asking each in its own idiom is not a compromise of the comparison.
    What is compared is what the two say about the page, and the prompt is how
    each is asked, not what either is being asked about.
    """

    crop_prompt: str = ""
    """What to ask the adjudicator about a strip. Defaults to `CROP_PROMPT`.

    A model that needs its own words for a page needs them for a strip too, so
    when the referee has its own prompt this usually gets it as well.
    """

    models: tuple[str, str] = ("", "")
    revisions: tuple[str, str] = ("", "")
    """The repository and pinned commit of each, from `models.toml`.

    Carried here rather than looked up when the sidecar is written, because the
    sidecar has to record what was actually served and `models.toml` can be
    edited between a run starting and finishing.
    """

    records: dict[Path, Record] = field(default_factory=dict, init=False)
    """Every page this pass has read, keyed by its image.

    Held rather than written, because a `Reader` is handed an image and returns
    text and has no idea where the Markdown for it is going. The caller knows,
    and writes the sidecars beside the pages when the batch is done.
    """

    warned: bool = field(default=False, init=False)
    log: object = field(default=None, init=False)

    pages: int = field(default=0, init=False)
    disagreed: int = field(default=0, init=False)
    overruled: int = field(default=0, init=False)
    marked: int = field(default=0, init=False)
    spent: int = field(default=0, init=False)

    async def read(self, image: Path, prompt: str) -> str:
        return (await self.look(image, prompt)).text

    async def look(self, image: Path, prompt: str) -> Outcome:
        """Read a page with both, adjudicate, and hand back text and a record."""
        self.pages += 1
        record = Record(page=image.stem, image_sha256=digest(image))
        self.records[image] = record
        _measure(record, image)

        started = time.monotonic()
        text = await self.first.read(image, prompt)
        record.first = Read(
            reader=self.names[0],
            model=self.models[0],
            revision=self.revisions[0],
            prompt_sha256=text_digest(prompt),
            seconds=round(time.monotonic() - started, 3),
            text_sha256=text_digest(text),
        )
        record.gates = _gates(text)

        if self.second is None:
            # Supported, and said once rather than per page. A pipeline running
            # on one reader is a pipeline with no second opinion, which is worth
            # a warning and is not worth six thousand of them.
            if not self.warned:
                self.warned = True
                self._say("no referee configured, running on one reader")
            return Outcome(text, record)

        asked = self.second_prompt or prompt
        started = time.monotonic()
        try:
            other = await self.second.read(image, asked)
        except Refused as err:
            # A referee that will not answer is not an error in the page. A's
            # reading stands and the sidecar says why there is no second opinion,
            # which is the difference between "they agreed" and "nobody asked".
            record.second = Read(
                reader=self.names[1],
                model=self.models[1],
                revision=self.revisions[1],
                refused=str(err),
            )
            return Outcome(text, record)
        record.referee_ran = True
        record.second = Read(
            reader=self.names[1],
            model=self.models[1],
            revision=self.revisions[1],
            prompt_sha256=text_digest(asked),
            seconds=round(time.monotonic() - started, 3),
            text_sha256=text_digest(other),
        )

        comparison = compare(text, other)
        record.counts = comparison.counts()
        record.agreed = comparison.agreed
        if comparison.agreed:
            return Outcome(text, record)

        self.disagreed += 1
        text = await self._adjudicate(image, text, other, comparison, record)
        return Outcome(text, record)

    async def _adjudicate(
        self,
        image: Path,
        text: str,
        other: str,
        comparison: Comparison,
        record: Record,
    ) -> str:
        asking = comparison.worth_asking
        left = self.budget
        with TemporaryDirectory(prefix="local-ocr-adj-") as scratch:
            for difference in asking:
                if left <= 0:
                    record.unadjudicated += 1
                    continue
                left -= 1
                self.spent += 1
                result = await self._ladder(image, text, difference, Path(scratch))
                record.adjudicated.append(result)
                if result.winner == Winner.SECOND:
                    self.overruled += 1
                elif result.step == Step.ILLEGIBLE:
                    self.marked += 1
                    text = mark_illegible(text, difference)
        # One reading wins the page, not one reading per difference. The count
        # decides: if the referee won more of the disputes than the primary did,
        # the referee read the page better and its whole reading is written.
        wins = sum(1 for a in record.adjudicated if a.winner == Winner.SECOND)
        losses = sum(1 for a in record.adjudicated if a.winner == Winner.FIRST)
        if wins > losses:
            record.chose = "second"
            return other
        record.chose = "first"
        return text

    async def _ladder(
        self,
        image: Path,
        text: str,
        difference: Difference,
        scratch: Path,
    ) -> Adjudication:
        """Spend on one difference, cheapest rung first, and stop on a verdict."""
        judge = self.adjudicator or self.second
        if judge is None or difference.where is Where.STRUCTURE:
            # A structural difference has no region on the page to crop to: it is
            # a count, and a count is settled by reading the whole page again,
            # which is what the two readers already did. So it is recorded and
            # not spent on, and the page choice below uses the other differences.
            return as_adjudication(
                difference,
                step=str(Step.BUDGET),
                winner=str(Winner.UNKNOWN),
                evidence="structural differences are not adjudicated by crop",
            )

        centre = where_of(difference, text)
        started = time.monotonic()
        for rung, height in zip((Step.CROP, Step.REREAD), CROPS, strict=True):
            band = crop(image, scratch / f"{rung}.png", centre, height)
            try:
                answer = await judge.read(band, self.crop_prompt or CROP_PROMPT)
            except Refused:
                continue
            winner = settles(answer, difference)
            if winner is not Winner.UNKNOWN:
                return as_adjudication(
                    difference,
                    step=str(rung),
                    winner=str(winner),
                    evidence=_clip(answer),
                    seconds=round(time.monotonic() - started, 3),
                )
        # Two looks at increasing resolution and the adjudicator agreed with
        # neither reading. The honest outcome is to say so on the page.
        return as_adjudication(
            difference,
            step=str(Step.ILLEGIBLE),
            winner=str(Winner.NEITHER),
            evidence="two crops settled nothing",
            seconds=round(time.monotonic() - started, 3),
        )

    def _say(self, line: str) -> None:
        if callable(self.log):
            self.log(line)

    def summary(self) -> str:
        """The one line worth printing at the end of a batch.

        Reports what the second reader cost and what it bought. A run where
        `disagreed` is high and `overruled` is near zero is a run where the
        referee is being paid to agree, and that is the number that decides
        whether M6 stays switched on.
        """
        return (
            f"second reader: {self.pages} pages, {self.disagreed} disagreed, "
            f"{self.spent} adjudications, {self.overruled} overruled, {self.marked} marked"
        )


def _gates(text: str) -> dict[str, str]:
    """The acceptance rules, run on the primary's reading, as the sidecar sees them.

    This is what makes the M6 number mean anything. "The referee caught what the
    rules did not" is only a claim if the rules were actually run on the same
    reading, and it is `caught()` below that reads what this writes.

    Five of the eight run here and three do not, and the reason is that a reader
    host is handed a directory of images and knows nothing else. Rules 4 and 6
    need the page map and the volume's head grammar, and rule 7 needs a TeX
    installation, none of which exist on this machine. So `Expect()` is left at
    its defaults, which switches those rules off rather than guessing at them,
    and what remains is short, math, leak, illegible and exercise. Those five are
    the ones that catch a malformed reading, which is exactly the class the
    referee is not for, so the subset is the right one for this comparison even
    though it is not the whole gate.
    """
    from local_ocr.rules.validate import Expect, validate

    problems = validate(text, Expect())
    if not problems:
        return {"rules": "ok"}
    return {str(problem.rule): problem.detail for problem in problems}


def _measure(record: Record, image: Path) -> None:
    """Pixel dimensions and the dpi they imply, best effort.

    Best effort because a bad image should cost a sidecar field and not a page.
    The dpi is inferred from the height against the known renders rather than
    read from the file, because a PNG's stated dpi is whatever wrote it and the
    corpus renders through several tools.
    """
    try:
        from PIL import Image

        with Image.open(image) as page:
            record.width, record.height = page.width, page.height
    except Exception:
        return
    for dpi, height in ((300, 2776), (600, 5552), (150, 1388)):
        if abs(record.height - height) <= height * 0.15:
            record.dpi = dpi
            return


def _clip(text: str, width: int = 400) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def caught(records: Sequence[Record]) -> dict[str, int]:
    """What the second reader found that the gates did not, as counts.

    This is the number M6 asks for and it is the number that justifies the
    milestone. A disagreement on a page the acceptance rules already rejected is
    not a catch: that page was going to be read again anyway. A disagreement on
    a page that passed every rule is exactly what the referee is for, and is the
    only thing here worth calling a catch.

    The caller supplies the gate verdicts in `record.gates`, so this makes no
    judgement of its own about whether a page passed.
    """
    out = {"pages": 0, "passed_gates": 0, "disagreed": 0, "caught": 0, "high": 0}
    for record in records:
        out["pages"] += 1
        if not record.referee_ran:
            continue
        passed = all(v == "ok" for v in record.gates.values()) if record.gates else True
        if passed:
            out["passed_gates"] += 1
        if record.agreed:
            continue
        out["disagreed"] += 1
        if passed:
            out["caught"] += 1
        out["high"] += record.counts.get(str(Severity.HIGH), 0)
    return out
