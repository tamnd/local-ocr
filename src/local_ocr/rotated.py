"""Small type set on its side, which is where the measured failure was.

    local-ocr kvant rotated --draw
    local-ocr kvant rotated --readings out/reader-a

M8 item 6 asks for a rotated and small type golden set, and it says why: six
point type at ninety degrees is where the failure was measured, not somewhere a
page average would show it. A page carrying one vertical illustrator credit in
the gutter is 2 000 characters of ordinary Russian and 24 characters of the hard
thing, so a reader that drops the credit entirely moves the page's character
rate by about one per cent and moves this set's number by a hundred.

The ground truth is free, which is the reason this set can exist at all. The
credit is real text in the publisher's PDF, drawn with a rotated text matrix,
and `pdftotext -bbox-layout` reports its box. A run set on its side has a box
several times taller than it is wide, so the geometry finds it and the text
layer says what it says, and nothing here needs a person to type a reference out.

The geometry finds the rotation and it does not measure the type size, which is
worth saying because the first version of this module claimed it did. Poppler
reports the same credit line, the same words at the same size on the printed
page, as 2.1 points across in 2020, 8.9 in 2007 and 16.6 in 2022. The narrow
side of a rotated box is not the cap height. The type in this set is small, at
roughly six to seven point on the paper, but that is a fact about what Kvant
sets sideways rather than something the filter below checks.

Finding it by looking for the word `Иллюстрация` in the extracted Markdown does
not work and that is the point of the set. The vertical runs are exactly the
runs an extractor loses or scrambles, so searching the extraction for them finds
the pages where nothing went wrong. Searching the geometry finds all of them: a
text search over the corpus turned up 22 pages, and the boxes turn up 302 runs
on 151 PDF pages of the 130 cached issues, spread from 2007 to 2024.

What the thresholds are and why. All three were swept over those issues rather
than picked, and the sweep is the reason each one is where it is:

  RATIO    2.0   Height over width. Poppler reports plenty of ordinary
                 horizontal words whose box comes out a little taller than wide,
                 and that noise sits between 1.0 and 1.3: `А.И.` at 12.5 by
                 13.0. Genuine rotated runs in this set are 3.5 and up, median
                 6.3, so two sits in the empty space between the two groups.
  WIDEST   18.0  Points across the narrow side. Not a type size gate, for the
                 reason above; a guard against the handful of ordinary
                 horizontal words that land in a box tall enough to clear the
                 ratio. Twelve was the first guess and it cut the set at 2020,
                 because that is where poppler's reported width for the same
                 credit crosses twelve. Eighteen keeps every year and changes
                 nothing else: the next runs up are at 48 points and are
                 horizontal words like `тела` in a tall figure box.
  SHORTEST 5     Characters. At four the set gains 33 copies of `Рис.` and
                 nothing else. A rotated `Рис.` is a real rotated run, but three
                 letters repeated 33 times measures whether the reader has seen
                 the shape once, not whether it can read what is set sideways.

The set is narrow and it is worth saying how narrow. 63 pages, 126 runs, 10
distinct strings, because what Kvant sets sideways is the illustrator credit and
almost nothing else. Every page carries exactly two runs, the word `Иллюстрации`
or `Иллюстрация` and a surname, and the surname is the measurement: the word is
on 62 of the 63 pages and a reader either knows that shape or does not, whereas
`М.Сумниной` turned ninety degrees is a string nobody can guess and no
spell-check can repair. Fifty of the surname runs are `Д.Гришуковой`, who
illustrated the magazine for fifteen years, so a reader that has memorised one
name will score better here than it deserves to and that should be read off the
per run list rather than off the headline.

Most of the runs the cache holds are not in the set, and that is the corpus
rather than the filter. Over the 130 cached issues the boxes find 302 runs on
151 PDF pages; 126 of them sit on a sheet the corpus has extracted and is not
holding out, 18 are on a held out page, and the remaining 158 are on sheets the
corpus does not hold at all, so `align` has nothing to map them to. Every one of
the 176 is named by `draw` on standard error rather than dropped in silence.

Recall and not a character rate. A run is caught or it is not, because half of
an illustrator's surname is not worth partial credit to anything downstream, and
because a rate over 24 characters on a page is a number with no resolution.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_ocr import kvant
from local_ocr.russian import fold

RATIO = 2.0
WIDEST = 18.0
SHORTEST = 5

WORD = re.compile(
    r'<word xMin="([\d.eE+-]+)" yMin="([\d.eE+-]+)" '
    r'xMax="([\d.eE+-]+)" yMax="([\d.eE+-]+)">(.*?)</word>'
)
"""One word and its box out of `pdftotext -bbox-layout`.

By regular expression and not by an XML parser. The file is 4 MB for one issue,
the element is written by poppler on one line in a fixed order, and the parser
would cost a dependency and a tree for a shape that is this rigid. If poppler
ever changes the order this stops matching and `runs` returns nothing, which is
visible, rather than matching the wrong thing, which is not.
"""

PAGE = "<page "

CYRILLIC = re.compile(r"[А-Яа-яЁё]")

LETTERS = re.compile(r"[^A-Za-zА-Яа-яЁё]+")
"""Everything that is not a letter, for the comparison in `caught`.

The credits are set as `Д.Гришуковой` and a reader may write `Д. Гришуковой` or
`Д.  Гришуковой`, and the space is the typesetting rather than the reading. The
letters are the transcription and they are what has to be right.
"""


@dataclass(frozen=True)
class Run:
    """One run of type set on its side, and where it is."""

    issue: str
    pdf_page: int
    """One based, the way `kvant.align` counts."""
    text: str
    width: float
    """Points across the narrow side, which is not the type size. See the module."""
    height: float

    @property
    def ratio(self) -> float:
        return self.height / self.width if self.width else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "issue": self.issue,
            "pdf_page": self.pdf_page,
            "text": self.text,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
        }


def boxes(pdf: Path, *, run: Callable[..., object] = subprocess.run) -> list[list[tuple]]:
    """Every word of a PDF with its box, one list per page, in order.

    One `pdftotext` for the whole file rather than one per page, for the reason
    `kvant.pdftext` gives: an issue is a tenth of a second whole and several
    seconds a page at a time, and this runs over 130 of them.
    """
    out = run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout  # type: ignore[attr-defined]
    pages = []
    for page in out.split(PAGE)[1:]:
        found = []
        for match in WORD.finditer(page):
            x0, y0, x1, y1 = (float(v) for v in match.group(1, 2, 3, 4))
            found.append((x0, y0, x1, y1, html.unescape(match.group(5))))
        pages.append(found)
    return pages


def sideways(x0: float, y0: float, x1: float, y1: float, text: str) -> bool:
    """Whether this box holds small Russian type set on its side."""
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return False
    if len(text) < SHORTEST or not CYRILLIC.search(text):
        return False
    return height / width >= RATIO and width <= WIDEST


def runs(issue: str, pdf: Path, *, read: Callable[[Path], list[list[tuple]]] = boxes) -> list[Run]:
    """The rotated small type of one issue, page by page."""
    found = []
    for number, page in enumerate(read(pdf), start=1):
        for x0, y0, x1, y1, text in page:
            if sideways(x0, y0, x1, y1, text):
                found.append(Run(issue, number, text, x1 - x0, y1 - y0))
    return found


def sheets(
    issue: str,
    pages: Sequence[kvant.Page],
    pdf: Path,
    *,
    text: Callable[[Path], list[str]] = kvant.pdftext,
) -> dict[int, str]:
    """Which corpus page each PDF page of this issue is, the inverse of `align`.

    Through `align` and not through arithmetic, for the reason align exists: the
    front matter of a Kvant issue is not the same number of leaves in every year
    and the offset between a sheet's ordinal and its PDF page is -2 in some
    issues and 0 in most. Guessing it silently renders the wrong page.
    """
    mine = [page for page in pages if page.id.startswith(f"{issue}/")]
    return {number: page_id for page_id, number in kvant.align(mine, pdf, text=text).items()}


def draw(
    corpus: Path,
    store: Path,
    *,
    read: Callable[[Path], list[list[tuple]]] = boxes,
    text: Callable[[Path], list[str]] = kvant.pdftext,
    avoid: Callable[[], set[str]] = lambda: set(kvant.read_manifest("kvant-test")),
    say: Callable[[str], None] = lambda _m: None,
) -> tuple[list[str], dict[str, list[Run]]]:
    """The pages that carry rotated small type, and the runs on each.

    Every page that has one, not a sample. There are 63 of them, the set is
    meant to be run whole, and a sample of a set this size would put the
    number's error bar above the differences it is supposed to resolve.

    Minus the held out pages. Nine of the 72 pages the geometry finds are also
    in `kvant-test`, which is drawn at random over the same corpus and has no
    reason to miss them. Keeping them would make this set need a purpose gate,
    since a card that names a held out page in its dropped list has shown
    somebody a held out page. Nine pages is the cheaper thing to give up: a set
    of 63 that anyone can run today beats a set of 72 that only opens at a
    milestone, and what is lost is nine copies of a credit line that appears 63
    more times.
    """
    held = avoid()
    pages = kvant.pages(corpus)
    issues = sorted({page.issue for page in pages})
    where: dict[str, list[Run]] = {}
    for issue in issues:
        pdf = kvant.scan(issue, store)
        if pdf is None:
            continue
        try:
            here = runs(issue, pdf, read=read)
        except (OSError, subprocess.SubprocessError) as err:
            say(f"{issue}: {err}")
            continue
        if not here:
            continue
        sheet = sheets(issue, pages, pdf, text=text)
        for one in here:
            page_id = sheet.get(one.pdf_page)
            if page_id is None:
                say(f"{issue}: PDF page {one.pdf_page} is not one of this issue's corpus pages")
                continue
            if page_id in held:
                say(f"{page_id}: held out in kvant-test, left out of this set")
                continue
            where.setdefault(page_id, []).append(one)
    return sorted(where), where


def caught(reading: str, run: Run) -> bool:
    """Whether the reading carries this run's letters.

    Letters only, and folded. A reader that writes `Д. Гришуковой` where the page
    is set `Д.Гришуковой` has read the page; the space is the typesetting. A
    reader that writes one letter wrong has not, and this says so, because the
    whole point of the set is that a name off by a letter is a name nobody can
    look up.
    """
    want = LETTERS.sub("", fold(run.text))
    return bool(want) and want in LETTERS.sub("", fold(reading))


@dataclass(frozen=True)
class PageCard:
    """One page of the set, and which of its runs came back."""

    id: str
    caught: tuple[str, ...]
    lost: tuple[str, ...]
    failure: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.failure)

    @property
    def total(self) -> int:
        return len(self.caught) + len(self.lost)


def judge(page_id: str, reading: str, here: Sequence[Run]) -> PageCard:
    got = [one.text for one in here if caught(reading, one)]
    lost = [one.text for one in here if not caught(reading, one)]
    return PageCard(page_id, tuple(got), tuple(lost))


def missing(page_id: str, here: Sequence[Run]) -> PageCard:
    """A page with no reading, charged with every run on it.

    Not skipped, for the reason `bakeoff.missing` gives: a reader that refuses
    the pages it would have done badly on is the failure mode a set that drops
    its hard cases would reward.
    """
    return PageCard(
        page_id,
        (),
        tuple(one.text for one in here),
        failure="no reading in the directory",
    )


@dataclass
class Card:
    """What one reader did on the rotated set."""

    model: str
    pages: list[PageCard] = field(default_factory=list)

    @property
    def failures(self) -> list[PageCard]:
        return [page for page in self.pages if page.failed]

    def caught(self) -> int:
        return sum(len(page.caught) for page in self.pages)

    def total(self) -> int:
        return sum(page.total for page in self.pages)

    def recall(self) -> float:
        """Runs caught over runs on the set.

        Over runs and not over pages, because a page can carry one run or four
        and a page rate would let the four run pages count once.
        """
        total = self.total()
        return self.caught() / total if total else 0.0

    def clean(self) -> int:
        """Pages where every run came back."""
        return sum(1 for page in self.pages if page.total and not page.lost)

    def worst(self, n: int = 10) -> list[PageCard]:
        return sorted(self.pages, key=lambda page: (-len(page.lost), page.id))[:n]

    def to_dict(self) -> dict[str, object]:
        return {
            "set": "kvant-rotated",
            "model": self.model,
            "pages": len(self.pages),
            "failed": len(self.failures),
            "runs": self.total(),
            "caught": self.caught(),
            "recall": round(self.recall(), 6),
            "clean_pages": self.clean(),
            "worst": [
                {"page": page.id, "lost": list(page.lost), "caught": list(page.caught)}
                for page in self.worst()
                if page.lost
            ],
        }

    def to_markdown(self) -> str:
        out = [
            f"# {self.model} on kvant-rotated",
            "",
            f"{len(self.pages)} pages carrying {self.total()} runs of small type set on its "
            f"side, {len(self.failures)} of them with no reading at all. A page with no reading "
            "is charged with every run on it.",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Runs caught | {self.caught()} of {self.total()} |",
            f"| Recall | {self.recall():.1%} |",
            f"| Pages with every run caught | {self.clean()} of {len(self.pages)} |",
            "",
            "A run is caught when its letters appear in the reading, folded and with the "
            "punctuation dropped. Half a surname is not partial credit here, because a name off "
            "by a letter is a name nobody can look up.",
            "",
            "## Pages with a run missing",
            "",
        ]
        shown = [page for page in self.worst() if page.lost]
        if not shown:
            out.append("None.")
        for page in shown:
            lost = ", ".join(page.lost)
            out.append(f"- {page.id}: lost {lost}")
        return "\n".join(out) + "\n"


def score(readings: Path, where: dict[str, list[Run]], *, model: str) -> Card:
    """Score a directory of readings against the runs recorded for each page."""
    from local_ocr.bakeoff import find_reading

    card = Card(model=model)
    for page_id in sorted(where):
        here = where[page_id]
        path = find_reading(readings, page_id)
        if path is None:
            card.pages.append(missing(page_id, here))
            continue
        card.pages.append(judge(page_id, path.read_text(encoding="utf-8"), here))
    return card


def write(card: Card, *, json_path: Path | None, markdown_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(card.to_dict(), indent=2, ensure_ascii=False), "utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(card.to_markdown(), encoding="utf-8")


def manifest_lines(chosen: Sequence[str], found: Sequence[Run]) -> str:
    """The manifest file, page ids under a header saying how it was drawn."""
    tallest = max((one.ratio for one in found), default=0.0)
    head = [
        "# kvant-rotated, tier B, "
        f"{len(chosen)} pages carrying {len(found)} runs of small type set on its side.",
        "# Drawn by geometry rather than by text: a word whose box is at least "
        f"{RATIO:.0f} times taller than it is wide, no more than {WIDEST:.0f} points across, "
        f"at least {SHORTEST} characters long and holding Cyrillic.",
        f"# The most upended run in the set has a box {tallest:.1f} times taller than it is "
        "wide. The narrow side is not the type size; see the module docstring.",
        "# The runs themselves are the reference and they are in kvant-rotated.json. They come "
        "from the publisher's own text layer, the same as every other Kvant set here, so a "
        "number off this one is a distance to the file and not to the printed page.",
        "#",
    ]
    return "\n".join(head + list(chosen)) + "\n"


REFERENCE = kvant.MANIFESTS / "kvant-rotated.json"
"""The runs themselves, beside the page id manifest.

Two files rather than one because the reference is not a page id. Every other
Kvant set is a list of page ids and nothing else, because the corpus holds the
reference text for those pages; this set's reference is a handful of strings per
page that live in a PDF text layer, and the PDF cache is on one machine and is
146 files of a corpus that has 16 826 pages. Writing the strings down here makes
the set runnable anywhere, and makes it reviewable, which a set whose ground
truth is a subprocess on somebody else's disk is not.
"""


def reference(path: Path = REFERENCE) -> dict[str, list[Run]]:
    """The set's ground truth: which runs are on which page."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; draw it with `local-ocr kvant rotated --draw`")
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[Run]] = {}
    for page_id, items in raw["pages"].items():
        out[page_id] = [
            Run(
                issue=page_id.partition("/")[0],
                pdf_page=int(item["pdf_page"]),
                text=item["text"],
                width=float(item["width"]),
                height=float(item["height"]),
            )
            for item in items
        ]
    return out


def write_manifests(chosen: Sequence[str], where: dict[str, list[Run]]) -> list[Path]:
    """Write both halves of the set and say where they went."""
    found = [one for page_id in chosen for one in where.get(page_id, [])]
    text_path = kvant.MANIFESTS / "kvant-rotated.txt"
    text_path.write_text(manifest_lines(chosen, found), encoding="utf-8")
    body = {
        "set": "kvant-rotated",
        "ratio": RATIO,
        "widest": WIDEST,
        "shortest": SHORTEST,
        "pages": {
            page_id: [one.to_dict() for one in where[page_id]]
            for page_id in chosen
            if page_id in where
        },
    }
    REFERENCE.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return [text_path, REFERENCE]
