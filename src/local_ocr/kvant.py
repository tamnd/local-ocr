"""A tier B set for Kvant, and why it is not the same tier B as Bourbaki's.

`tamnd/kvant` is a second corpus with a second cache, found through
`KVANT_CORPUS` and `KVANT_CACHE`, and nothing here copies a page out of either
one. What gets committed is a list of page ids, the same as for Bourbaki.

The tree is a different shape and that is the whole reason this is a module and
not four more branches inside `corpus.py`. A Bourbaki page is
`pages/<volume>/0042.md` and carries `book`, `pdf_page` and `method`. A Kvant
page is `content/ru/2018/10/pages/0016.md` and carries `issue`, `page_index`,
`page_label` and `extraction`. Threading both through one reader would mean a
dataclass where half the fields are empty on any given page, which is the shape
that makes a bug in one corpus reachable from the other.

The deeper difference is what the reference text actually is, and it changes
what a number off this set is allowed to claim.

Bourbaki tier B is a text extracted from the PDF by geometry, with no model in
the loop at any point. When it says a page reads `A VIII.25` the page reads
`A VIII.25`, and a disagreement is the reading's fault.

Kvant tier B is the publisher's own text layer. That sounds stronger and is
weaker, because it was produced by whatever typesetting pipeline the magazine
ran and it is only ever consistent with itself. Kvant measured this rather than
assumed it: 25 native pages read again by a model looking at the scan of the
same sheet, 18 597 words compared, and 33.9% of them were words the file does
not have in any spelling. Most of that is figures. A coordinate grid with
numbers in it is a picture in the file, so the text layer writes a figure marker
and the model reading the photograph types the numbers out. But not all of it
is: sheet 40 of the sixth issue of 2017 has `вэтой` for `в этой` and `рые` for
`некоторые`, which is the text layer losing a space and losing a prefix.

So a CER against this set is a distance to the publisher's file, not a distance
to the printed page, and it is quoted that way. What it is good for is the
comparison it was built for, which is one reader against another reader over the
same 200 pages.

## What is in the set

The corpus has 16 826 page files under `content/ru`. 2 063 of them carry
`extraction: native` and the other 14 763 are on the vision path. The native
pages run from 2007 to 2025 across 130 issues, and every one of those 130 is
entirely native: there is no issue where some pages have a text layer and some
do not, because the decision was taken per file and not per page.

That fact is what makes the exclusions the milestone asks for a no-op here, and
it is worth writing down rather than discovering later. The four born digital
files whose text is mojibake through a missing ToUnicode map, and the April 2023
file with no text at all, produced no native page at all. They are the four mid
run gaps in `manifests/issues.yaml` against the corpus, `kvant_2021_11-12`,
`kvant_2022_8`, `kvant_2024_5-6`, and `kvant_2023_4` which is the empty one. The
upstream routing already sent them to the vision lane, so drawing from
`extraction: native` excludes them by construction.

`usable` is therefore not doing the exclusion. It is checking that the exclusion
is still happening, which is a different job and the reason it stays. On today's
corpus it rejects 0 of 2 063. If the routing regresses, the set gets smaller and
`kvant check` says so, rather than the numbers quietly getting worse.

## The two thresholds, and the measurements that set them

The Cyrillic floor is 0.5 of the letters on the page. Across the 2 063 native
pages the measured minimum is 0.718, the first percentile is 0.870 and the
median is 0.985, so no real page is anywhere near the floor; the low end is
pages dense in Latin variables and not pages that are broken. A page mojibaked
through a missing ToUnicode map scores near zero, because every Cyrillic byte
comes out as a Latin or symbol codepoint. There is a factor of more than seven
between the two, which is the room the floor sits in.

The length floor is 200 characters. The measured minimum is 606 and the first
percentile is 1 039. A page whose text layer is empty scores under ten.

Neither number is tuned. They are both set an order of magnitude away from
anything the corpus contains, because a threshold placed near the data is a
threshold that will start rejecting real pages the first time the corpus grows.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from local_ocr import corpus as corpuslib
from local_ocr.golden import Burned, Purpose, _rank, math_density

CORPUS_ENV = "KVANT_CORPUS"
CACHE_ENV = "KVANT_CACHE"

MANIFESTS = Path(__file__).resolve().parent / "manifests"

CYRILLIC = re.compile(r"[Ѐ-ӿ]")
LATIN = re.compile(r"[A-Za-z]")

CYRILLIC_FLOOR = 0.5
"""Least share of the letters on the page that has to be Cyrillic."""

SHORTEST = 200
"""Fewest characters of body a page can have and still be a page."""

PDF_SHA = re.compile(r"^  sha256: ([0-9a-f]{64})$", re.M)
"""The digest of the issue's PDF, at the end of its page manifest.

Two spaces and not four. The sheets above it each carry their own keys at four,
and the pdf block at the bottom is the only one at two, so the anchor is the
indentation. A YAML library would be the honest way to read this and it would
also be a dependency this repository does not otherwise need, for one line.
"""


class NoKvant(Exception):
    """The Kvant corpus or its cache is not on this machine."""


def root(path: str | os.PathLike[str] | None = None) -> Path:
    raw = str(path) if path is not None else os.environ.get(CORPUS_ENV, "")
    if not raw:
        raise NoKvant(f"set {CORPUS_ENV} to a checkout of tamnd/kvant")
    found = Path(raw).expanduser()
    if not (found / "content" / "ru").is_dir():
        raise NoKvant(f"{found} has no content/ru in it")
    return found


def cache(path: str | os.PathLike[str] | None = None) -> Path:
    """Where the scans live, which is a separate tree from the corpus.

    The corpus is text and is a repository. The cache is nine point eight
    gigabytes of PDF in a content addressed store and is not, and a set that
    cannot be rendered is a set that cannot be scored, so the draw checks it.
    """
    raw = str(path) if path is not None else os.environ.get(CACHE_ENV, "")
    if not raw:
        raise NoKvant(f"set {CACHE_ENV} to the kvant scan cache")
    found = Path(raw).expanduser()
    if not (found / "blobs").is_dir():
        raise NoKvant(f"{found} has no blobs directory in it")
    return found


def available(corpus: str | os.PathLike[str] | None = None) -> bool:
    try:
        root(corpus)
    except NoKvant:
        return False
    return True


@dataclass(frozen=True)
class Page:
    """One page of one issue: where it came from and what the file says it is."""

    issue: str
    """`kvant_2018_10`, which is the key into the cache's page manifests."""
    year: int
    page_index: int
    """The sheet ordinal in the PDF, zero based, covers and inserts included."""
    page_label: str
    """The number printed on the paper, which is not the sheet ordinal."""
    extraction: str
    """native, vision or publisher. Which lane wrote this page."""
    body: str
    path: Path

    @property
    def id(self) -> str:
        """`kvant_2018_10/0016`, which is what the manifests record."""
        return f"{self.issue}/{self.page_index:04d}"

    @property
    def pdf_page(self) -> int:
        """The page number to hand a renderer, which counts from one.

        `page_index` is the ordinal in the manifest and starts at zero, so the
        cover is sheet 0 and PDF page 1. Off by one here would render the whole
        set one sheet early and every number off it would be noise.
        """
        return self.page_index + 1


def read_page(path: Path) -> Page:
    fields, body = corpuslib.parse_front_matter(path.read_text(encoding="utf-8"))
    issue = fields.get("issue", "")
    try:
        index = int(fields.get("page_index", path.stem))
    except ValueError as err:
        raise ValueError(f"{path}: page_index is not a number") from err
    try:
        year = int(fields.get("year", "0"))
    except ValueError:
        year = 0
    return Page(
        issue=issue,
        year=year,
        page_index=index,
        page_label=fields.get("page_label", "").strip('"'),
        extraction=fields.get("extraction", ""),
        body=body,
        path=path,
    )


def pages(corpus: Path) -> list[Page]:
    """Every page file in the Russian tree, in a stable order.

    Article files sit next to these under `articles/` and are deliberately not
    read. They are the same text cut a second way, by piece rather than by
    sheet, so counting both would put most of the corpus in twice.
    """
    out = [read_page(path) for path in sorted((corpus / "content" / "ru").rglob("pages/*.md"))]
    return sorted(out, key=lambda page: page.id)


def cyrillic_share(body: str) -> float:
    """The share of the letters on the page that are Cyrillic.

    Letters and not characters. A page of Russian is roughly half punctuation,
    digits and whitespace by character count, and dividing by all of it would
    make the number depend on how much mathematics the page happens to carry.
    """
    cyr = len(CYRILLIC.findall(body))
    latin = len(LATIN.findall(body))
    return cyr / (cyr + latin) if cyr + latin else 0.0


def usable(page: Page) -> str | None:
    """Why this page cannot be in the set, or None if it can be.

    A reason and not a boolean, because the whole point of the check is what it
    says when it starts rejecting things.
    """
    if page.extraction != "native":
        return f"on the {page.extraction or 'unknown'} lane, not the text layer"
    if len(page.body.strip()) < SHORTEST:
        return f"{len(page.body.strip())} characters of body, under {SHORTEST}"
    share = cyrillic_share(page.body)
    if share < CYRILLIC_FLOOR:
        return f"{share:.2f} of its letters are Cyrillic, under {CYRILLIC_FLOOR}"
    return None


def scan(issue: str, store: Path) -> Path | None:
    """The PDF this issue was scanned from, or None if it is not cached.

    Content addressed, so the path is the digest and the digest is in the
    issue's page manifest. Nothing here verifies the bytes: the store is
    written by kvant and re-hashing ten gigabytes to draw a set of 200 pages
    would cost more than the check is worth.
    """
    manifest_path = store / "pages" / f"{issue}.yaml"
    if not manifest_path.is_file():
        return None
    found = PDF_SHA.search(manifest_path.read_text(encoding="utf-8"))
    if not found:
        return None
    digest = found.group(1)
    blob = store / "blobs" / digest[:2] / digest[2:]
    return blob if blob.is_file() else None


@dataclass(frozen=True)
class Set:
    name: str
    size: int
    held_out: bool
    what: str


SETS: dict[str, Set] = {
    "kvant-dev": Set(
        "kvant-dev",
        200,
        False,
        "Stratified across the 130 issues that have a text layer, over "
        "sampling dense mathematics. Contaminated by construction, so its "
        "numbers are not reportable.",
    ),
    "kvant-test": Set(
        "kvant-test",
        200,
        True,
        "The same construction, disjoint, and not looked at. Run against it at "
        "a milestone and not between.",
    ),
}


def manifest(name: str) -> Path:
    if name not in SETS:
        known = ", ".join(sorted(SETS))
        raise KeyError(f"no Kvant set called {name!r}; there are {known}")
    return MANIFESTS / f"{name}.txt"


def read_manifest(name: str) -> list[str]:
    path = manifest(name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; draw it with `local-ocr kvant draw`")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.startswith("#")
    ]


def load(
    name: str,
    *,
    purpose: Purpose,
    corpus: Path | None = None,
) -> list[Page]:
    """The pages of a set, if the caller is allowed to have them.

    The same guard as `golden.load`, for the same reason. `kvant-test` is held
    out and a number off it that chooses a prompt burns it silently.
    """
    entry = SETS.get(name)
    if entry is None:
        known = ", ".join(sorted(SETS))
        raise KeyError(f"no Kvant set called {name!r}; there are {known}")
    if entry.held_out and purpose is not Purpose.MILESTONE:
        raise Burned(
            f"{name} is held out and this is a {purpose.value} path. "
            "A number from it that chooses a prompt or a checkpoint burns the "
            "set, and the burn is silent: it still has "
            f"{entry.size} pages in it and still produces a number. Use "
            "kvant-dev, or draw a new held out set and retire this one."
        )
    wanted = set(read_manifest(name))
    here = {page.id: page for page in pages(root(corpus))}
    missing = sorted(page_id for page_id in wanted if page_id not in here)
    if missing:
        raise NoKvant(
            f"{len(missing)} of {len(wanted)} pages are not in the corpus, "
            f"starting with {missing[0]}"
        )
    return [here[page_id] for page_id in sorted(wanted)]


@dataclass(frozen=True)
class Draw:
    dev: list[str]
    test: list[str]
    rejected: dict[str, str]
    """Page id to the reason `usable` gave, for everything the gate dropped."""


def _quota(sizes: dict[str, int], want: int) -> dict[str, int]:
    """Split `want` across the issues in proportion to how big each one is.

    Largest remainder, and this is the one place the construction deliberately
    departs from `golden.draw`. That one rounds each issue's share on its own,
    which is fine on Bourbaki where six volumes each get thirty odd pages and a
    half page of rounding is nothing. Kvant has 130 issues and 2 063 pages, so
    every issue's share is about 1.5 and every one of them rounds badly. Drawn
    that way the set came out at 175 of a nominal 200, four issues got nothing
    at all, and a manifest whose header says 200 and whose body has 175 lines
    is the kind of quiet shortfall that never gets noticed.

    So the floors are handed out first and the leftover goes to the issues with
    the largest fractional parts, which lands on `want` exactly whenever there
    are enough pages to land on.
    """
    total = sum(sizes.values())
    if not total or want <= 0:
        return dict.fromkeys(sizes, 0)
    exact = {issue: want * n / total for issue, n in sizes.items()}
    out = {issue: int(share) for issue, share in exact.items()}
    short = want - sum(out.values())
    order = sorted(sizes, key=lambda issue: (-(exact[issue] - out[issue]), issue))
    for issue in order[:short]:
        out[issue] += 1
    return out


def draw(corpus: Path, store: Path, dev_size: int = 200, test_size: int = 200) -> Draw:
    """Choose both sets from the corpus as it stands.

    Stratified by issue in proportion to how many native pages each contributes,
    and within an issue split three to two between the dense half and the rest,
    which is the construction `golden.draw` uses on Bourbaki. Keeping the two
    the same matters more than either being ideal: a difference between the two
    corpora's numbers should be the corpora and not the sampling. The one
    departure is how the per issue quotas are rounded, and `_quota` says why.
    """
    everything = pages(corpus)
    rejected: dict[str, str] = {}
    keep: list[Page] = []
    for page in everything:
        reason = usable(page)
        if reason is not None:
            if page.extraction == "native":
                # A page on the vision lane is not a rejection, it is the other
                # lane. Only a native page that failed the gate is news.
                rejected[page.id] = reason
            continue
        keep.append(page)

    uncached = {page.issue for page in keep if scan(page.issue, store) is None}
    for page in list(keep):
        if page.issue in uncached:
            rejected[page.id] = "the issue's PDF is not in the scan cache"
    keep = [page for page in keep if page.issue not in uncached]

    by_issue: dict[str, list[Page]] = {}
    for page in keep:
        by_issue.setdefault(page.issue, []).append(page)

    dev: list[str] = []
    test: list[str] = []
    sizes = {issue: len(here) for issue, here in by_issue.items()}
    dev_quota = _quota(sizes, dev_size)
    test_quota = _quota(sizes, test_size)
    for issue in sorted(by_issue):
        here = by_issue[issue]
        densities = sorted(math_density(page.body) for page in here)
        middle = densities[len(densities) // 2] if densities else 0.0
        dense = sorted(
            (page.id for page in here if math_density(page.body) >= middle),
            key=lambda page_id: _rank(page_id, "kvant-dense"),
        )
        plain = sorted(
            (page.id for page in here if math_density(page.body) < middle),
            key=lambda page_id: _rank(page_id, "kvant-plain"),
        )
        for want, into in ((dev_quota[issue], dev), (test_quota[issue], test)):
            taken_dense = min(len(dense), round(want * 0.6))
            taken_plain = min(len(plain), want - taken_dense)
            # Three to two is a preference and not a constraint. An issue whose
            # pages all score the same density has an empty plain half, because
            # every page is at or above the median, and taking only the dense
            # quota there loses the rest of the issue's allocation. On this
            # corpus that was nineteen pages of the two hundred. So whichever
            # half still has pages makes up the shortfall.
            taken_dense = min(len(dense), taken_dense + (want - taken_dense - taken_plain))
            taken_plain = min(len(plain), want - taken_dense)
            into.extend(dense[:taken_dense])
            into.extend(plain[:taken_plain])
            dense = dense[taken_dense:]
            plain = plain[taken_plain:]

    return Draw(sorted(dev), sorted(test), rejected)


@dataclass(frozen=True)
class Drift:
    name: str
    recorded: int
    would_draw: int
    gone: list[str]
    arrived: list[str]

    @property
    def steady(self) -> bool:
        return not self.gone and not self.arrived

    def line(self) -> str:
        if self.steady:
            return f"{self.name}: {self.recorded} pages, unchanged"
        return (
            f"{self.name}: {self.recorded} recorded, {self.would_draw} today, "
            f"{len(self.gone)} gone, {len(self.arrived)} new"
        )


def check(corpus: Path, store: Path) -> list[Drift]:
    """Compare both recorded sets against the corpus as it stands now."""
    drawn = draw(corpus, store)
    today = {"kvant-dev": drawn.dev, "kvant-test": drawn.test}
    here = {page.id for page in pages(corpus)}
    out: list[Drift] = []
    for name in SETS:
        recorded = read_manifest(name)
        now = today[name]
        out.append(
            Drift(
                name=name,
                recorded=len(recorded),
                would_draw=len(now),
                gone=sorted(page_id for page_id in recorded if page_id not in here),
                arrived=sorted(set(now) - set(recorded)),
            )
        )
    return out


def write_manifests(drawn: Draw) -> list[Path]:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, ids in (("kvant-dev", drawn.dev), ("kvant-test", drawn.test)):
        entry = SETS[name]
        header = [
            f"# {name}, tier B, {len(ids)} pages.",
            f"# {entry.what}",
            "#",
            "# The reference is the publisher's own text layer and not a",
            "# geometry extraction, so a CER against it is a distance to the",
            "# file rather than to the printed page. See local_ocr/kvant.py.",
            "#",
            "# Page ids into tamnd/kvant. Drawn once by `local-ocr kvant draw`",
            "# and checked by `local-ocr kvant check`. Do not edit by hand.",
        ]
        if entry.held_out:
            header.append("#")
            header.append("# HELD OUT. Reading these pages for anything other than a")
            header.append("# milestone report burns the set, and the burn is silent.")
        path = manifest(name)
        path.write_text("\n".join(header + ids) + "\n", encoding="utf-8")
        written.append(path)
    return written
