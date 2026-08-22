"""The four golden sets, and the one that must not be looked at.

Four fixed sets, chosen once, recorded by page id in `manifests/`, never changed
casually. What is committed is the list of ids and a note saying how the list
was arrived at. The pages themselves stay in the corpus.

    golden-dev         200 pages, tier B, contaminated by construction
    golden-test        200 pages, tier B, disjoint from dev, not looked at
    golden-hard        124 pages, tier A, every page a person had to read
    golden-incumbent   496 pages, tier C, what the fleet's frontier model did

`golden-test` is the one with machinery around it. A number from it that is used
to choose a prompt or a checkpoint burns the set, and the burn is silent: the
set still has 200 pages in it and still produces a number, and that number is
now a measure of how well the thing was fitted to those 200 pages. Nothing about
the file changes. So the guard is here, at the only door into it, and it asks
what the caller is going to do with the pages before it hands them over.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from local_ocr import corpus as corpuslib
from local_ocr.rules import mathtex

# Package data rather than a directory at the top of the repository, so that a
# set is readable from an installed wheel and not only from a source checkout.
MANIFESTS = Path(__file__).resolve().parent / "manifests"

# The six volumes with a usable text layer, extracted from the PDF by geometry
# with no model involved at any point. They are what makes this project
# measurable: for every one of these pages there is an image and a text, and the
# text was never guessed.
TIER_B_VOLUMES = (
    "alg-viii",
    "alg-viii-fr",
    "lie-vii-ix",
    "ta-i-iv-fr",
    "ts-i-ii-fr",
    "ts-iii-v-fr",
)


class Purpose(Enum):
    """What the caller is going to do with the pages.

    Not decoration. It is the argument the guard on `golden-test` reads.
    """

    DEVELOPMENT = "development"
    """Looking at pages, choosing a prompt, comparing two settings."""

    TRAINING = "training"
    """Building a fine tuning set or an evaluation loop inside one."""

    MILESTONE = "milestone"
    """Reporting a number at a milestone, having decided everything already."""


class Burned(Exception):
    """Something tried to read the held out set for a purpose that burns it."""


@dataclass(frozen=True)
class GoldenSet:
    name: str
    tier: str
    size: int
    held_out: bool
    what: str


SETS: dict[str, GoldenSet] = {
    "golden-dev": GoldenSet(
        "golden-dev",
        "B",
        200,
        False,
        "Stratified across the six native volumes, French and English, over "
        "sampling dense mathematics. Contaminated by construction, so its "
        "numbers are not reportable.",
    ),
    "golden-test": GoldenSet(
        "golden-test",
        "B",
        200,
        True,
        "The same construction, disjoint, and not looked at. Run against it at "
        "a milestone and not between.",
    ),
    "golden-hard": GoldenSet(
        "golden-hard",
        "A",
        124,
        False,
        "Every page a person had to read by hand, which means every page that "
        "defeated everything else. The useful output here is not a percentage "
        "but a list of what the model still cannot do.",
    ),
    # 496 and not the 495 the spec was written against. One page was read by
    # the fleet between the two, which is what a set defined by a predicate over
    # a live corpus does. It is frozen at 496 here, and `golden check` is what
    # says so out loud the next time the corpus moves under it.
    "golden-incumbent": GoldenSet(
        "golden-incumbent",
        "C",
        496,
        False,
        "Read by the fleet's frontier model and accepted by the eight rules. "
        "No ground truth, so the comparison is three way and the residue goes "
        "to the printed page.",
    ),
}


def manifest(name: str) -> Path:
    if name not in SETS:
        known = ", ".join(sorted(SETS))
        raise KeyError(f"no golden set called {name!r}; there are {known}")
    return MANIFESTS / f"{name}.txt"


def read_manifest(name: str) -> list[str]:
    """The page ids of a set, straight off disk, with no guard on them.

    Private on purpose: reading the ids of the held out set is harmless and
    reading its pages is not, so the guard sits on `load` and not here.
    """
    path = manifest(name)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; draw it with `local-ocr golden draw`")
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
) -> list[corpuslib.Page]:
    """The pages of a set, if the caller is allowed to have them.

    `purpose` is required and has no default. A default would be chosen once, by
    whoever wrote the next caller, and it would be the permissive one.
    """
    entry = SETS.get(name)
    if entry is None:
        known = ", ".join(sorted(SETS))
        raise KeyError(f"no golden set called {name!r}; there are {known}")
    if entry.held_out and purpose is not Purpose.MILESTONE:
        raise Burned(
            f"{name} is held out and this is a {purpose.value} path. "
            "A number from it that chooses a prompt or a checkpoint burns the "
            "set, and the burn is silent: it still has "
            f"{entry.size} pages in it and still produces a number. Use "
            "golden-dev, or draw a new held out set and retire this one."
        )
    return corpuslib.page_ids(corpuslib.root(corpus), read_manifest(name))


def math_density(body: str) -> float:
    """The fraction of a page that sits inside a math span.

    This is what "dense mathematics" means when the draw over samples it. A page
    of running prose with three inline spans scores near zero; a page of
    displays scores high, and those are the pages where a model that reads text
    well still reads the page wrongly.
    """
    if not body:
        return 0.0
    marked = mathtex.in_math(body)
    return sum(marked) / len(marked)


def _rank(page_id: str, salt: str) -> str:
    """A stable shuffle key.

    A draw has to be reproducible on another machine and in another year, so it
    cannot come from a random number generator whose seeding is a detail of the
    interpreter. A digest of the page id is the same everywhere forever, and
    `local-ocr golden draw --check` re-derives the whole draw and compares.
    """
    return hashlib.sha256(f"{salt}\x00{page_id}".encode()).hexdigest()


@dataclass(frozen=True)
class Draw:
    dev: list[str]
    test: list[str]
    hard: list[str]
    incumbent: list[str]


def draw(corpus: Path, dev_size: int = 200, test_size: int = 200) -> Draw:
    """Choose all four sets from the corpus as it stands.

    Stratified by volume in proportion to how many pages each contributes, and
    within a volume split three to two between the dense half of its pages and
    the rest. Dev and test are taken from the same ordering, dev first and test
    immediately after, so they are disjoint by construction rather than by a
    check afterwards.
    """
    everything = corpuslib.pages(corpus)
    hard = sorted(page.id for page in everything if page.manual)
    incumbent = sorted(page.id for page in everything if page.method == "ocr")

    native = [
        page
        for page in everything
        if page.method == "native" and page.book in TIER_B_VOLUMES and page.body.strip()
    ]
    by_volume: dict[str, list[corpuslib.Page]] = {}
    for page in native:
        by_volume.setdefault(page.book, []).append(page)

    dev: list[str] = []
    test: list[str] = []
    total = len(native)
    for book in sorted(by_volume):
        pages_here = by_volume[book]
        share = len(pages_here) / total if total else 0.0
        want_dev = round(dev_size * share)
        want_test = round(test_size * share)
        densities = sorted(math_density(page.body) for page in pages_here)
        middle = densities[len(densities) // 2] if densities else 0.0
        dense = sorted(
            (page.id for page in pages_here if math_density(page.body) >= middle),
            key=lambda page_id: _rank(page_id, "dense"),
        )
        plain = sorted(
            (page.id for page in pages_here if math_density(page.body) < middle),
            key=lambda page_id: _rank(page_id, "plain"),
        )
        # Three parts dense to two parts ordinary. The point of the set is to
        # find out where a model breaks, and it breaks on the mathematics.
        for want, into in ((want_dev, dev), (want_test, test)):
            taken_dense = min(len(dense), round(want * 0.6))
            taken_plain = min(len(plain), want - taken_dense)
            into.extend(dense[:taken_dense])
            into.extend(plain[:taken_plain])
            dense = dense[taken_dense:]
            plain = plain[taken_plain:]

    return Draw(sorted(dev), sorted(test), hard, incumbent)


def in_tier(page: corpuslib.Page, tier: str) -> bool:
    """Whether a page still meets the predicate the set was drawn on.

    The predicates are `draw`'s, lifted out so that a set can be asked the
    question again afterwards, which is the part that was missing. A set is a
    list of ids frozen once and the pages under those ids go on being read: a
    tier B page whose text layer turned out to have dropped a glyph gets read by
    the fleet instead, and the id does not change when it does. What changes is
    that the reference stopped being an extraction nobody guessed at and became
    a model's reading, and a number measured against that is one model marking
    another's homework.
    """
    if tier == "A":
        return page.manual
    if tier == "B":
        return page.method == "native" and page.book in TIER_B_VOLUMES and bool(page.body.strip())
    if tier == "C":
        return page.method == "ocr"
    return True


def stale(name: str, pages: Sequence[corpuslib.Page]) -> list[corpuslib.Page]:
    """The pages of a set whose reference has left the tier the set was drawn on.

    A page a person has since read by hand is never stale, whatever its method
    field says. The tier is about where the reference came from and a person is
    the best source there is, so a hand read page is an improvement on any tier
    and not a departure from one.
    """
    tier = SETS[name].tier
    return [page for page in pages if not page.manual and not in_tier(page, tier)]


@dataclass(frozen=True)
class Drift:
    """How a recorded set differs from what the same draw would choose today."""

    name: str
    recorded: int
    would_draw: int
    gone: list[str]
    """Ids in the manifest that are no longer pages in the corpus."""
    arrived: list[str]
    """Ids the draw would choose now and did not choose then."""
    left: list[str] = field(default_factory=list)
    """Ids still in the set whose page no longer meets the set's tier.

    Worse than the other two and easy to miss next to them. A page that has gone
    leaves a hole somebody notices; a page that has arrived is a set that has
    stopped being a fair draw. A page that has quietly changed tier is still
    there, still the right count, and its reference is now a guess.
    """

    @property
    def steady(self) -> bool:
        return not self.gone and not self.arrived and not self.left

    def line(self) -> str:
        if self.steady:
            return f"{self.name}: {self.recorded} pages, unchanged"
        line = (
            f"{self.name}: {self.recorded} recorded, {self.would_draw} today, "
            f"{len(self.gone)} gone, {len(self.arrived)} new"
        )
        if self.left:
            tier = SETS[self.name].tier
            line += f", {len(self.left)} no longer tier {tier}"
        return line


def check(corpus: Path) -> list[Drift]:
    """Compare every recorded set against the corpus as it stands now.

    A set is frozen deliberately, so drift is not an error. It is a fact, and
    the reason to print it is that the two sets that are defined by a predicate,
    hard and incumbent, grow every time somebody reads a page by hand or the
    fleet accepts one. A number reported against a set that has stopped
    describing what its name says it describes is worse than no number, because
    the name is what anybody reads.
    """
    drawn = draw(corpus)
    today = {
        "golden-dev": drawn.dev,
        "golden-test": drawn.test,
        "golden-hard": drawn.hard,
        "golden-incumbent": drawn.incumbent,
    }
    by_id = {page.id: page for page in corpuslib.pages(corpus)}
    out: list[Drift] = []
    for name in SETS:
        recorded = read_manifest(name)
        now = today[name]
        still_here = [by_id[page_id] for page_id in recorded if page_id in by_id]
        out.append(
            Drift(
                name=name,
                recorded=len(recorded),
                would_draw=len(now),
                gone=sorted(page_id for page_id in recorded if page_id not in by_id),
                arrived=sorted(set(now) - set(recorded)),
                left=sorted(page.id for page in stale(name, still_here)),
            )
        )
    return out


def write_manifests(drawn: Draw) -> list[Path]:
    """Record a draw, with a header saying what it is and how to check it."""
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, ids in (
        ("golden-dev", drawn.dev),
        ("golden-test", drawn.test),
        ("golden-hard", drawn.hard),
        ("golden-incumbent", drawn.incumbent),
    ):
        entry = SETS[name]
        header = [
            f"# {name}, tier {entry.tier}, {len(ids)} pages.",
            f"# {entry.what}",
            "#",
            "# Page ids into tamnd/bourbaki. Drawn once by `local-ocr golden draw`",
            "# and checked by `local-ocr golden check`. Do not edit by hand.",
        ]
        if entry.held_out:
            header.append("#")
            header.append("# HELD OUT. Reading these pages for anything other than a")
            header.append("# milestone report burns the set, and the burn is silent.")
        path = manifest(name)
        path.write_text("\n".join(header + ids) + "\n", encoding="utf-8")
        written.append(path)
    return written
