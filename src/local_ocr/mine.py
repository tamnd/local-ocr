"""Training candidates, out of the arguments the readers had.

§08 wants to fine tune a reader on the pages this corpus is hardest on, and the
hard part of a fine tune is never the training, it is the data. A supervised
pair needs an image and the right answer, and the right answer is exactly what
nobody has: if the corpus had it, none of this would be necessary.

The adjudicated disagreements are the closest thing to it that comes for free.
When two readers of different families disagree about a formula, and a third
look at a crop of that formula agrees with one of them, that is a labelled
example produced as a by product of reading the page. Nobody sat down to make
it. It is already paid for.

## What counts as a candidate, and what does not

A candidate needs three things and the third is the one that gets skipped.

It needs a verdict, so a disagreement nobody adjudicated is not a candidate.
It needs the losing reading, because the value of the example is not the right
answer alone, it is the pair: this is what the model produced and this is what
the page says. And it needs to be reproducible, which means the image hash, so
that the pair can be checked against the page six months later rather than
taken on trust.

What does not count, and this is the part worth being strict about:

A disagreement settled at the `illegible` rung is not a candidate. Nobody knows
what the page says there, and training on a guess dressed as a label is the one
way to make a model confidently wrong in a new place.

A disagreement where the primary won is a candidate for the *referee* and not
for the primary, and the two piles must not be mixed. Training reader A on
examples where reader A was already right teaches nothing and costs a step.
`for_reader` is what keeps them apart.

A structural disagreement is not a candidate here. The label would be a count,
and a count is not something a reader can be trained toward one page at a time.

## Why it reads sidecars rather than a database

Sidecars are written by the run that produced them, next to the page, and they
are the only artefact that certainly exists after the reader host is rebuilt.
A miner that reads them can be pointed at any directory of output, including one
copied off a host months later, and will produce the same candidates. That is
worth more here than a query language.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from local_ocr.sidecar import SUFFIX, Record, load

# Rungs whose verdicts are usable as labels. `illegible` is deliberately absent:
# see the module docstring. `budget` is absent because it means nothing was
# spent, so there is no verdict to use.
SETTLED = ("crop", "reread")

# Differences whose labels are text on the page. A structural difference is a
# count and has no text to train toward.
TRAINABLE = ("formula", "prose")


@dataclass(frozen=True)
class Candidate:
    """One labelled example, mined from one adjudicated disagreement."""

    page: str
    image_sha256: str
    for_reader: str
    """The reader this example would train, being the one that got it wrong."""

    wrong: str
    """What that reader produced."""

    right: str
    """What the adjudicator agreed the page says."""

    where: str
    what: str
    severity: str
    step: str
    """Which rung settled it. `reread` means it took a tighter crop, which is
    itself a signal: those are the genuinely hard ones."""

    evidence: str
    score: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def sidecars(root: Path) -> Iterator[Path]:
    """Every sidecar under a directory, in a stable order."""
    yield from sorted(p for p in root.rglob("*" + SUFFIX) if p.is_file())


def from_record(record: Record, names: tuple[str, str] = ("first", "second")) -> list[Candidate]:
    """The candidates in one page's record."""
    out: list[Candidate] = []
    for item in record.adjudicated:
        if item.step not in SETTLED:
            continue
        if item.where not in TRAINABLE:
            continue
        if item.winner == "second":
            loser, right, wrong = names[0], item.second, item.first
        elif item.winner == "first":
            loser, right, wrong = names[1], item.first, item.second
        else:
            continue
        if not right.strip() or not wrong.strip():
            # One side produced nothing, so the pair is "read the page" rather
            # than "read this the other way", and that is not what a fine tune
            # on formulas is for.
            continue
        out.append(
            Candidate(
                page=record.page,
                image_sha256=record.image_sha256,
                for_reader=loser,
                wrong=wrong,
                right=right,
                where=item.where,
                what=item.what,
                severity=item.severity,
                step=item.step,
                evidence=item.evidence,
                score=item.score,
            )
        )
    return out


def mine(root: Path) -> list[Candidate]:
    """Every candidate under a directory of output.

    A sidecar that cannot be parsed is skipped rather than fatal. The miner runs
    over months of output written by several versions and one truncated file
    should cost one page, not the run.
    """
    out: list[Candidate] = []
    for path in sidecars(root):
        try:
            record = load(path)
        except (OSError, ValueError):
            continue
        names = (
            record.first.reader if record.first else "first",
            record.second.reader if record.second else "second",
        )
        out.extend(from_record(record, names))
    return out


def counts(candidates: list[Candidate]) -> dict[str, dict[str, int]]:
    """The shape of a pile of candidates, which is what decides whether to train.

    Split by reader and by kind, because the question §08 has to answer is not
    how many examples there are, it is whether there are enough of one kind for
    one reader to be worth a training run. Two hundred candidates spread over
    two readers and three kinds is not a curriculum.
    """
    out: dict[str, dict[str, int]] = {}
    for one in candidates:
        by_reader = out.setdefault(one.for_reader, {})
        by_reader[one.where] = by_reader.get(one.where, 0) + 1
        by_reader["total"] = by_reader.get("total", 0) + 1
    return out


def to_jsonl(candidates: list[Candidate]) -> str:
    """One candidate per line, which is what every training harness reads."""
    return "".join(one.to_json() + "\n" for one in candidates)


def report(candidates: list[Candidate]) -> str:
    """The Markdown summary, for whoever has to decide whether this is enough."""
    shape = counts(candidates)
    made = f"{len(candidates)} mined from adjudicated disagreements."
    out = ["# Training candidates", "", made, ""]
    if not candidates:
        out.append(
            "None. Either no run has used a referee yet, or the two readers have "
            "not disagreed about anything a crop could settle."
        )
        return "\n".join(out) + "\n"
    out.append("| Reader | Formula | Prose | Total |")
    out.append("| --- | --- | --- | --- |")
    for reader in sorted(shape):
        row = shape[reader]
        formula, prose, total = row.get("formula", 0), row.get("prose", 0), row.get("total", 0)
        out.append(f"| {reader} | {formula} | {prose} | {total} |")
    out.append("")
    hard = [c for c in candidates if c.step == "reread"]
    out.append(
        f"{len(hard)} of them took a second, tighter crop to settle, which makes them "
        "the hardest examples in the pile and the ones worth looking at by hand first."
    )
    out.append("")
    return "\n".join(out) + "\n"
