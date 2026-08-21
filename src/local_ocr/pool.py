"""The training pool, and the two sets it is not allowed to contain.

§08 wants a style LoRA trained on tier B, which is the six volumes with a usable
text layer. Those pages cost nothing to label: the text was lifted out of the
PDF by geometry with no model anywhere in the path, so for every one of them
there is an image and an answer that was never guessed. That is the whole
argument for training here rather than on mined disagreements, which are better
examples and there are two orders of magnitude fewer of them.

The pool is built by id, from the same page ids the golden sets are recorded in,
because the exclusion §08 asks for is an exclusion by id:

    golden-test    held out, and excluded here. Not for a checkpoint choice,
                   not for early stopping, not once.
    golden-hard    124 tier A pages, the only comparison against a person, and
                   the one set that cannot be regenerated. Never trained on.
    golden-dev     contaminated by construction, so it may be trained on, and
                   its numbers stop meaning anything the moment it is. That is
                   what it is for. The report says so out loud every time.

The exclusion reads `golden.read_manifest` and never `golden.load`. That is
deliberate and it is not a shortcut around the guard. `load` on the held out set
with a training purpose raises `Burned`, which is correct, and a builder that
had to catch that exception to do its job would be a builder that opens the door
it is supposed to be locking. Reading the ids of a held out set is harmless.
Reading its pages is not, and this module never asks for them.

## What is in a line, and what is deliberately not

An image path, a page id, and the completion. Not the prompt.

The prompt is 1 400 identical tokens on every example, §08 trains on the
completion only for exactly that reason, and which prompt a run uses is a choice
made at training time and not at build time. Keeping it out means a prompt
change does not mean a pool rebuild, and it means the file can be read by a
person, which a file with 2 400 copies of the same 1 400 tokens cannot.

The completion is `evaluate.conformance_reference`, which is the running head
put back on the front of the body. It has to be the head, because half the house
rules a style LoRA exists to teach are about the first line of the page, and the
corpus files the head in the front matter rather than in the body. It has to be
the head in the right order, which is why `evaluate.head_line` learned to read
the page number before this module was written.

## The validation split, and why it is carved out here

Early stopping needs a loss on pages the run did not train on, and §08 forbids
using `golden-test` for it. So a slice of the pool is held back at build time,
chosen by a digest of the page id so that two builds on two machines choose the
same pages, and marked in the file. Nothing downstream has to remember to do it,
and a training script that ignores the field is a training script that trains on
its own validation set, which is at least visible in the file it read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_ocr import corpus as corpuslib
from local_ocr import evaluate, golden, pageimages

# The sets whose pages must not appear in the pool. Named rather than derived
# from `held_out`, because `golden-hard` is not held out in the sense the guard
# on `load` means and is still not trainable: it is the only comparison against
# a person and it cannot be drawn again.
FORBIDDEN = ("golden-test", "golden-hard")

# One page in twenty, held back for the loss curve. Small because the pool is
# small and the thing being measured is a style adjustment, so a hundred pages
# is enough to see a checkpoint get worse.
VALIDATION = 0.05


class Contaminated(Exception):
    """A forbidden page id reached the pool. The build stops rather than warns."""


@dataclass(frozen=True)
class Example:
    """One page, as a training harness wants it."""

    id: str
    book: str
    image: str
    completion: str
    split: str
    """`train` or `val`. See the module docstring: the validation slice is
    carved out here so that early stopping never reaches for `golden-test`."""

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "book": self.book,
                "image": self.image,
                "completion": self.completion,
                "split": self.split,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


@dataclass
class Pool:
    """What a build produced, and what it left behind and why."""

    examples: list[Example] = field(default_factory=list)
    excluded: dict[str, int] = field(default_factory=dict)
    """How many pages each forbidden set kept out, by set name."""

    no_image: list[str] = field(default_factory=list)
    """Tier B pages with no rendered image on disk. `local-ocr pages` makes them.

    Listed whether or not they were dropped, so that `--all-pages` reports the
    rendering it is deferring rather than reporting a clean zero."""

    empty: list[str] = field(default_factory=list)
    """Pages whose body is blank, which are plates and dividers."""

    dev: int = 0
    """How many of the examples are `golden-dev` pages. Reported, not excluded."""

    @property
    def train(self) -> list[Example]:
        return [one for one in self.examples if one.split == "train"]

    @property
    def val(self) -> list[Example]:
        return [one for one in self.examples if one.split == "val"]

    @property
    def books(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for one in self.examples:
            out[one.book] = out.get(one.book, 0) + 1
        return out


def _split(page_id: str, fraction: float) -> str:
    """Which side of the validation line a page falls, decided by its id.

    A digest and not a random number generator, for the reason `golden.draw`
    gives: a draw has to come out the same on another machine and in another
    year, and the seeding of an interpreter's generator is a detail of the
    interpreter.
    """
    if fraction <= 0:
        return "train"
    digest = hashlib.sha256(f"pool\x00{page_id}".encode()).digest()
    cut = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if cut < fraction else "train"


def forbidden_ids(sets: Sequence[str] = FORBIDDEN) -> set[str]:
    """Every page id that must not be trained on."""
    out: set[str] = set()
    for name in sets:
        out.update(golden.read_manifest(name))
    return out


def build(
    corpus: Path | None = None,
    *,
    books: Sequence[str] = golden.TIER_B_VOLUMES,
    validation: float = VALIDATION,
    require_image: bool = True,
) -> Pool:
    """The pool, from the corpus as it stands.

    `require_image` is on by default and is what makes the count honest: a page
    with no image is not a training example, it is a rendering job, and counting
    it would report a pool larger than the one a trainer could read. The pages
    it drops are listed so `local-ocr pool` can say how to make them.
    """
    root = corpuslib.root(corpus)
    held = {name: set(golden.read_manifest(name)) for name in FORBIDDEN}
    banned = set().union(*held.values()) if held else set()
    dev = set(golden.read_manifest("golden-dev"))
    out = Pool()
    counted: dict[str, int] = {}
    for page in corpuslib.pages(root, list(books)):
        if page.method != "native":
            continue
        page_id = page.id
        if page_id in banned:
            for name, ids in held.items():
                if page_id in ids:
                    counted[name] = counted.get(name, 0) + 1
            continue
        if not page.body.strip():
            out.empty.append(page_id)
            continue
        image = pageimages.image_path(root, page.book, page.pdf_page)
        if not image.exists():
            out.no_image.append(page_id)
            if require_image:
                continue
        if page_id in dev:
            out.dev += 1
        out.examples.append(
            Example(
                id=page_id,
                book=page.book,
                image=str(image),
                completion=evaluate.conformance_reference(page),
                split=_split(page_id, validation),
            )
        )
    out.excluded = {name: counted.get(name, 0) for name in FORBIDDEN}
    check(out)
    return out


def check(pool: Pool, sets: Sequence[str] = FORBIDDEN) -> None:
    """Assert what the module exists to guarantee, on the way out of `build`.

    Cheap, and it runs on every build rather than only in the test, because the
    failure it catches is silent everywhere else: a pool with sixteen held out
    pages in it trains fine, converges fine, and reports a number off
    `golden-test` that is a measure of how well the model memorised those pages.
    """
    banned = forbidden_ids(sets)
    leaked = sorted(one.id for one in pool.examples if one.id in banned)
    if leaked:
        raise Contaminated(
            f"{len(leaked)} held out pages reached the training pool, "
            f"first {leaked[0]}. {', '.join(sets)} are never trained on."
        )


def to_jsonl(pool: Pool) -> str:
    """One example per line, which is what every training harness reads."""
    return "".join(one.to_json() + "\n" for one in pool.examples)


def report(pool: Pool) -> str:
    """The Markdown summary, for whoever has to decide whether this is enough."""
    out = ["# Training pool", ""]
    out.append(
        f"{len(pool.examples)} pages, {len(pool.train)} to train on and "
        f"{len(pool.val)} held back for the loss curve."
    )
    out.append("")
    if not pool.examples:
        out.append(
            f"Empty, with {len(pool.no_image)} tier B pages waiting on a render. "
            "`local-ocr pages` writes the images, and `--all-pages` counts what "
            "the pool would hold once they are there."
            if pool.no_image
            else "Empty. The corpus has no native extraction on these volumes."
        )
        return "\n".join(out) + "\n"
    out.append("| Volume | Pages |")
    out.append("| --- | --- |")
    for book in sorted(pool.books):
        out.append(f"| {book} | {pool.books[book]} |")
    out.append("")
    out.append("## What was left out")
    out.append("")
    out.append("| Set | Pages kept out |")
    out.append("| --- | --- |")
    for name in FORBIDDEN:
        out.append(f"| {name} | {pool.excluded.get(name, 0)} |")
    out.append("")
    out.append(
        f"{len(pool.empty)} pages carry no body text and {len(pool.no_image)} have no "
        "rendered image. The second number is a rendering job and not a shortage of "
        "pages: `local-ocr pages` writes them into the corpus images tree."
    )
    hard = set(golden.read_manifest("golden-hard"))
    with_dev = len(hard & set(golden.read_manifest("golden-dev")))
    with_test = len(hard & set(golden.read_manifest("golden-test")))
    out.append("")
    out.append(
        "`golden-hard` is a predicate over the pages a person had to read and not a "
        f"draw, so it overlaps the sets that were drawn: {with_dev} of its pages are "
        f"`golden-dev` pages and {with_test} are `golden-test` pages. That is why it "
        "is excluded by name rather than left to the tier B filter, which would have "
        f"let those {with_dev} through."
    )
    out.append("")
    out.append(
        f"{pool.dev} of the pages are `golden-dev` pages. §08 allows that and this is "
        "the sentence that says it happened: any number reported off `golden-dev` "
        "after an adapter trained on this pool is a number about the training set."
    )
    return "\n".join(out) + "\n"
