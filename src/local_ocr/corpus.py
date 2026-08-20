"""Reading the corpus, without depending on it.

`tamnd/bourbaki` is a checkout somewhere else on the machine, found through
`BOURBAKI_CORPUS`. Nothing in this repository copies pages out of it: the
volumes are somebody's copyright and this repository is MIT, so what gets
committed here is a list of page ids and never a page.

That has a consequence worth stating. Everything that needs page text is
therefore skipped when the corpus is not present, including in CI. The parity
tests in `tests/test_go_parity.py` are the ones that always run, and they were
written to need nothing but their fixtures for exactly this reason.

The front matter is parsed rather than loaded with a YAML library, and it is
worth saying exactly what that buys and what it costs. Most of it is flat
scalars, but `locator:` nests two keys under it and `flags:` is a list, and this
parser flattens the first and turns the second into keys that are not keys. That
is harmless only because nothing here reads either of them, and it is written
down rather than left to be discovered: anything that later wants the locator
has to fix this first, not build on top of it.

The running head is in the front matter and not in the body. `page_label`,
`running_head` and `folio` carry it, put there by `extract`, and the body begins
below it. That is the single most consequential fact in this file. A model reads
the printed page, so its transcription starts with the head; the reference does
not have one; and a harness that compared the two as they stand would charge
every reading forty characters of error on every page for doing exactly what the
prompt tells it to do.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CORPUS_ENV = "BOURBAKI_CORPUS"


class NoCorpus(Exception):
    """The corpus is not on this machine, or not where it said it was."""


def root(path: str | os.PathLike[str] | None = None) -> Path:
    """Where the corpus is, from the argument or from the environment."""
    raw = str(path) if path is not None else os.environ.get(CORPUS_ENV, "")
    if not raw:
        raise NoCorpus(f"set {CORPUS_ENV} to a checkout of tamnd/bourbaki")
    found = Path(raw).expanduser()
    if not (found / "pages").is_dir():
        raise NoCorpus(f"{found} has no pages directory in it")
    return found


def available(path: str | os.PathLike[str] | None = None) -> bool:
    """Whether the corpus can be read, for a test that has to skip without it."""
    try:
        root(path)
    except NoCorpus:
        return False
    return True


@dataclass(frozen=True)
class Page:
    """One page file: its id, its front matter and its body."""

    book: str
    pdf_page: int
    method: str
    """native, ocr or blank. Which tier of ground truth this page is."""
    manual: bool
    """A person read the printed page and stands behind the text."""
    body: str
    path: Path
    page_label: str = ""
    """`A VIII.25`, the Bourbaki page reference printed in the running head."""
    running_head: str = ""
    """The title half of the head, which the facing page carries instead."""
    folio: str = ""
    """The ordinary page number, where the volume prints one."""

    @property
    def has_head(self) -> bool:
        """Whether this page prints a running head at all.

        A chapter opening, a part title and the front matter do not, and rule 4
        has to stand down on those or it rejects a correct reading.
        """
        return bool(self.page_label or self.running_head or self.folio)

    @property
    def id(self) -> str:
        """`alg-viii/0042`, which is what the golden manifests record."""
        return f"{self.book}/{self.pdf_page:04d}"


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a page file into its front matter and its body."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 3)
    if end < 0:
        return {}, text
    fields: dict[str, str] = {}
    for line in text[4:end].split("\n"):
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields, text[end + 5 :].lstrip("\n")


def read_page(path: Path) -> Page:
    fields, body = parse_front_matter(path.read_text(encoding="utf-8"))
    try:
        pdf_page = int(fields.get("pdf_page", path.stem))
    except ValueError as err:
        raise ValueError(f"{path}: pdf_page is not a number") from err
    return Page(
        book=fields.get("book", path.parent.name),
        pdf_page=pdf_page,
        method=fields.get("method", ""),
        manual=fields.get("manual", "") == "true",
        body=body,
        path=path,
        page_label=fields.get("page_label", ""),
        running_head=fields.get("running_head", ""),
        folio=fields.get("folio", ""),
    )


def pages(corpus: Path, books: list[str] | None = None) -> list[Page]:
    """Every page file, in a stable order, optionally from named volumes only."""
    out: list[Page] = []
    for book in sorted((corpus / "pages").iterdir()):
        if not book.is_dir() or (books is not None and book.name not in books):
            continue
        for path in sorted(book.glob("*.md")):
            out.append(read_page(path))
    return out


def page_ids(corpus: Path, ids: list[str]) -> list[Page]:
    """Read the named pages, and say plainly which of them are not there.

    A golden set is a list of ids that was drawn once. If a page has since been
    renamed or removed the set is measuring something other than what it was
    drawn to measure, and quietly returning 198 pages for a set of 200 is the
    kind of silence that makes a benchmark flattering.
    """
    found: list[Page] = []
    missing: list[str] = []
    for page_id in ids:
        book, _, number = page_id.partition("/")
        path = corpus / "pages" / book / f"{number}.md"
        if not path.is_file():
            missing.append(page_id)
            continue
        found.append(read_page(path))
    if missing:
        raise NoCorpus(
            f"{len(missing)} of {len(ids)} pages are not in the corpus, starting with {missing[0]}"
        )
    return found
