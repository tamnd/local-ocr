"""The Russian bake off: readings in, a card out, and a ranking across cards.

    local-ocr kvant eval --readings out/reader-a --model reader-a

M8 item 5 asks for the §02 shortlist re evaluated from scratch for Russian
rather than inherited from the Bourbaki choice, and this is the instrument that
does it. It is deliberately not `local-ocr eval`. That harness judges against a
Bourbaki golden set, where the reference is a human checked page and eight
acceptance rules apply to the reading. Neither holds here. The Kvant reference
is the publisher's own text layer, which is machine output that nobody checked,
and the Bourbaki acceptance rules are about a French mathematics monograph and
say nothing useful about a Russian magazine.

The one thing that makes this harder than it looks is that the reference is
wrong in a known way. §07 measured it: on a two column page the text layer
follows the order the drawing operations sit in the file rather than the order a
person reads in, so it prints the first problem, jumps to the top of the right
column for the tail of the third, and comes back. The vision path gets that
right. So a straight character error rate against the reference punishes the
reading that is correct, by roughly the length of a column, and the better a
reader is at columns the worse its number looks.

The answer here is to match blocks first and score inside them. Blocks are
paired by shared vocabulary, order free, and the character rate is computed pair
by pair; the order question is then answered separately by a Kendall tau over
where those same pairs sit. The two use one matching function so they cannot
disagree about which block is which.

A pair is a passage and not a block, because the text layer disagrees with the
reading about where a block ends as well as about what order the blocks go in.
It shatters a display heading into one block a word and it fuses a column of
five paragraphs into one block, both on the same page, so the matching has to
put several blocks of one side against several of the other and score the
joined text. `metrics.order._groups` is where that happens and its docstring
carries the measurement: on 198 real Kvant pages, doing this rather than
matching block to block moved content CER from 57.3 % to 27.4 % without one
character of any reading changing.

That leaves the obvious hole, which is that a reader could emit one perfect
paragraph, drop the rest of the page and score zero error on what it matched. So
the headline number is not the matched rate. It is `content`, which charges
every character of every reference block that nothing matched as an error. A
reader that reads a tenth of the page gets a content rate near 0.9 and the
matched rate is reported next to it as the diagnostic it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from local_ocr import kvant, russian
from local_ocr.metrics import cer
from local_ocr.metrics import order as orderlib
from local_ocr.rules import textguard


def find_reading(directory: Path, page_id: str) -> Path | None:
    """Where a reading of one Kvant page might be.

    The same several shapes `evaluate.find_reading` allows, for the same reason:
    `ocr-batch` writes one file per image and the image names come from
    `kvant pages`, which flattens `kvant_2018_10/0016` to `kvant_2018_10-0016`.
    A separate function rather than a shared one because the id here splits on
    the issue and not on a book, and the two ids are not interchangeable.
    """
    issue, _, number = page_id.partition("/")
    for candidate in (
        directory / issue / f"{number}.md",
        directory / issue / f"{number}.txt",
        directory / f"{issue}-{number}.md",
        directory / f"{issue}_{number}.md",
        directory / f"{number}.md",
    ):
        if candidate.is_file():
            return candidate
    return None


@dataclass(frozen=True)
class PageCard:
    """One Russian page judged, every number on it."""

    id: str
    matched: cer.Score
    """Edits inside paired blocks, over the reference characters in them."""
    content: cer.Score
    """The same, plus every character of every reference block nothing matched.

    The headline. A block the reading never produced is charged in full, which
    is what stops a reader that quietly drops half a page from outscoring one
    that reads all of it with a few errors in each paragraph.
    """
    prose: cer.Score
    """`content`, with the mathematics cut out of both sides first.

    Reported beside it because the two answer different questions. A Russian
    page of physics is a third mathematics by character, and a reader that is
    good at Cyrillic and poor at LaTeX and one that is the reverse can land on
    the same content rate by different routes.
    """
    order: orderlib.Order
    oov: int
    """Out of vocabulary words the page repeats, which is the §07 gate."""
    homoglyphs: int
    """Cyrillic lookalikes standing where a Latin variable belongs."""
    failure: str = ""
    """Why there was no reading at all, empty when there was one."""

    @property
    def failed(self) -> bool:
        return bool(self.failure)


def judge(page: kvant.Page, read: str, words: set[str]) -> PageCard:
    """One page against its text layer."""
    want = textguard.normalise(page.body)
    got = textguard.normalise(read)
    pairs, dropped, _ = orderlib.paired(got, want)

    edits = 0
    inside = 0
    for mine, theirs in pairs:
        whole, _ = cer.page(theirs, mine)
        edits += whole.edits
        inside += whole.length
    missed = sum(len(textguard.normalise(block).strip()) for block in dropped)

    # Prose is computed on the joined blocks rather than block by block, because
    # cutting the mathematics out of a block can leave it empty and a page of
    # empty denominators is not a rate.
    joined_read = "\n\n".join(mine for mine, _ in pairs)
    joined_want = "\n\n".join(theirs for _, theirs in pairs) + "\n\n" + "\n\n".join(dropped)
    _, prose = cer.page(joined_want, joined_read)

    return PageCard(
        id=page.id,
        matched=cer.Score(edits, inside),
        content=cer.Score(edits + missed, inside + missed),
        prose=prose,
        order=orderlib.order(got, want),
        oov=len(russian.oov(read, words)),
        homoglyphs=len(russian.homoglyphs(read)),
    )


def missing(page: kvant.Page) -> PageCard:
    """A page with no reading, judged as the empty string.

    Not skipped. §05 is blunt about why: a benchmark that drops its hard cases
    produces a confident number that is wrong in the flattering direction, and a
    reader that refuses the pages it would have done badly on is exactly the
    failure mode that would exploit it. Every character of the page is charged.
    """
    want = textguard.normalise(page.body).strip()
    return PageCard(
        id=page.id,
        matched=cer.Score(0, 0),
        content=cer.Score(len(want), len(want)),
        prose=cer.Score(len(want), len(want)),
        order=orderlib.Order(tau=0.0, inversions=0, matched=0, read=0, expected=0),
        oov=0,
        homoglyphs=0,
        failure="no reading in the directory",
    )


@dataclass
class Card:
    """What one reader did on one Kvant set."""

    set_name: str
    model: str
    pages: list[PageCard] = field(default_factory=list)

    @property
    def failures(self) -> list[PageCard]:
        return [p for p in self.pages if p.failed]

    def rate(self, which: str) -> float:
        """A character rate, micro averaged over the set.

        Micro and not the mean of the per page rates, so a two line page of
        contents cannot weigh the same as a dense page of problems.
        """
        edits = sum(getattr(p, which).edits for p in self.pages)
        length = sum(getattr(p, which).length for p in self.pages)
        return edits / length if length else 0.0

    def tau(self) -> float:
        read = [p for p in self.pages if not p.failed]
        return sum(p.order.tau for p in read) / len(read) if read else 0.0

    def inversions(self) -> int:
        """The worst page's inverted pair count, not the mean.

        §06 of the order module is the reason. A perfectly interleaved two column
        page still scores a tau above +0.5, so the mean tau of a run where a
        tenth of the pages are woven looks fine. The count of inverted pairs is
        what separates a woven page from a clean one, and the worst page is what
        says whether the reader can do columns at all.
        """
        return max((p.order.inversions for p in self.pages), default=0)

    def coverage(self) -> float:
        """Share of the reference's characters that landed in a passage.

        Characters and not blocks. Blocks was the first version and it reads far
        worse than the truth on this reference, because the publisher's text
        layer shatters a letter spaced heading into one block a letter and a
        page can carry twenty of those against ten paragraphs. On the dev set
        592 reference blocks went unmatched and they are 1.1 % of the reference
        by character, so the block number said 74 % of a page was found where
        the character number says 98.9 %. The character number is the one that
        answers what a person means by how much of the page did it get.
        """
        found = sum(p.matched.length for p in self.pages)
        want = sum(p.content.length for p in self.pages)
        return found / want if want else 0.0

    def flagged(self, which: str) -> float:
        """Share of pages carrying at least one flag of this kind."""
        if not self.pages:
            return 0.0
        return sum(1 for p in self.pages if getattr(p, which)) / len(self.pages)

    def worst(self, n: int = 10) -> list[PageCard]:
        return sorted(self.pages, key=lambda p: -p.content.rate)[:n]

    def to_dict(self) -> dict[str, object]:
        return {
            "set": self.set_name,
            "model": self.model,
            "pages": len(self.pages),
            "failed": len(self.failures),
            "content_cer": round(self.rate("content"), 6),
            "matched_cer": round(self.rate("matched"), 6),
            "prose_cer": round(self.rate("prose"), 6),
            "content_coverage": round(self.coverage(), 6),
            "order_tau": round(self.tau(), 6),
            "worst_page_inversions": self.inversions(),
            "oov_page_rate": round(self.flagged("oov"), 6),
            "homoglyph_page_rate": round(self.flagged("homoglyphs"), 6),
            "worst": [
                {
                    "page": p.id,
                    "content_cer": round(p.content.rate, 6),
                    "matched_cer": round(p.matched.rate, 6),
                    "tau": round(p.order.tau, 6),
                    "inversions": p.order.inversions,
                }
                for p in self.worst()
            ],
            "failures": [{"page": p.id, "why": p.failure} for p in self.failures],
        }

    def to_markdown(self) -> str:
        read = len(self.pages) - len(self.failures)
        out = [
            f"# {self.model} on {self.set_name}",
            "",
            f"{read} of {len(self.pages)} pages read, {len(self.failures)} with no reading at "
            "all. A page with no reading is charged in full and stays in every denominator "
            "below.",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Content CER | {self.rate('content'):.2%} |",
            f"| Matched block CER | {self.rate('matched'):.2%} |",
            f"| Prose CER | {self.rate('prose'):.2%} |",
            f"| Content coverage | {self.coverage():.1%} |",
            f"| Reading order tau | {self.tau():+.3f} |",
            f"| Worst page inversions | {self.inversions()} |",
            f"| Pages with a repeated unknown word | {self.flagged('oov'):.1%} |",
            f"| Pages with a homoglyph in mathematics | {self.flagged('homoglyphs'):.1%} |",
            "",
            "Content CER charges every character of every reference block the reading did not "
            "produce. Matched block CER is the rate inside the blocks that were paired, so the "
            "gap between the two is what the reader dropped rather than what it misread.",
            "",
            "The reference is the publisher's text layer and its column order is wrong, which "
            "is why order is a separate number here and not folded into the character rate. A "
            "tau below +1 is not by itself a mark against the reader.",
            "",
            "## Worst pages by content CER",
            "",
        ]
        for p in self.worst():
            out.append(
                f"- {p.id}: content {p.content.rate:.2%}, matched {p.matched.rate:.2%}, "
                f"tau {p.order.tau:+.3f}, {p.order.inversions} inversions"
            )
        if self.failures:
            out.append("")
            out.append("## Pages with no reading")
            out.append("")
            for p in self.failures:
                out.append(f"- {p.id}: {p.failure}")
        return "\n".join(out) + "\n"


def run(
    pages: list[kvant.Page], readings: Path, words: set[str], *, model: str, set_name: str
) -> Card:
    card = Card(set_name=set_name, model=model)
    for page in pages:
        path = find_reading(readings, page.id)
        if path is None:
            card.pages.append(missing(page))
            continue
        card.pages.append(judge(page, path.read_text(encoding="utf-8"), words))
    return card


def table(cards: list[Card]) -> str:
    """The bake off itself: several readers, one table, ranked by content CER.

    Ranked and not merely listed. The point of M8 item 5 is to end with a
    default chosen by a number rather than inherited from the Bourbaki bake off,
    and a table that leaves the ranking to the reader invites the default to
    stay where it is because moving it would need an argument.
    """
    ordered = sorted(cards, key=lambda c: c.rate("content"))
    out = [
        "| Reader | Content CER | Matched CER | Prose CER | Coverage | Tau | Worst inversions "
        "| OOV pages | Homoglyph pages | No reading |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for card in ordered:
        out.append(
            f"| {card.model} | {card.rate('content'):.2%} | {card.rate('matched'):.2%} | "
            f"{card.rate('prose'):.2%} | {card.coverage():.1%} | {card.tau():+.3f} | "
            f"{card.inversions()} | {card.flagged('oov'):.1%} | "
            f"{card.flagged('homoglyphs'):.1%} | {len(card.failures)} |"
        )
    return "\n".join(out) + "\n"


def write(card: Card, *, json_path: Path | None, markdown_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(card.to_dict(), indent=2, ensure_ascii=False) + "\n")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(card.to_markdown(), encoding="utf-8")
