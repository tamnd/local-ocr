"""Reading order, measured on its own and not folded into CER.

CER cannot see this. Two readings of a two column page that carry exactly the
same characters in two different orders have the same word error against each
other only if the alignment happens to find the swap, and against a reference
they can differ by the whole length of a column while every word on the page is
correct. So a page can be read perfectly and assembled wrongly, and the number
that is supposed to say how well it was read will say it was read badly for a
reason that has nothing to do with reading.

Kvant is where this matters. The born digital path takes the text layer out of
the publisher's PDF, and on a two column page that layer runs down the physical
order of the drawing operations rather than the order a person reads in, so it
interleaves the columns. The vision path looks at the picture and gets the order
right. Which means the reference text is the one that is wrong, CER against it
would punish the reading that is correct, and the only way to see any of that is
to measure order as its own thing and report it next to CER rather than inside
it.

The unit is the paragraph, because that is the granularity at which order
actually goes wrong. A column interleave moves whole blocks; it does not move
words within a sentence.

One number here is worth knowing before anybody sets a threshold on it. A page
whose two columns are perfectly interleaved does not score near zero. Weaving
the halves of 386 real Kvant pages together gives a tau between +0.502 and
+0.667, median +0.538, and not one of them falls below +0.5. Alternating blocks
leave most pairs in the right relative order and destroy only the local order;
reading the page backwards is what gives -1. So a threshold set on tau by
intuition, anything at half or below, would let the exact failure this module
was written for go straight past, and that is why the raw count of inverted
pairs is reported beside tau and not instead of it. On the same 386 pages the
median interleave is 21 inverted pairs and the worst is 5777, which is the
number that actually separates a woven page from a clean one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_ocr.rules import textguard

BLOCK = re.compile(r"\n\s*\n")
"""What separates one block from the next.

A blank line, which is how the corpus writes a paragraph break and how it writes
the break around a display formula. So a display equation is a block and its
position is measured like any other, which is right: an equation that ends up in
the wrong column is exactly the failure being looked for.
"""

WORD = re.compile(r"\w+", re.UNICODE)

FRONT = re.compile(r"\A---\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
"""The YAML header the corpus writes above every page.

Cut before anything else. It is metadata rather than page content, it is written
by the tool and not read by the model, and it is one block that always matches
its counterpart, so leaving it in adds a free block to both counts and makes
coverage and tau flatter than the page deserves. Found by running the metric
over 400 real Kvant pages: every one of them had it.
"""

MATCH = 0.5
"""How much of a block's vocabulary has to be shared to call it the same block.

Half. Two readings of the same paragraph differ in the mathematics markup, in
hyphenation at a line break and in the odd misread word, and they still share
most of their words. Two different paragraphs of the same article share the
function words and little else, and the function words are a small part of a set
because a set counts each of them once.
"""


@dataclass(frozen=True)
class Order:
    """How much of the reading is in the order the reference has it in."""

    tau: float
    """Kendall tau over the blocks that matched, from -1 to 1.

    One is the same order. Zero is no relation. Minus one is exactly reversed,
    which is what a page read bottom to top would give.
    """

    inversions: int
    """Pairs of blocks whose order disagrees.

    Reported beside tau because tau is a ratio and hides how big the page was. A
    tau of 0.9 on a page of forty blocks is seventy eight pairs out of order and
    is a real problem; the same tau on a page of five blocks is one swap.
    """

    matched: int
    """Blocks matched to a block of the reference."""

    read: int
    """Blocks in the reading."""

    expected: int
    """Blocks in the reference."""

    @property
    def coverage(self) -> float:
        """Share of the reference's blocks that were found at all.

        Kept apart from tau on purpose. A reading that dropped half the page and
        put the rest in perfect order has a tau of 1, and reporting that number
        on its own would say the page was read perfectly. Order is only a
        sentence about the blocks that are there.
        """
        return self.matched / self.expected if self.expected else 0.0

    def __str__(self) -> str:
        return (
            f"tau {self.tau:+.3f}, {self.inversions} inversions, "
            f"{self.matched} of {self.expected} blocks matched"
        )


def blocks(text: str) -> list[str]:
    """The text cut into blocks, normalised, with the header and empties gone."""
    body = textguard.normalise(textguard.strip(text))
    body = FRONT.sub("", body)
    return [b.strip() for b in BLOCK.split(body) if b.strip()]


def _keys(text: str) -> list[tuple[frozenset[str], str]]:
    return [(frozenset(w.lower() for w in WORD.findall(b)), b) for b in blocks(text)]


def _overlap(a: tuple[frozenset[str], str], b: tuple[frozenset[str], str]) -> float:
    """How much two blocks have in common, from 0 to 1.

    Words when there are words, and the text itself when there are none. A block
    can be all punctuation or all markup, `⟦folio 45⟧` and a bare rule being the
    two that turn up, and a word set cannot say anything about those. Before the
    fallback, 13 of 400 real Kvant pages could not match themselves: the page
    plainly contained the block and the metric reported it dropped.
    """
    if not a[0] or not b[0]:
        return 1.0 if a[1] == b[1] else 0.0
    return len(a[0] & b[0]) / len(a[0] | b[0])


def _match(
    read: list[tuple[frozenset[str], str]], want: list[tuple[frozenset[str], str]]
) -> dict[int, int]:
    """Which reference block each of the reading's blocks is, by index.

    Greedy and best first rather than left to right. Left to right would let an
    early block take a partial match that a later block matches better, and the
    later block would then match nothing and be counted as dropped, which turns
    one bad guess into two wrong numbers.

    One function, used by both the tau and the character rate, so that the two
    numbers cannot disagree about which block is which.
    """
    scored = [(_overlap(a, b), i, j) for i, a in enumerate(read) for j, b in enumerate(want)]
    scored.sort(key=lambda s: (-s[0], s[1], s[2]))
    taken_read: set[int] = set()
    taken_want: set[int] = set()
    found: dict[int, int] = {}
    for score, i, j in scored:
        if score < MATCH:
            break
        if i in taken_read or j in taken_want:
            continue
        taken_read.add(i)
        taken_want.add(j)
        found[i] = j
    return found


def _pairs(
    read: list[tuple[frozenset[str], str]], want: list[tuple[frozenset[str], str]]
) -> list[int]:
    """For the reading's blocks in order, which reference block each one is."""
    found = _match(read, want)
    return [found[i] for i in sorted(found)]


def _tau(sequence: list[int]) -> tuple[float, int]:
    """Kendall tau and the raw count of pairs out of order.

    Counted directly rather than through a library. There are at most a few tens
    of blocks on a page, the quadratic loop is nothing, and the definition being
    visible is worth more here than the speed: tau has several variants that
    differ on ties, and this sequence has no ties because every block matches at
    most one reference block.
    """
    n = len(sequence)
    if n < 2:
        return 1.0, 0
    inversions = sum(1 for i in range(n) for j in range(i + 1, n) if sequence[i] > sequence[j])
    total = n * (n - 1) // 2
    return (total - 2 * inversions) / total, inversions


def paired(read: str, want: str) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """The blocks of two readings matched up, and what each side had left over.

    Three things back: the matched pairs in the reference's order, the reference
    blocks nothing matched, and the reading blocks nothing matched.

    This is the part underneath `order`. `order` throws the text away and keeps
    the positions, which is all a tau needs; a character error rate needs the
    text, and it needs it matched the same way, because matching it a second
    time by a different rule would let the two numbers disagree about which
    block is which. Splitting the matching out is what lets a reader be scored
    on what it read without the reference's own bad column order being charged
    to it as character errors.
    """
    a, b = _keys(read), _keys(want)
    found = _match(a, b)
    back = {j: i for i, j in found.items()}
    pairs = [(a[back[j]][1], b[j][1]) for j in sorted(back)]
    return (
        pairs,
        [b[j][1] for j in range(len(b)) if j not in back],
        [a[i][1] for i in range(len(a)) if i not in found],
    )


def order(read: str, want: str) -> Order:
    """Compare the order of one reading against another.

    Not against a gold standard, because on the pages this exists for there is
    no gold: the reference is the publisher's text layer and it is the thing
    that has the order wrong. So this says how two readings differ, and which of
    them is right is a question a person answers by looking at one page and then
    applies to the rest of the run.
    """
    a, b = _keys(read), _keys(want)
    matched = _pairs(a, b)
    tau, inversions = _tau(matched)
    return Order(
        tau=tau,
        inversions=inversions,
        matched=len(matched),
        read=len(a),
        expected=len(b),
    )
