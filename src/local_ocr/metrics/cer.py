"""Character error rate, over the prose and over the whole page.

Reported separately for the two, because a page that is half display
mathematics has a whole page CER dominated by LaTeX spelling, and LaTeX spelling
is what CDM is for. A model that writes `\\dfrac` where the reference writes
`\\frac` has changed nothing about what the page says and would carry a CER
penalty on every character of the difference.

Both readings are normalised first, with the same `textguard.normalise` the
corpus is written through. Otherwise the metric measures the typography the
corpus already fixes: a model that writes `\\mathbb{Z}` scores worse than one
that writes `\\mathbf{Z}`, and by the time either reaches a page file they are
the same text.
"""

from __future__ import annotations

from dataclasses import dataclass

from local_ocr.rules import textguard
from local_ocr.rules.mathtex import split


def _levenshtein_slow(a: str, b: str) -> int:
    """Two rows of the usual table.

    Kept even though it is not what runs, because it is short enough to check by
    eye and `rapidfuzz` is not. `tests/test_metrics.py` asserts the two agree,
    which is what makes the fast one trustworthy at 3000 characters a page.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def distance(a: str, b: str) -> int:
    """Levenshtein distance, fast where it can be.

    A page is about three thousand characters, so the table is nine million
    cells and the pure Python version takes seconds per page. `rapidfuzz` is in
    the eval extra and does it in microseconds. When it is absent the slow one
    runs, so a machine without the extra still produces the same number and
    takes longer over it.
    """
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError:
        return _levenshtein_slow(a, b)
    return int(Levenshtein.distance(a, b))


def prose_of(text: str) -> str:
    """The page with its mathematics taken out, delimiters and all.

    Each span is replaced by a single space rather than deleted, so that two
    words either side of a removed formula do not become one word and produce a
    difference that is an artefact of the measuring.

    The dollars go with the span. `mathtex.Span` records where the mathematics
    starts, which is inside the delimiters, so a span whose delimiters were left
    behind would put `$ $` into the prose and let a model that lost a formula
    entirely keep the two characters that say there was one.
    """
    spans, unclosed = split(text)
    cuts: list[tuple[int, int]] = []
    for span in spans:
        width = 2 if span.display else 1
        cuts.append((span.start - width, span.end + width))
    if unclosed is not None:
        width = 2 if unclosed.display else 1
        cuts.append((unclosed.start - width, len(text)))

    out: list[str] = []
    at = 0
    for start, end in cuts:
        out.append(text[at:start])
        out.append(" ")
        at = end
    out.append(text[at:])
    return "".join(out)


@dataclass(frozen=True)
class Score:
    edits: int
    length: int
    """Characters in the reference, which is the denominator."""

    @property
    def rate(self) -> float:
        """Edits per reference character. Above 1.0 is possible and is a signal."""
        if self.length == 0:
            return 0.0 if self.edits == 0 else 1.0
        return self.edits / self.length


def page(reference: str, read: str) -> tuple[Score, Score]:
    """The whole page rate and the prose rate, in that order."""
    left = textguard.normalise(reference).strip()
    right = textguard.normalise(read).strip()
    whole = Score(distance(left, right), len(left))
    left_prose = _collapse(prose_of(left))
    right_prose = _collapse(prose_of(right))
    return whole, Score(distance(left_prose, right_prose), len(left_prose))


def _collapse(text: str) -> str:
    """Runs of whitespace to one space.

    A printed paragraph is one line in this corpus and a model that wraps it at
    eighty columns has made a formatting mistake, not a reading mistake. The
    formatting is a conformance rule with its own rate; letting it also inflate
    the CER would count it twice and drown the reading errors it sits on top of.
    """
    return " ".join(text.split())
