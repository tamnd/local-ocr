"""Formula comparison by what it prints, not by how it is spelled.

The metric is CDM, from CVPR 2025, and the reason it is CDM and not BLEU or
exact match is one sentence long: `\\frac` and `\\dfrac` print the same thing and
`\\to` and `\\rightarrow` print the same thing, so a metric that reads the source
punishes a model for a synonym while passing it for a lost subscript. The
published spread is stark, pix2tex 0.636 against UniMERNet 0.968, and it is a
spread that source comparison does not produce.

## What this is, and what it is not

The published CDM renders both formulas to a raster, runs a character detector
over each, and matches the detections. This does the same comparison one step
earlier: it asks the mathtext layout engine where every glyph landed and matches
those. Same idea, no detector, and therefore no detector error.

That is a difference worth naming rather than glossing, so every report this
writes records `backend: mathtext`, and a number from here should not be put in
a table next to a published CDM number as though the two were interchangeable.
What it is good for is comparing two readers of the same page, which is the only
thing this repository ever asks of it.

## What it cannot render

mathtext is a subset of LaTeX and Bourbaki uses things outside it, `pmatrix`
above all. A span neither side can render is recorded as unrenderable and left
out of the mean, and the count is reported, because a metric that silently drops
its hard cases produces a confident number that is wrong in the flattering
direction. A span one side renders and the other does not is recorded as a third
thing again: it is evidence, but it is not a score of zero, because mathtext
lacking a macro is not the same as the model inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from local_ocr.rules.mathtex import Span, split

# How far apart two glyphs of the same character may sit, as a fraction of the
# larger rendering's diagonal, and still be the same glyph. Loose enough that a
# formula which is a hair wider does not lose every character after the first
# difference, tight enough that a subscript is not matched to a superscript.
TOLERANCE = 0.06


class Unrenderable(Exception):
    """mathtext could not lay this formula out."""


@dataclass(frozen=True)
class Glyph:
    char: str
    size: float
    x: float
    y: float


@dataclass(frozen=True)
class Layout:
    glyphs: tuple[Glyph, ...]
    width: float
    height: float

    @property
    def diagonal(self) -> float:
        return max(1.0, (self.width**2 + self.height**2) ** 0.5)


# mathtext has no \displaystyle, so the one style difference it does expose is
# used instead: in a display, \frac is set the way \dfrac is set everywhere. A
# reading that writes \dfrac inside $$...$$ has therefore written the same thing
# as one that writes \frac, and inside $...$ it has not, which is the truth
# about how the two print and the reason the flag has to be carried this far.
_DISPLAY_STYLE = ((r"\frac", r"\dfrac"),)


def _in_display(body: str) -> str:
    out = body
    for text_style, display_style in _DISPLAY_STYLE:
        out = out.replace(display_style, text_style).replace(text_style, display_style)
    return out


@lru_cache(maxsize=4096)
def layout(formula: str, display: bool = False) -> Layout:
    """Ask mathtext where every glyph of a formula lands.

    Cached, because a page is compared against two or three other readings of
    itself and the same span is laid out every time.
    """
    try:
        from matplotlib.font_manager import FontProperties
        from matplotlib.mathtext import MathTextParser
    except ImportError as err:  # pragma: no cover - a missing extra, not a branch
        raise Unrenderable("formula comparison needs the eval extra: uv sync --extra eval") from err

    body = formula.strip()
    if not body:
        raise Unrenderable("the span is empty")
    if display:
        body = _in_display(body)
    try:
        parsed = MathTextParser("path").parse(f"${body}$", dpi=100, prop=FontProperties(size=12))
    except Exception as err:
        raise Unrenderable(str(err)) from err
    width, height, _depth, glyphs, _rects = parsed
    return Layout(
        tuple(Glyph(chr(g[2]), float(g[1]), float(g[4]), float(g[5])) for g in glyphs),
        float(width),
        float(height),
    )


def compare(left: str, right: str, display: bool = False) -> float:
    """The CDM score of two formulas, between 0 and 1.

    Both are laid out, glyphs of the same character are matched nearest pair
    first, and how near counts as near is a fraction of the larger of the two
    renderings, so a formula that is a hair wider does not lose every character
    after the first difference. The score is twice the matches over the total, so
    a formula that drops half its characters and one that invents as many are
    punished the same way, which they should be: both changed what the page says.
    """
    a, b = layout(left, display), layout(right, display)
    if not a.glyphs and not b.glyphs:
        return 1.0
    if not a.glyphs or not b.glyphs:
        return 0.0
    scale = max(a.diagonal, b.diagonal)
    matched = _match(a, b, TOLERANCE * scale)
    return 2 * matched / (len(a.glyphs) + len(b.glyphs))


def _match(a: Layout, b: Layout, tolerance: float) -> int:
    """Pair glyphs of the same character, closest pair first.

    Greedy and not a full assignment. The formulas being compared are two
    readings of the same printed span, so the pairs that matter are nearly
    coincident and a greedy pass finds them; the cases where greedy and optimal
    differ are formulas that already disagree badly enough that the score is low
    either way.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, one in enumerate(a.glyphs):
        for j, other in enumerate(b.glyphs):
            if one.char != other.char:
                continue
            gap = ((one.x - other.x) ** 2 + (one.y - other.y) ** 2) ** 0.5
            if gap <= tolerance:
                # A subscript and a superscript are the same character set at
                # the same size, so size alone does not separate them and
                # position does. Size is still in the key, to break ties towards
                # the pair that is the same size as well as in the same place.
                pairs.append((gap + abs(one.size - other.size), i, j))
    pairs.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = 0
    for _, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched += 1
    return matched


@dataclass
class SpanReport:
    """One pairing of a reference span against a read one."""

    index: int
    reference: str
    read: str
    score: float | None = None
    """None when the pair was not scored, and then `note` says why."""
    note: str = ""

    @property
    def scored(self) -> bool:
        return self.score is not None


@dataclass
class PageReport:
    """Every span of one page, and what became of it."""

    spans: list[SpanReport] = field(default_factory=list)

    @property
    def scored(self) -> list[SpanReport]:
        return [s for s in self.spans if s.scored]

    @property
    def mean(self) -> float | None:
        """The mean over the spans that were scored, and None if none were."""
        got = self.scored
        if not got:
            return None
        return sum(s.score or 0.0 for s in got) / len(got)

    @property
    def below_threshold(self) -> int:
        """Spans under 0.99, which is the number §05 asks for and the useful one.

        A mean hides a page that scored 1.0 on nineteen spans and 0.2 on the
        twentieth, and the twentieth is a theorem that now says something else.
        """
        return sum(1 for s in self.scored if (s.score or 0.0) < 0.99)

    @property
    def unrenderable(self) -> int:
        return sum(1 for s in self.spans if not s.scored and s.note.startswith("neither"))

    @property
    def one_sided(self) -> int:
        return sum(1 for s in self.spans if not s.scored and s.note.startswith("one side"))

    @property
    def unpaired(self) -> int:
        """Spans one page has and the other does not.

        Counted and not scored, because a lost display is not a formula that
        scored badly, it is a formula that is not there, and rolling the two
        together into one mean tells whoever reads the report neither fact.
        """
        return sum(1 for s in self.spans if not s.scored and s.note.startswith("unpaired"))


def compare_pages(reference: str, read: str) -> PageReport:
    """Pair the spans of two readings of a page by position, and score each.

    Pairing by position and not by content, because content is what is being
    measured. A page whose spans are paired by similarity would score well by
    construction: it would match every formula to whichever formula it most
    resembles and never notice that the third display went missing.
    """
    left, _ = split(reference)
    right, _ = split(read)
    report = PageReport()
    for i in range(max(len(left), len(right))):
        one: Span | None = left[i] if i < len(left) else None
        other: Span | None = right[i] if i < len(right) else None
        if one is None or other is None:
            report.spans.append(
                SpanReport(
                    index=i,
                    reference=one.text if one else "",
                    read=other.text if other else "",
                    note="unpaired: the two readings do not have the same number of spans",
                )
            )
            continue
        try:
            left_layout = layout(one.text, one.display)
        except Unrenderable as err:
            left_layout, left_why = None, str(err)
        else:
            left_why = ""
        try:
            right_layout = layout(other.text, other.display)
        except Unrenderable as err:
            right_layout, right_why = None, str(err)
        else:
            right_why = ""

        if left_layout is None and right_layout is None:
            report.spans.append(
                SpanReport(i, one.text, other.text, None, f"neither side renders: {left_why}")
            )
            continue
        if left_layout is None or right_layout is None:
            which = "reference" if left_layout is None else "reading"
            why = left_why or right_why
            report.spans.append(
                SpanReport(
                    i, one.text, other.text, None, f"one side does not render, the {which}: {why}"
                )
            )
            continue
        # The reference decides the style. A reading that turned a display into
        # an inline span has made a mistake the conformance rules catch, and it
        # should not also change what the formula is compared as.
        report.spans.append(
            SpanReport(i, one.text, other.text, compare(one.text, other.text, one.display))
        )
    return report
