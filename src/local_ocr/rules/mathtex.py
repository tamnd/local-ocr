"""Where the mathematics of a body starts and stops.

A transcription of `mathtex.Split` in `tamnd/bourbaki-solver`. The rules are
LaTeX's and not Markdown's: a backslash escapes the character after it, so `\\$`
is a dollar sign and not a delimiter, two dollars open a display and one opens
an inline span, and a display closes only on two.

It matters here for one reason beyond tidiness. `stars` rewrites an asterisk
outside the mathematics and leaves one inside it alone, because the same glyph
is a binary law on one side of a dollar and Bourbaki's forward reference mark on
the other. A splitter that disagrees with the Go one by a single rune rewrites
the wrong glyph, and `K^*` runs through the volumes in their thousands.
"""

from __future__ import annotations

from dataclasses import dataclass

_NONE, _INLINE, _DISPLAY = 0, 1, 2


@dataclass(frozen=True)
class Span:
    """One stretch of mathematics in a body, without its delimiters."""

    text: str
    display: bool
    line: int
    """The body line the opening delimiter sits on, counting from one."""
    start: int
    end: int
    """Where the text sits in the body, counted in runes and not bytes.

    A rule needs only the text, but anything that repairs the mathematics has to
    put it back where it came from, and searching for the text again finds the
    wrong copy of a span that is written twice on a line.
    """


def split(body: str) -> tuple[list[Span], Span | None]:
    """Cut a normalised body into its math spans.

    The second return is the span left open at the end of the body, and None
    when there is none. It carries the line its opening delimiter sits on, which
    is the line somebody has to go and look at. The end of the file is where the
    problem shows up and never where it is.
    """
    spans: list[Span] = []
    state = _NONE
    line = 1
    display = False
    open_line = 1
    start = 0
    rs = list(body)
    n = len(rs)
    i = 0
    while i < n:
        ch = rs[i]
        if ch == "\n":
            line += 1
            i += 1
            continue
        if ch == "\\":
            # The escape takes the next character with it, whatever it is, and a
            # newline still has to be counted.
            if i + 1 < n:
                if rs[i + 1] == "\n":
                    line += 1
                i += 1
            i += 1
            continue
        if ch != "$":
            i += 1
            continue
        double = i + 1 < n and rs[i + 1] == "$"
        if state == _NONE:
            display, open_line = double, line
            start = i + 1
            if double:
                state = _DISPLAY
                start = i + 2
                i += 1
            else:
                state = _INLINE
        elif state == _INLINE:
            spans.append(Span("".join(rs[start:i]), display, open_line, start, i))
            state = _NONE
        else:
            if not double:
                # A lone dollar inside a display is text.
                i += 1
                continue
            spans.append(Span("".join(rs[start:i]), display, open_line, start, i))
            state = _NONE
            i += 1
        i += 1
    if state != _NONE:
        return spans, Span("".join(rs[start:]), display, open_line, start, n)
    return spans, None


def in_math(text: str) -> list[bool]:
    """Mark every rune of a body that sits inside a math span.

    A span that never closes takes the rest of the body with it. That is not
    what the page means, it is a file the audit is already reporting, and until
    somebody reads the printed page there is no telling where the mathematics
    was supposed to stop. Marking the tail leaves it alone, which is the right
    thing to do with text whose reading is not known.
    """
    spans, unclosed = split(text)
    mark = [False] * len(text)

    def cover(start: int, end: int) -> None:
        for i in range(max(0, start), min(end, len(mark))):
            mark[i] = True

    for span in spans:
        cover(span.start, span.end)
    if unclosed is not None:
        cover(unclosed.start, len(mark))
    return mark
