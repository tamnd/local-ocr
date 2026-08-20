"""Go's `%q`, because the details of a rejection are compared against it.

A rejected page carries a reason into the queue history and into the failures
report, and two of the eight rules put a fragment of the page in that reason
with `%q` around it. Python's `repr` is not that: it prefers single quotes, it
escapes different things, and it spells a code point `\\xe2` where Go spells it
`\\u00e2`. A reimplementation whose reasons differ from the original's is a
reimplementation whose reports cannot be diffed against the original's, which is
most of what a report is for.

This is `strconv.Quote` and nothing else. It is small because the alternative is
to loosen every parity test that compares a detail string, and a test that only
checks the rule name would pass a rule that names the wrong line.
"""

from __future__ import annotations

import unicodedata

_SHORT = {
    "\a": "\\a",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\v": "\\v",
    "\\": "\\\\",
    '"': '\\"',
}

# strconv.IsPrint is the Unicode letters, marks, numbers, punctuation and
# symbols, plus the ASCII space and nothing else that is a space. A private use
# character is none of those, which is what makes the provider marker that
# reached the corpus print as an escape rather than as nothing at all.
_PRINTABLE_CATEGORIES = ("L", "M", "N", "P", "S")


def quote(text: str) -> str:
    """Render text the way Go's `%q` and `strconv.Quote` do."""
    out = ['"']
    for ch in text:
        short = _SHORT.get(ch)
        if short is not None:
            out.append(short)
            continue
        code = ord(ch)
        if 0x20 <= code < 0x7F:
            out.append(ch)
            continue
        if code < 0x20 or code == 0x7F:
            out.append(f"\\x{code:02x}")
            continue
        if unicodedata.category(ch)[0] in _PRINTABLE_CATEGORIES:
            out.append(ch)
            continue
        if code < 0x10000:
            out.append(f"\\u{code:04x}")
        else:
            out.append(f"\\U{code:08x}")
    out.append('"')
    return "".join(out)
