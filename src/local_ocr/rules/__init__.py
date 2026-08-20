r"""The Go rules, again, in Python.

Every module in here is a transcription of a package in `tamnd/bourbaki-solver`
and not a design of its own. The reason they exist at all is that the machine
reading the pages has to decide whether an answer is a transcription before it
spends another fifteen minutes on the next one, and it cannot call into Go to
ask.

A second implementation of a rule is a second opinion about what the rule is,
and two opinions that drift produce a corpus half of which was accepted under
one set of rules. So none of this is trusted on its own: `cmd/ocrfixtures` on
the Go side records what the originals return, `tests/fixtures/` carries the
answers, and `tests/test_go_parity.py` asserts this code agrees on every case.
When a rule changes over there, these tests fail over here. That is the point.

Where a Go regexp is mirrored the pattern carries `re.ASCII`, because Go's
`\b`, `\s`, `\d` and `\w` are ASCII and Python's are not, and the difference is
invisible until a page of French accents lands on it.
"""

from local_ocr.rules.mathtex import Span, split
from local_ocr.rules.textguard import (
    Leak,
    check,
    clean,
    dollars,
    normalise,
    normalise_prose,
    stars,
    strip,
)
from local_ocr.rules.validate import (
    Expect,
    Problem,
    Rule,
    ok,
    reasons,
    rules_of,
    validate,
)

__all__ = [
    "Expect",
    "Leak",
    "Problem",
    "Rule",
    "Span",
    "check",
    "clean",
    "dollars",
    "normalise",
    "normalise_prose",
    "ok",
    "reasons",
    "rules_of",
    "split",
    "stars",
    "strip",
    "validate",
]
