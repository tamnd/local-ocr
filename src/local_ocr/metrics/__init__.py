"""What a reading is measured by.

Five things, and they answer different questions, which is why none of them is
folded into the others.

    cer           did it read the words
    cdm           did it read the formulas
    order         did it put them on the page in the order a person reads
    conformance   did it follow the eight house rules of these volumes
    acceptance    would the eight rules in ocr/validate.go have taken it

Order is the one that is not about a gold standard. On a two column page the
publisher's own text layer is the thing that has the order wrong and the vision
reading is the thing that has it right, so the metric says how two readings
differ and leaves which of them is correct to a person who looks at one page.

The last one is one line and lives here rather than in a module of its own,
because it is not a measurement of the text. It is the corpus's own gate,
asked in advance, and the number it produces predicts throughput: a rejected
page is a page read twice.
"""

from __future__ import annotations

from local_ocr.metrics import cdm, cer, conformance, order
from local_ocr.rules.validate import Expect, ok

__all__ = ["Expect", "accepted", "cdm", "cer", "conformance", "order"]


def accepted(text: str, expect: Expect) -> bool:
    """Whether the eight acceptance rules would take this reading as it stands.

    First read acceptance is this over a set of pages, and the reason to report
    it is throughput rather than quality. It says nothing about whether the page
    is right; a confident wrong subscript passes every one of the eight.
    """
    return ok(text, expect)
