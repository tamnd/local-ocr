"""The eval harness: a golden set in, a report out.

    local-ocr eval --set golden-dev --readings out/reader-a --model reader-a

The readings come from a directory rather than from a live model on purpose.
`ocr-batch` produces readings and this judges them, and keeping the two apart is
what lets the same harness judge the fleet's output, which no local model
produced and which is the whole of tier C. It also means an eval can be re-run
against a saved directory a year later without the model that wrote it still
existing.

A page that has no reading is not skipped. It is judged as the empty string,
which makes its character error rate 1.0, breaks every house rule that applied
to it, and fails acceptance, and it is named in the failures list as well. §05 is
blunt about why: a benchmark that silently drops its hard cases produces a
confident number that is wrong in the flattering direction.

The JSON carries no timestamp. It is meant to be diffed between runs, and a
timestamp would make every diff non empty and bury the one line that changed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from local_ocr import corpus as corpuslib
from local_ocr import golden
from local_ocr.metrics import cdm, cer, conformance
from local_ocr.rules import textguard
from local_ocr.rules.validate import (
    Confidence,
    Expect,
    Grammar,
    parse_page_label,
    reasons,
    rules_of,
    validate,
)


def expect_from(page: corpuslib.Page) -> Expect:
    """What the pipeline already knows about this page, out of its front matter.

    `extract` put the printed head there, so a golden set page carries its own
    answer to rules 4 and 6 and both can be run here. That matters: without it,
    acceptance measured in this harness would stand two of the eight rules down
    and report a number higher than the fleet would ever see.
    """
    label = parse_page_label(page.page_label)
    if label is None:
        return Expect(
            book=page.book,
            pdf_page=page.pdf_page,
            blank=page.method == "blank",
            # Not HEAD_LABEL, which is the default, because this volume prints
            # no page label and rule 4 would then demand one and reject every
            # correct reading of it. `lie-vii-ix` is 439 such pages.
            #
            # FOOT_NUMBER rather than HEAD_NUMBER only because validate asks one
            # question, whether the grammar is HEAD_LABEL, and the front matter
            # does not say which of the other two a volume uses. Naming the
            # wrong one of two things that behave identically is better than
            # claiming to know.
            grammar=Grammar.FOOT_NUMBER,
            has_head=page.has_head,
        )
    return Expect(
        book=page.book,
        pdf_page=page.pdf_page,
        blank=page.method == "blank",
        grammar=Grammar.HEAD_LABEL,
        chapter=label.chapter,
        page=label.page,
        # Printed, not interpolated: it was read off this page by `extract` and
        # is in the front matter of this page, so rule 6 is entitled to run.
        confidence=Confidence.FROM_HEAD,
        has_head=True,
    )


def printed_number(label: str) -> int | None:
    """The page number out of a page label, so `A VIII.144` gives 144."""
    digits = "".join(ch for ch in label.rpartition(".")[2] if ch.isdigit())
    return int(digits) if digits else None


def head_line(page: corpuslib.Page) -> str:
    """The running head of a page, put back together as one line.

    The front matter records the title and the page label separately and does
    not record which way round the page printed them, so the order has to come
    from somewhere. It comes from the page number, and the rule was measured
    rather than assumed. Over the 131 golden-dev pages that print both halves
    and whose reading carries both:

      even numbered page   66 pages, label first, `A VIII.4  MODULES ARTINIENS`
      odd numbered page    65 pages, title first, `STRUCTURE DES MODULES  A VIII.35`

    No exceptions either way. That is the verso and the recto. The label sits in
    the outer margin, which is the left hand side of a verso and the right hand
    side of a recto, and a bound volume puts the even numbers on the verso.

    This used to always write the title first, which was right on the odd pages
    and wrong on the even ones, and the docstring said so and called it good
    enough because the only caller parses a label out of the line. It stopped
    being good enough when §08's training pool started using the same line as
    the target a model is trained to produce.

    The folio is only used when there is nothing else. `running_head` is the
    whole printed line where a volume prints one, number and all: `lie-vii-ix`
    records "8 CARTAN SUBALGEBRAS AND REGULAR ELEMENTS Ch. VII" with the folio 8
    already in it, and appending the folio again would build a head no page has
    ever carried and then measure readings against it.
    """
    if page.running_head and page.page_label:
        number = printed_number(page.page_label)
        if number is not None and number % 2 == 0:
            return f"{page.page_label} {page.running_head}"
        return f"{page.running_head} {page.page_label}"
    parts = [page.running_head, page.page_label]
    line = " ".join(part for part in parts if part).strip()
    return line or page.folio


def conformance_reference(page: corpuslib.Page) -> str:
    """The reference as a faithful reading of the printed page would have it."""
    head = head_line(page)
    return f"{head}\n\n{page.body}" if head else page.body


def without_head(page: corpuslib.Page, reading: str) -> str:
    """The reading with its running head taken off, when it has one.

    The reference body has no head in it, so a reading that has one must have it
    removed before the two are compared or every page pays for obeying the
    prompt. Removed conservatively: only a short first line, only on a page that
    prints a head at all, and only when the line is recognisably that head.
    """
    if not page.has_head:
        return reading
    lines = reading.split("\n")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if _is_head(page, line.strip()):
            return "\n".join(lines[i + 1 :]).lstrip("\n")
        return reading
    return reading


def _is_head(page: corpuslib.Page, line: str) -> bool:
    if len(line) > 120 or line.startswith("#"):
        return False
    label = parse_page_label(page.page_label)
    if label is not None and parse_page_label(line) == label:
        return True
    if page.running_head and page.running_head.casefold() in line.casefold():
        return True
    return bool(page.folio) and line.strip() == page.folio


def find_reading(directory: Path, page_id: str) -> Path | None:
    """Where a reading of one page might be.

    Several shapes, because the readings come from several places: `ocr-batch`
    writes one file per image next to the tree it was given, the corpus writes
    `book/0042.md`, and a hand made comparison directory is usually flat.
    """
    book, _, number = page_id.partition("/")
    for candidate in (
        directory / book / f"{number}.md",
        directory / book / f"{number}.txt",
        directory / f"{book}-{number}.md",
        directory / f"{book}_{number}.md",
        directory / f"{number}.md",
    ):
        if candidate.is_file():
            return candidate
    return None


@dataclass
class PageResult:
    """One page judged, every metric on it."""

    id: str
    whole: cer.Score
    prose: cer.Score
    formulas: cdm.PageReport
    accepted: bool
    problems: str
    """The acceptance rules that rejected it, as one line, empty when it passed."""
    broke: list[str]
    """The house rules that applied to this page and were not obeyed."""
    rejected_by: list[str] = field(default_factory=list)
    """The acceptance rules that rejected it, by name, empty when it passed."""
    failure: str = ""
    """Why there was no reading at all, empty when there was one."""

    @property
    def failed(self) -> bool:
        return bool(self.failure)


@dataclass
class Report:
    set_name: str
    model: str
    results: list[PageResult] = field(default_factory=list)
    house: conformance.Conformance = field(default_factory=conformance.Conformance)
    unscored: list[str] = field(default_factory=list)
    """Pages of the set that were left out because their reference is no longer
    ground truth. Named individually and not just counted, because the remedy is
    that somebody reads those particular pages off the printed page."""

    @property
    def failures(self) -> list[PageResult]:
        return [r for r in self.results if r.failed]

    def cer_rates(self) -> tuple[float, float]:
        """Whole page and prose, micro averaged over every page in the set.

        Micro and not the mean of the per page rates, so that a short page whose
        reading went badly wrong does not weigh the same as a full one that went
        slightly wrong. Both numbers are in the JSON; this is the headline.
        """
        whole_edits = sum(r.whole.edits for r in self.results)
        whole_len = sum(r.whole.length for r in self.results)
        prose_edits = sum(r.prose.edits for r in self.results)
        prose_len = sum(r.prose.length for r in self.results)
        return (
            whole_edits / whole_len if whole_len else 0.0,
            prose_edits / prose_len if prose_len else 0.0,
        )

    def cdm_summary(self) -> dict[str, object]:
        scored = [s for r in self.results for s in r.formulas.scored]
        spans = [s for r in self.results for s in r.formulas.spans]
        mean = sum(s.score or 0.0 for s in scored) / len(scored) if scored else None
        below = sum(1 for s in scored if (s.score or 0.0) < 0.99)
        return {
            # Named in the report because it is not the published CDM. See the
            # module docstring in metrics/cdm.py.
            "backend": "mathtext",
            "spans": len(spans),
            "scored": len(scored),
            "mean": round(mean, 6) if mean is not None else None,
            "below_0_99": below,
            "below_0_99_rate": round(below / len(scored), 6) if scored else None,
            "unrenderable": sum(r.formulas.unrenderable for r in self.results),
            "one_sided": sum(r.formulas.one_sided for r in self.results),
            "unpaired": sum(r.formulas.unpaired for r in self.results),
        }

    def acceptance(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.accepted) / len(self.results)

    def rejections(self) -> list[tuple[str, int]]:
        """Which acceptance rules rejected pages, and how many each.

        A first read acceptance of 11.5 per cent says a reader is unusable and
        nothing about why. On the first real run of this harness, 177 of the 200
        pages were rejected by one rule, the running head, and everything else
        put together rejected six. That is a different piece of news from a
        reader that is wrong in nine ways, and the report has to be able to tell
        them apart. A page rejected by two rules is counted under both.
        """
        counts: dict[str, int] = {}
        for result in self.results:
            for rule in result.rejected_by:
                counts[rule] = counts.get(rule, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    def worst(self, n: int = 10) -> list[PageResult]:
        """The pages to look at first, which is the point of running this."""
        return sorted(self.results, key=lambda r: -r.prose.rate)[:n]

    def to_json(self) -> dict[str, object]:
        whole, prose = self.cer_rates()
        return {
            "set": self.set_name,
            "model": self.model,
            "pages": {
                # The set's own size, so that a reader of two reports can see
                # that one of them scored fewer pages than the set holds. With
                # `in_set` counting only what was scored, a set that quietly
                # lost four pages reports a full house.
                "in_set": len(self.results) + len(self.unscored),
                "scored": len(self.results),
                "read": len(self.results) - len(self.failures),
                "failed": len(self.failures),
                "unscored": self.unscored,
            },
            "cer": {
                "page": round(whole, 6),
                "prose": round(prose, 6),
                "page_mean": _mean(r.whole.rate for r in self.results),
                "prose_mean": _mean(r.prose.rate for r in self.results),
            },
            "cdm": self.cdm_summary(),
            "acceptance": {
                "rate": round(self.acceptance(), 6),
                "accepted": sum(1 for r in self.results if r.accepted),
                "of": len(self.results),
            },
            "rejections": [{"rule": rule, "pages": count} for rule, count in self.rejections()],
            "conformance": [
                {
                    "rule": count.name,
                    "applicable": count.applicable,
                    "obeyed": count.obeyed,
                    "rate": round(count.rate, 6) if count.rate is not None else None,
                }
                for count in self.house.rows()
            ],
            "failures": [{"page": r.id, "why": r.failure} for r in self.failures],
            "worst": [
                {
                    "page": r.id,
                    "prose_cer": round(r.prose.rate, 6),
                    "page_cer": round(r.whole.rate, 6),
                    "cdm_mean": (
                        round(r.formulas.mean, 6) if r.formulas.mean is not None else None
                    ),
                    "accepted": r.accepted,
                    "broke": r.broke,
                }
                for r in self.worst()
            ],
        }

    def to_markdown(self) -> str:
        whole, prose = self.cer_rates()
        cdm_bits = self.cdm_summary()
        read = len(self.results) - len(self.failures)
        out = [
            f"# {self.model} on {self.set_name}",
            "",
            f"{read} of {len(self.results)} pages read, {len(self.failures)} failed. "
            "A failed page is judged as an empty reading and stays in every denominator below.",
            "",
            *(
                [
                    f"{len(self.unscored)} of the set's "
                    f"{len(self.results) + len(self.unscored)} pages were left out, because "
                    "their reference is no longer an extraction and scoring against it would "
                    "be one reader marking another. They are the pages the extraction got "
                    "wrong, so they are harder than average and leaving them out moves every "
                    "number here in the flattering direction: " + ", ".join(self.unscored) + ".",
                    "",
                ]
                if self.unscored
                else []
            ),
            "| Metric | Value |",
            "| --- | --- |",
            f"| Prose CER | {prose:.2%} |",
            f"| Whole page CER | {whole:.2%} |",
            f"| Formula CDM mean (mathtext) | {_fmt(cdm_bits['mean'])} |",
            f"| Formula spans below 0.99 | {_fmt(cdm_bits['below_0_99_rate'], pct=True)} "
            f"of {cdm_bits['scored']} scored |",
            f"| First read acceptance | {self.acceptance():.1%} |",
            "",
            f"Spans not scored: {cdm_bits['unrenderable']} neither side renders, "
            f"{cdm_bits['one_sided']} one side does not, {cdm_bits['unpaired']} unpaired.",
            "",
            "## House rules",
            "",
            "| Rule | Rate | Pages it applied to |",
            "| --- | --- | --- |",
        ]
        for count in self.house.rows():
            rate = f"{count.rate:.1%}" if count.rate is not None else "did not apply"
            out.append(f"| {count.name} | {rate} | {count.applicable} |")
        out.append("")
        out.append("## Why pages were rejected")
        out.append("")
        rejections = self.rejections()
        if not rejections:
            out.append("Nothing was rejected.")
        else:
            out.append("| Acceptance rule | Pages it rejected |")
            out.append("| --- | --- |")
            for rule, count in rejections:
                out.append(f"| {rule} | {count} |")
        out.append("")
        out.append("## Worst pages by prose CER")
        out.append("")
        for r in self.worst():
            broke = ", ".join(r.broke) if r.broke else "no house rule broken"
            out.append(f"- {r.id}: prose {r.prose.rate:.2%}, {broke}")
        if self.failures:
            out.append("")
            out.append("## Pages with no reading")
            out.append("")
            for r in self.failures:
                out.append(f"- {r.id}: {r.failure}")
        return "\n".join(out) + "\n"


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 6) if items else 0.0


def _fmt(value: object, pct: bool = False) -> str:
    if value is None:
        return "no spans"
    return f"{float(value):.1%}" if pct else f"{float(value):.4f}"


def judge(page: corpuslib.Page, read: str, house: conformance.Conformance) -> PageResult:
    """One page against its reference, every metric, nothing swallowed.

    Three different texts, on purpose, because the three questions are not the
    same question.

    The character error rate and the formula comparison see the body without its
    running head on either side, because the reference has none and the reading
    should. The acceptance rules see the reading whole, head and all, because
    rule 4 is about the head being there. The house rules see the reading as it
    arrived, normalised on neither side, against a reference with its head put
    back.

    That last one is not a detail. `textguard.normalise` repairs two of the
    eight house rules on its way past: it rewrites `\\mathbb{Z}` to `\\mathbf{Z}`
    and it fences a bare asterisk. Judging conformance after it has run reports
    100 per cent on both rules for a model that gets both wrong on every page,
    which is exactly what the first run of this harness against a deliberately
    damaged set of readings did. The repair is real and the corpus benefits from
    it; the measurement has to happen before it.
    """
    reference = textguard.normalise(page.body)
    reading = textguard.normalise(textguard.strip(read))
    body = without_head(page, reading)
    as_written = textguard.strip(read)
    whole, prose = cer.page(reference, body)
    expect = expect_from(page)
    problems = validate(reading, expect)
    return PageResult(
        id=page.id,
        whole=whole,
        prose=prose,
        formulas=cdm.compare_pages(reference, body),
        accepted=not problems,
        problems=reasons(problems),
        rejected_by=[str(rule) for rule in rules_of(problems)],
        broke=house.observe(conformance_reference(page), as_written),
    )


def evaluate(
    set_name: str,
    readings: Path,
    *,
    purpose: golden.Purpose,
    model: str = "",
    corpus: Path | None = None,
) -> Report:
    """Run a golden set against a directory of readings."""
    pages = golden.load(set_name, purpose=purpose, corpus=corpus)
    # Pages whose reference has stopped being ground truth since the set was
    # drawn. Scoring them would compare a reader against another reader's
    # reading of the same page, and a model that agrees with its predecessor
    # would score well for it.
    #
    # Dropped and counted, not dropped quietly. These are pages the extraction
    # got wrong badly enough that somebody sent them to the fleet, so they are
    # harder than average and taking them out moves every number in the
    # flattering direction. The count is on the report for that reason and the
    # fix is a person reading them, not a smaller set.
    dropped = {page.id for page in golden.stale(set_name, pages)}
    report = Report(
        set_name=set_name,
        model=model or readings.name,
        unscored=sorted(dropped),
    )
    for page in pages:
        if page.id in dropped:
            continue
        path = find_reading(readings, page.id)
        if path is None:
            text, why = "", f"no reading found under {readings}"
        else:
            text = path.read_text(encoding="utf-8")
            why = "" if text.strip() else f"{path} is empty"
        result = judge(page, text, report.house)
        result.failure = why
        report.results.append(result)
    return report


def write(report: Report, *, json_path: Path | None, markdown_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(report.to_json(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report.to_markdown(), encoding="utf-8")
