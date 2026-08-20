"""The harness, on a corpus of four pages made up for the purpose.

Made up rather than read out of `tamnd/bourbaki`, so these run in CI where there
is no corpus, and so the assertions are about behaviour that is decided here
rather than about whatever page 42 of Algebra VIII happens to say this month.
"""

from __future__ import annotations

import json

import pytest

from local_ocr import corpus as corpuslib
from local_ocr import evaluate
from local_ocr.metrics import conformance
from local_ocr.rules import textguard

# The body as `extract` files it: no running head, because the head went into
# the front matter. HEAD is what the page prints above it, and READING is
# therefore what a faithful transcription of the printed page looks like.
BODY = (
    "## § 2. THE RADICAL\n"
    "\n"
    "**Proposition 4.** — Let $A$ be a ring whose radical is $\\mathfrak{r}$, and let "
    "$\\mathbf{Z}$ be the ring of rational integers. Then the canonical map "
    "$A \\to A/\\mathfrak{r}$ is surjective and its kernel is the radical, which is what "
    "was to be proved here at some length so that the page is not a short one.\n"
    "\n"
    "$$x^2 + y^2 = z^2$$\n"
)
HEAD = "THE RADICAL A VIII.13"
READING = f"{HEAD}\n\n{BODY}"


def page(
    book: str = "alg-viii",
    number: int = 42,
    body: str = BODY,
    label: str = "A VIII.13",
    head: str = "THE RADICAL",
) -> corpuslib.Page:
    return corpuslib.Page(
        book=book,
        pdf_page=number,
        method="native",
        manual=False,
        body=body,
        path=None,  # type: ignore[arg-type]
        page_label=label,
        running_head=head,
    )


class TestExpect:
    def test_the_front_matter_answers_rules_four_and_six(self):
        """Without this, two of the eight rules stand down and acceptance flatters."""
        expect = evaluate.expect_from(page())
        assert expect.has_head
        assert expect.chapter == "VIII"
        assert expect.page == 13

    def test_a_page_with_no_printed_head_says_so(self):
        expect = evaluate.expect_from(page(label="", head=""))
        assert not expect.has_head
        assert expect.page == 0

    def test_a_head_with_no_page_label_still_counts_as_a_head(self):
        """Whole volumes print a title and no label, and rule 4 applies to those."""
        expect = evaluate.expect_from(page(label="", head="THE RADICAL"))
        assert expect.has_head
        assert expect.page == 0, "rule 6 has nothing to check against, and stands down"


class TestHead:
    def test_the_head_is_taken_off_the_reading(self):
        assert evaluate.without_head(page(), READING).strip() == BODY.strip()

    def test_a_reading_with_no_head_is_left_alone(self):
        assert evaluate.without_head(page(), BODY) == BODY

    def test_a_page_that_prints_no_head_keeps_its_first_line(self):
        blank = page(label="", head="")
        assert evaluate.without_head(blank, READING) == READING

    def test_a_first_line_that_is_not_the_head_is_left_alone(self):
        reading = "The theorem below is proved in Chapter IV.\n\n" + BODY
        assert evaluate.without_head(page(), reading) == reading


class TestJudge:
    def test_a_perfect_reading(self):
        result = evaluate.judge(page(), READING, conformance.Conformance())
        assert result.prose.edits == 0
        assert result.whole.edits == 0
        assert result.accepted
        assert result.broke == []

    def test_a_reading_that_kept_its_head_is_not_charged_for_it(self):
        """The reference has no head in it. Charging for one would tax every page."""
        with_head = evaluate.judge(page(), READING, conformance.Conformance())
        without = evaluate.judge(page(), BODY, conformance.Conformance())
        assert with_head.whole.edits == 0
        assert without.whole.edits == 0

    def test_a_reading_that_lost_its_head_breaks_the_house_rule(self):
        result = evaluate.judge(page(), BODY, conformance.Conformance())
        assert "running head" in result.broke
        assert not result.accepted, "rule 4 wants the head on the first line"

    def test_a_reading_that_wrote_the_wrong_ring_breaks_the_house_rule(self):
        """The rule textguard repairs, which is the reason conformance runs before it.

        `normalise` turns `\\mathbb{Z}` back into `\\mathbf{Z}`, so a conformance
        check that ran on the normalised reading reported 100 per cent on this
        rule for a set of readings where one page in five had been deliberately
        given the wrong ring. This test is that run, written down.
        """
        wrong = READING.replace("\\mathbf{Z}", "\\mathbb{Z}")
        result = evaluate.judge(page(), wrong, conformance.Conformance())
        assert "rings" in result.broke
        assert result.whole.edits == 0, "and the repair still keeps it out of the CER"

    def test_a_reading_with_a_bare_star_breaks_the_house_rule(self):
        """The other rule normalise repairs, for the same reason."""
        head = "## § 2. THE RADICAL"
        body = BODY.replace(head, f"{head}\n\n*\n\nSee IV, p. 12.\n\n*")
        faithful = f"{HEAD}\n\n{textguard.normalise(body)}"
        loose = f"{HEAD}\n\n{body}"
        reference = page(body=textguard.normalise(body))
        assert evaluate.judge(reference, faithful, conformance.Conformance()).broke == []
        assert (
            "forward references"
            in evaluate.judge(reference, loose, conformance.Conformance()).broke
        )

    def test_a_reading_that_lost_a_word(self):
        result = evaluate.judge(
            page(), READING.replace("surjective", ""), conformance.Conformance()
        )
        assert result.prose.edits > 0

    def test_an_empty_reading_is_judged_and_not_skipped(self):
        """§05: a failed page counts as a failure, it does not leave the denominator."""
        result = evaluate.judge(page(), "", conformance.Conformance())
        assert result.prose.rate == 1.0
        assert not result.accepted

    def test_a_fenced_reading_is_unwrapped_first(self):
        """Models fence their output. That is not an error in the transcription."""
        result = evaluate.judge(page(), f"```markdown\n{READING}\n```", conformance.Conformance())
        assert result.prose.edits == 0


class TestFindReading:
    def test_the_shapes_a_reading_arrives_in(self, tmp_path):
        (tmp_path / "alg-viii").mkdir()
        (tmp_path / "alg-viii" / "0042.md").write_text("x", encoding="utf-8")
        (tmp_path / "alg-viii-0043.md").write_text("x", encoding="utf-8")
        assert evaluate.find_reading(tmp_path, "alg-viii/0042") is not None
        assert evaluate.find_reading(tmp_path, "alg-viii/0043") is not None
        assert evaluate.find_reading(tmp_path, "alg-viii/0044") is None


class TestReport:
    def build(self) -> evaluate.Report:
        report = evaluate.Report(set_name="golden-dev", model="reader-a")
        house = report.house
        report.results.append(evaluate.judge(page(), READING, house))
        missing = evaluate.judge(page(number=43), "", house)
        missing.failure = "no reading found under out/reader-a"
        report.results.append(missing)
        return report

    def test_the_failure_is_named_and_counted(self):
        report = self.build()
        blob = report.to_json()
        assert blob["pages"] == {"in_set": 2, "read": 1, "failed": 1}
        assert blob["failures"][0]["page"] == "alg-viii/0043"

    def test_the_failed_page_stays_in_the_denominators(self):
        report = self.build()
        assert report.acceptance() == 0.5
        _whole, prose = report.cer_rates()
        assert prose > 0.0

    def test_the_json_has_no_timestamp_in_it(self):
        """It is diffed between runs, and a timestamp buries the line that changed."""
        blob = json.dumps(self.build().to_json())
        assert "time" not in blob and "date" not in blob

    def test_the_json_names_the_cdm_backend(self):
        """Because it is not the published CDM and a report must not imply it is."""
        assert self.build().to_json()["cdm"]["backend"] == "mathtext"

    def test_the_markdown_says_what_it_did_not_do(self):
        text = self.build().to_markdown()
        assert "1 failed" in text
        assert "alg-viii/0043" in text

    def test_writing_both_files(self, tmp_path):
        report = self.build()
        evaluate.write(
            report,
            json_path=tmp_path / "reports" / "eval.json",
            markdown_path=tmp_path / "reports" / "eval.md",
        )
        blob = json.loads((tmp_path / "reports" / "eval.json").read_text(encoding="utf-8"))
        assert blob["model"] == "reader-a"
        assert (tmp_path / "reports" / "eval.md").read_text(encoding="utf-8").startswith("# ")

    def test_the_worst_pages_come_first(self):
        report = self.build()
        assert report.worst()[0].id == "alg-viii/0043"


class TestGuardsInTheHarness:
    def test_the_held_out_set_is_refused_here_too(self, tmp_path):
        from local_ocr import golden

        with pytest.raises(golden.Burned):
            evaluate.evaluate(
                "golden-test",
                tmp_path,
                purpose=golden.Purpose.DEVELOPMENT,
            )
