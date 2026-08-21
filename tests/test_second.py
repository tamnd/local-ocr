"""Two readers, and what happens when they disagree.

The comparison is tested on fixtures next door. This file tests the part that
spends money, so every reader here is a stub that answers from a dictionary and
counts how many times it was asked. Counting the asks is half the point: the
budget is the thing that keeps M6 from tripling the cost of every page, and a
budget that is not enforced is a budget nobody notices until the throughput
number arrives.
"""

from __future__ import annotations

import asyncio

from PIL import Image

from local_ocr.batch import Refused
from local_ocr.compare import Difference, Severity, Where
from local_ocr.second import (
    BUDGET,
    CROPS,
    SecondPass,
    Step,
    Winner,
    caught,
    crop,
    mark_illegible,
    settles,
    where_of,
)
from local_ocr.sidecar import Record

PAGE = """18 ALGEBRAIC STRUCTURES Ch. I

Let $G$ be a group with identity $e$. For all $x$ and $y$ in $G$ we have
$$
x \\cdot y = y \\cdot x
$$
whenever $G$ is commutative.

## EXERCISES

1. Show that the centre of $G$ is a subgroup.

2. Show that $\\aleph_0$ is the cardinal of $\\mathbf{N}$.
"""

OTHER = PAGE.replace("\\aleph_0", "\\aleph_1")


def page_image(tmp_path, name="0027.png", size=(1831, 2776)):
    path = tmp_path / name
    Image.new("L", size, color=255).save(path)
    return path


class Stub:
    """A reader that answers from a list and remembers what it was asked."""

    def __init__(self, *answers, refuse=False):
        self.answers = list(answers)
        self.asked: list[tuple[str, str]] = []
        self.refuse = refuse

    async def read(self, image, prompt):
        self.asked.append((image.name, prompt))
        if self.refuse:
            raise Refused("no")
        if not self.answers:
            return ""
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class TestWhereOf:
    def test_finds_the_line(self):
        text = "\n".join(f"line {n}" for n in range(10))
        one = Difference(Where.PROSE, "prose sentence 1", "line 8", "line 8!", Severity.MEDIUM, "")
        assert where_of(one, text) > 0.7

    def test_the_middle_when_it_cannot_tell(self):
        one = Difference(Where.FORMULA, "span 0", "nowhere", "", Severity.HIGH, "")
        assert where_of(one, "a\nb\nc") == 0.5

    def test_the_middle_for_an_empty_reading(self):
        one = Difference(Where.FORMULA, "span 0", "x", "", Severity.HIGH, "")
        assert where_of(one, "") == 0.5

    def test_it_falls_back_to_the_second_reading(self):
        text = "alpha\nbeta\ngamma"
        one = Difference(Where.FORMULA, "span 0", "", "gamma", Severity.HIGH, "")
        assert where_of(one, text) > 0.5


class TestCrop:
    def test_it_is_full_width(self, tmp_path):
        page = page_image(tmp_path)
        out = crop(page, tmp_path / "band.png", 0.5, 0.1)
        with Image.open(out) as got:
            assert got.width == 1831

    def test_the_band_is_the_fraction_asked_for(self, tmp_path):
        page = page_image(tmp_path)
        out = crop(page, tmp_path / "band.png", 0.5, 0.25)
        with Image.open(out) as got:
            assert abs(got.height - 2776 * 0.25) <= 1

    def test_a_band_at_the_top_does_not_run_off_the_page(self, tmp_path):
        page = page_image(tmp_path)
        out = crop(page, tmp_path / "band.png", 0.0, 0.2)
        with Image.open(out) as got:
            assert got.height == int(2776 * 0.2)

    def test_a_band_at_the_bottom_does_not_either(self, tmp_path):
        page = page_image(tmp_path)
        out = crop(page, tmp_path / "band.png", 1.0, 0.2)
        with Image.open(out) as got:
            assert got.height == int(2776 * 0.2)

    def test_the_second_rung_is_smaller_than_the_first(self):
        # The whole reason there are two rungs. If they were the same size the
        # second look would be the first look again.
        assert CROPS[1] < CROPS[0]


class TestSettles:
    def one(self):
        return Difference(
            Where.FORMULA, "span 3", "\\aleph_0", "\\aleph_1", Severity.HIGH, "CDM 0.5"
        )

    def test_the_first_reading(self):
        assert settles("the cardinal is \\aleph_0 here", self.one()) is Winner.FIRST

    def test_the_second_reading(self):
        assert settles("the cardinal is \\aleph_1 here", self.one()) is Winner.SECOND

    def test_neither(self):
        assert settles("the cardinal is \\aleph_2", self.one()) is Winner.UNKNOWN

    def test_both_is_not_a_verdict(self):
        assert settles("\\aleph_0 and \\aleph_1", self.one()) is Winner.UNKNOWN

    def test_an_empty_answer_is_not_a_verdict(self):
        assert settles("", self.one()) is Winner.UNKNOWN
        assert settles("   \n ", self.one()) is Winner.UNKNOWN

    def test_whitespace_does_not_matter(self):
        assert settles("the  cardinal\nis \\aleph_0", self.one()) is Winner.FIRST


class TestMarkIllegible:
    def test_it_replaces_the_losing_text(self):
        one = Difference(Where.FORMULA, "span 0", "\\aleph_0", "\\aleph_1", Severity.HIGH, "")
        got = mark_illegible("the cardinal \\aleph_0 is infinite", one)
        assert "⟪illegible⟫" in got
        assert "\\aleph_0" not in got

    def test_it_replaces_once(self):
        one = Difference(Where.FORMULA, "span 0", "x", "y", Severity.HIGH, "")
        assert mark_illegible("x and x", one).count("⟪illegible⟫") == 1

    def test_a_structural_difference_changes_nothing(self):
        one = Difference(Where.STRUCTURE, "heading count", "2", "3", Severity.HIGH, "")
        assert mark_illegible("2 headings here", one) == "2 headings here"

    def test_text_that_is_not_there_changes_nothing(self):
        one = Difference(Where.FORMULA, "span 0", "nowhere", "y", Severity.HIGH, "")
        assert mark_illegible("the page", one) == "the page"


class TestNoReferee:
    def test_it_reads_with_one_and_says_so(self, tmp_path):
        page = page_image(tmp_path)
        said = []
        pipe = SecondPass(first=Stub(PAGE))
        pipe.log = said.append
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.text == PAGE
        assert out.record.referee_ran is False
        assert said == ["no referee configured, running on one reader"]

    def test_it_says_so_once(self, tmp_path):
        page = page_image(tmp_path)
        said = []
        pipe = SecondPass(first=Stub(PAGE))
        pipe.log = said.append
        for _ in range(5):
            asyncio.run(pipe.look(page, "read this"))
        assert len(said) == 1

    def test_the_sidecar_still_records_the_first_reader(self, tmp_path):
        page = page_image(tmp_path)
        out = asyncio.run(SecondPass(first=Stub(PAGE)).look(page, "read this"))
        assert out.record.first is not None
        assert out.record.first.text_sha256
        assert out.record.image_sha256


class TestAgreement:
    def test_two_identical_readings_cost_nothing(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("should never be asked")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(PAGE), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.agreed is True
        assert out.record.referee_ran is True
        assert judge.asked == []
        assert pipe.disagreed == 0

    def test_typography_alone_is_still_agreement(self, tmp_path):
        page = page_image(tmp_path)
        other = PAGE.replace("whenever $G$ is commutative.", "whenever\n\n$G$ is commutative.")
        judge = Stub("never")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(other), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.agreed is True
        assert judge.asked == []

    def test_the_first_reading_is_what_is_written(self, tmp_path):
        page = page_image(tmp_path)
        pipe = SecondPass(first=Stub(PAGE), second=Stub(PAGE))
        assert asyncio.run(pipe.read(page, "read this")) == PAGE


class TestRefereeRefuses:
    def test_the_first_reading_stands(self, tmp_path):
        page = page_image(tmp_path)
        pipe = SecondPass(first=Stub(PAGE), second=Stub(refuse=True))
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.text == PAGE
        assert out.record.referee_ran is False

    def test_the_sidecar_says_nobody_asked_rather_than_they_agreed(self, tmp_path):
        page = page_image(tmp_path)
        out = asyncio.run(SecondPass(first=Stub(PAGE), second=Stub(refuse=True)).look(page, "p"))
        assert out.record.second is not None
        assert out.record.second.refused == "no"
        assert out.record.referee_ran is False


class TestAdjudication:
    def test_a_disagreement_is_sent_to_the_adjudicator(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("the cardinal is \\aleph_1")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.agreed is False
        assert judge.asked
        assert out.record.adjudicated

    def test_the_crop_wins_it_on_the_first_rung(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("the cardinal is \\aleph_1")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        formula = [a for a in out.record.adjudicated if a.where == "formula"]
        assert formula
        assert formula[0].step == str(Step.CROP)
        assert formula[0].winner == str(Winner.SECOND)

    def test_the_referee_winning_writes_the_referees_reading(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("the cardinal is \\aleph_1")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.chose == "second"
        assert out.text == OTHER
        assert pipe.overruled >= 1

    def test_the_primary_winning_keeps_the_primarys_reading(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("the cardinal is \\aleph_0")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.chose == "first"
        assert out.text.startswith("18 ALGEBRAIC")
        assert pipe.overruled == 0

    def test_it_climbs_to_the_second_rung(self, tmp_path):
        page = page_image(tmp_path)
        # First crop settles nothing, second one does. The two answers are
        # popped in order, so this asserts the ladder is climbed rather than
        # the same rung being tried twice.
        judge = Stub("something else entirely", "the cardinal is \\aleph_1")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        formula = [a for a in out.record.adjudicated if a.where == "formula"]
        assert formula[0].step == str(Step.REREAD)
        assert len(judge.asked) >= 2

    def test_two_crops_that_settle_nothing_mark_it_illegible(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("neither one nor the other")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        formula = [a for a in out.record.adjudicated if a.where == "formula"]
        assert formula[0].step == str(Step.ILLEGIBLE)
        assert formula[0].winner == str(Winner.NEITHER)
        assert "⟪illegible⟫" in out.text
        assert pipe.marked >= 1

    def test_an_adjudicator_that_refuses_does_not_crash_the_page(self, tmp_path):
        page = page_image(tmp_path)
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=Stub(refuse=True))
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.text
        assert out.record.adjudicated

    def test_the_referee_is_the_adjudicator_by_default(self, tmp_path):
        page = page_image(tmp_path)
        referee = Stub(OTHER, "the cardinal is \\aleph_1")
        pipe = SecondPass(first=Stub(PAGE), second=referee)
        asyncio.run(pipe.look(page, "read this"))
        # One page read plus at least one crop.
        assert len(referee.asked) >= 2

    def test_a_structural_difference_is_recorded_and_not_spent_on(self, tmp_path):
        page = page_image(tmp_path)
        without = PAGE.replace("## EXERCISES\n\n", "")
        judge = Stub("anything")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(without), adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        structural = [a for a in out.record.adjudicated if a.where == "structure"]
        assert structural
        assert all(a.step == str(Step.BUDGET) for a in structural)


class TestBudget:
    def many(self):
        """A reading that disagrees in five places at once."""
        return (
            PAGE.replace("\\aleph_0", "\\aleph_1")
            .replace("commutative", "commutable")
            .replace("## EXERCISES", "## EXERCISE")
            .replace("centre", "center")
            .replace("identity", "unit")
        )

    def test_it_stops_at_the_budget(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("nothing that matches either")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(self.many()), budget=2, adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert len(out.record.adjudicated) + out.record.unadjudicated >= 3
        assert pipe.spent == 2

    def test_what_it_could_not_pay_for_is_counted(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("nothing that matches either")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(self.many()), budget=1, adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.unadjudicated >= 1

    def test_a_budget_of_zero_spends_nothing(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("anything")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(self.many()), budget=0, adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        assert judge.asked == []
        assert out.record.unadjudicated >= 1
        assert pipe.spent == 0

    def test_the_default_is_three(self):
        assert BUDGET == 3

    def test_the_worst_difference_is_paid_for_first(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("nothing that matches")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(self.many()), budget=1, adjudicator=judge)
        out = asyncio.run(pipe.look(page, "read this"))
        paid = [a for a in out.record.adjudicated if a.step != str(Step.BUDGET)]
        assert paid
        assert paid[0].severity == "high"


class TestMeasurement:
    def test_the_dpi_is_inferred(self, tmp_path):
        page = page_image(tmp_path, size=(1831, 2776))
        out = asyncio.run(SecondPass(first=Stub(PAGE)).look(page, "read this"))
        assert out.record.dpi == 300
        assert out.record.width == 1831

    def test_six_hundred_dpi_is_told_apart(self, tmp_path):
        page = page_image(tmp_path, size=(3662, 5552))
        out = asyncio.run(SecondPass(first=Stub(PAGE)).look(page, "read this"))
        assert out.record.dpi == 600

    def test_an_odd_size_is_left_at_zero_rather_than_guessed(self, tmp_path):
        page = page_image(tmp_path, size=(400, 400))
        out = asyncio.run(SecondPass(first=Stub(PAGE)).look(page, "read this"))
        assert out.record.dpi == 0
        assert out.record.height == 400

    def test_the_summary_line_says_what_it_cost_and_bought(self, tmp_path):
        page = page_image(tmp_path)
        judge = Stub("the cardinal is \\aleph_1")
        pipe = SecondPass(first=Stub(PAGE), second=Stub(OTHER), adjudicator=judge)
        asyncio.run(pipe.look(page, "read this"))
        line = pipe.summary()
        assert "1 pages" in line
        assert "1 disagreed" in line
        assert "overruled" in line


class Priced(Stub):
    """A reader that reports what the server charged, one page at a time."""

    def __init__(self, *answers, prompt=1400, completion=700):
        super().__init__(*answers)
        self.price = (prompt, completion)
        self.collected: list[str] = []

    def usage(self, image):
        self.collected.append(image.name)
        return self.price


class TestTokens:
    """The sidecar field that read zero on every page of the M6 run.

    122 pages were read with both readers and adjudicated up to three times each,
    and not one of the sidecars could say what any of it cost, because `read`
    hands back a string and the counts were thrown away with the response. They
    are collected on the side now, and a zero here means nobody counted rather
    than nothing was spent.
    """

    def test_the_first_reader_is_charged_to_the_sidecar(self, tmp_path):
        page = page_image(tmp_path)
        out = asyncio.run(SecondPass(first=Priced(PAGE)).look(page, "read this"))
        assert out.record.first.prompt_tokens == 1400
        assert out.record.first.completion_tokens == 700

    def test_the_referee_is_charged_separately(self, tmp_path):
        page = page_image(tmp_path)
        pipe = SecondPass(
            first=Priced(PAGE),
            second=Priced(PAGE, prompt=900, completion=640),
        )
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.first.prompt_tokens == 1400
        assert out.record.second.prompt_tokens == 900
        assert out.record.second.completion_tokens == 640

    def test_a_reader_that_cannot_count_writes_zeros_and_keeps_working(self, tmp_path):
        # codex is a subprocess against a subscription and reports nothing. A
        # sidecar of zeros is what it wrote before this existed, and breaking it
        # for a token count would be a poor trade.
        page = page_image(tmp_path)
        out = asyncio.run(SecondPass(first=Stub(PAGE)).look(page, "read this"))
        assert out.record.first.prompt_tokens == 0
        assert out.record.first.completion_tokens == 0

    def test_a_reader_whose_usage_raises_does_not_lose_the_page(self, tmp_path):
        class Angry(Stub):
            def usage(self, image):
                raise RuntimeError("no")

        page = page_image(tmp_path)
        out = asyncio.run(SecondPass(first=Angry(PAGE)).look(page, "read this"))
        assert out.text == PAGE
        assert out.record.first.prompt_tokens == 0

    def test_a_reader_that_answers_something_else_is_not_believed(self, tmp_path):
        class Odd(Stub):
            def usage(self, image):
                return 4096

        page = page_image(tmp_path)
        out = asyncio.run(SecondPass(first=Odd(PAGE)).look(page, "read this"))
        assert out.record.first.prompt_tokens == 0

    def test_a_refused_referee_is_not_charged(self, tmp_path):
        page = page_image(tmp_path)
        pipe = SecondPass(first=Priced(PAGE), second=Stub(refuse=True))
        out = asyncio.run(pipe.look(page, "read this"))
        assert out.record.second.refused
        assert out.record.second.prompt_tokens == 0


class TestCaught:
    def test_a_disagreement_on_a_page_that_passed_is_a_catch(self):
        record = Record(referee_ran=True, agreed=False, gates={"head": "ok", "math": "ok"})
        assert caught([record])["caught"] == 1

    def test_a_disagreement_on_a_page_that_failed_is_not(self):
        # That page was going to be read again anyway, so the referee bought
        # nothing. This is the distinction the milestone number turns on.
        record = Record(referee_ran=True, agreed=False, gates={"head": "no head found"})
        assert caught([record])["caught"] == 0

    def test_agreement_is_not_a_catch(self):
        record = Record(referee_ran=True, agreed=True, gates={"head": "ok"})
        got = caught([record])
        assert got["caught"] == 0
        assert got["disagreed"] == 0

    def test_a_page_with_no_referee_counts_as_a_page_and_nothing_else(self):
        got = caught([Record(referee_ran=False, agreed=True)])
        assert got["pages"] == 1
        assert got["passed_gates"] == 0
        assert got["disagreed"] == 0

    def test_high_severity_differences_are_totalled(self):
        record = Record(
            referee_ran=True, agreed=False, counts={"high": 3, "low": 9}, gates={"head": "ok"}
        )
        assert caught([record])["high"] == 3

    def test_nothing_in_nothing_out(self):
        assert caught([])["pages"] == 0


class TestGates:
    """The acceptance rules, recorded against the primary's reading.

    Recorded rather than acted on. A page that fails a rule is still written and
    still compared; the gate verdict only decides whether a disagreement on it
    counts as a catch, and getting that backwards would turn the milestone
    number into a count of pages that were going to be read again anyway.
    """

    def read(self, tmp_path, first, second=None):
        page = page_image(tmp_path)
        pass_ = SecondPass(Stub(first), Stub(second) if second else None)
        return asyncio.run(pass_.look(page, "read this")).record

    def test_a_clean_reading_passes(self, tmp_path):
        assert self.read(tmp_path, PAGE).gates == {"rules": "ok"}

    def test_a_short_reading_does_not(self, tmp_path):
        got = self.read(tmp_path, "18 ALGEBRAIC STRUCTURES Ch. I\n\nToo little.\n")
        assert "short" in got.gates

    def test_an_unbalanced_dollar_does_not(self, tmp_path):
        got = self.read(tmp_path, PAGE.replace("$G$ be a group", "$G be a group"))
        assert "math" in got.gates

    def test_the_gate_is_run_on_the_primary_and_not_the_referee(self, tmp_path):
        # The claim the milestone makes is about the reading that would have been
        # written if nobody had asked a second reader, so that is the reading the
        # rules run on.
        short = "18 ALGEBRAIC STRUCTURES Ch. I\n\nToo little.\n"
        got = self.read(tmp_path, short, PAGE)
        assert "short" in got.gates

    def test_a_page_that_passed_and_disagreed_is_the_number(self, tmp_path):
        got = self.read(tmp_path, PAGE, OTHER)
        assert got.gates == {"rules": "ok"}
        assert got.agreed is False
        assert caught([got])["caught"] == 1
