"""The four measurements, on pages small enough to check by eye.

The point of these is not coverage. It is that every number this project reports
comes out of one of these four functions, and a metric that is wrong in the
flattering direction is worse than no metric, so each one is pinned against a
case where the right answer is not a matter of opinion.
"""

from __future__ import annotations

import pytest

from local_ocr.metrics import accepted, cdm, cer, conformance
from local_ocr.rules.validate import Confidence, Expect, Grammar


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


needs_renderer = pytest.mark.skipif(
    not _has_matplotlib(), reason="formula comparison needs the eval extra"
)


class TestDistance:
    def test_the_fast_path_and_the_slow_one_agree(self):
        """What makes rapidfuzz trustworthy here.

        The slow one is short enough to read and check; the fast one is not. If
        they ever disagree the fast one is not measuring Levenshtein and every
        CER this project has reported is wrong.
        """
        cases = [
            ("", ""),
            ("", "abc"),
            ("abc", ""),
            ("kitten", "sitting"),
            ("$\\mathbf{Z}$", "$\\mathbb{Z}$"),
            ("A VIII.13 § 2.5", "A VIII.I3 § 2.5"),
            ("une base de transcendance", "une base de transcendence"),
        ]
        for a, b in cases:
            assert cer.distance(a, b) == cer._levenshtein_slow(a, b), (a, b)

    def test_the_classic(self):
        assert cer.distance("kitten", "sitting") == 3


class TestProse:
    def test_the_mathematics_comes_out(self):
        got = cer.prose_of("let $x$ be a point of $E$ here")
        assert "$" not in got
        assert "let" in got and "be a point of" in got

    def test_a_removed_span_leaves_a_gap(self):
        """Otherwise the words either side join and the diff invents an error."""
        got = cer.prose_of("a$x$b")
        assert got == "a b"

    def test_a_synonym_in_a_formula_does_not_touch_the_prose_rate(self):
        reference = "the map $\\frac{a}{b} \\to c$ is injective"
        read = "the map $\\dfrac{a}{b} \\rightarrow c$ is injective"
        _, prose = cer.page(reference, read)
        assert prose.edits == 0

    def test_a_lost_word_does(self):
        _, prose = cer.page("the map is injective", "the map is")
        assert prose.edits > 0

    def test_a_wrapped_paragraph_is_not_a_reading_error(self):
        """It is a conformance failure, and counting it twice drowns the real errors."""
        reference = "one long printed paragraph that the column broke in two places"
        read = "one long printed paragraph\nthat the column broke\nin two places"
        _, prose = cer.page(reference, read)
        assert prose.edits == 0

    def test_an_empty_reading_costs_every_character(self):
        _, prose = cer.page("a page of text", "")
        assert prose.rate == 1.0


@needs_renderer
class TestFormulas:
    def test_a_synonym_scores_one(self):
        assert cdm.compare("a \\to b", "a \\rightarrow b") == 1.0

    def test_frac_and_dfrac_agree_in_a_display_and_differ_inline(self):
        """Which is the truth about how the two print, and not a fudge.

        In a display `\\frac` is set the way `\\dfrac` is set, so there the two
        are the same formula. Inline they are not: `\\dfrac` is the instruction
        to set it large anyway, and it prints larger.
        """
        assert cdm.compare("\\frac{a}{b}", "\\dfrac{a}{b}", display=True) == 1.0
        assert cdm.compare("\\frac{a}{b}", "\\dfrac{a}{b}", display=False) < 1.0

    def test_a_lost_subscript_does_not(self):
        """The failure §05 exists to catch, and the reason CDM and not BLEU."""
        assert cdm.compare("x_1 + x_2", "x_1 + x_1") < 1.0

    def test_a_subscript_is_not_a_superscript(self):
        assert cdm.compare("x_1", "x^1") < 1.0

    def test_what_it_cannot_render_is_named_and_not_scored(self):
        with pytest.raises(cdm.Unrenderable):
            cdm.layout("\\begin{pmatrix} a & b \\end{pmatrix}")

    def test_a_page_pairs_its_spans_by_position(self):
        report = cdm.compare_pages("$a$ and $b$", "$a$ and $b$")
        assert len(report.spans) == 2
        assert report.mean == 1.0

    def test_a_missing_display_is_unpaired_and_not_a_low_score(self):
        report = cdm.compare_pages("$a$ and $$b + c$$", "$a$")
        assert report.unpaired == 1
        assert report.mean == 1.0, "the span that was read was read correctly"

    def test_a_page_with_nothing_scorable_has_no_mean(self):
        assert cdm.compare_pages("no mathematics", "no mathematics").mean is None


class TestConformance:
    def judge(self, reference: str, read: str) -> list[str]:
        house = conformance.Conformance()
        return house.observe(reference, read)

    def test_a_faithful_reading_breaks_nothing(self):
        page = (
            "A VIII.13 § 2.5\n"
            "\n"
            "## § 2. THE RADICAL\n"
            "\n"
            "**Proposition 4.** — Let $A$ be a ring and let $\\mathbf{Z}$ be the integers.\n"
        )
        assert self.judge(page, page) == []

    def test_blackboard_bold_is_caught(self):
        reference = "Let $\\mathbf{Z}$ be the integers."
        read = "Let $\\mathbb{Z}$ be the integers."
        assert "rings" in self.judge(reference, read)

    def test_a_lost_running_head_is_caught(self):
        reference = "A VIII.13 § 2.5\n\nthe text of the page\n"
        read = "the text of the page\n"
        assert "running head" in self.judge(reference, read)

    def test_a_dropped_dangerous_bend_makes_the_rule_apply_and_fail(self):
        """The rule the model cannot escape by leaving the hard thing out."""
        reference = "before\n\n☡\n\nafter"
        assert "dangerous bend" in self.judge(reference, "before\n\nafter")

    def test_a_bend_run_into_the_text_is_caught(self):
        reference = "before\n\n☡\n\nafter"
        assert "dangerous bend" in self.judge(reference, "before\n\n☡ after")

    def test_a_bare_asterisk_where_the_fence_belongs_is_caught(self):
        reference = "\\*\n\nthis leans on a later Book\n\n\\*"
        read = "*\n\nthis leans on a later Book\n\n*"
        assert "forward references" in self.judge(reference, read)

    def test_a_statement_head_that_lost_its_bold_is_caught(self):
        reference = "**Proposition 4.** — Let $A$ be a ring."
        read = "Proposition 4. — Let $A$ be a ring."
        assert "statement heads" in self.judge(reference, read)

    def test_a_heading_that_lost_its_hashes_is_caught(self):
        reference = "## § 2. THE RADICAL\n\ntext"
        assert "headings" in self.judge(reference, "§ 2. THE RADICAL\n\ntext")

    def test_a_wrapped_paragraph_is_caught(self):
        reference = "x " * 120
        read = (
            "The theorem holds for every finitely generated module over a\n"
            "noetherian ring, and the proof is the one given above.\n"
        )
        assert "paragraphs" in self.judge(reference, read)

    def test_a_footnote_without_its_note_is_caught(self):
        reference = "text[^1]\n\n[^1]: the note\n"
        assert "footnotes" in self.judge(reference, "text[^1]\n")

    def test_a_model_inventing_its_own_placeholder_is_caught(self):
        assert "illegible" in self.judge("text", "the value is [illegible] here")

    def test_a_rule_with_nothing_to_do_does_not_count(self):
        """A page with no bend on it is not evidence about the bend rule."""
        house = conformance.Conformance()
        house.observe("plain prose", "plain prose")
        assert house.counts["dangerous bend"].applicable == 0
        assert house.counts["dangerous bend"].rate is None

    def test_the_rows_come_out_in_the_order_the_spec_lists_them(self):
        house = conformance.Conformance()
        house.observe("## head\n\n☡\n", "## head\n\n☡\n")
        names = [count.name for count in house.rows()]
        assert names == [check.name for check in conformance.CHECKS if check.name in house.counts]


class TestAcceptance:
    def expect(self) -> Expect:
        return Expect(
            book="alg-viii",
            pdf_page=42,
            grammar=Grammar.HEAD_LABEL,
            chapter="VIII",
            page=13,
            confidence=Confidence.FROM_HEAD,
            has_head=True,
        )

    def test_a_full_page_with_its_head_is_accepted(self):
        page = "A VIII.13 § 2.5\n\n" + ("The radical of a ring is an ideal. " * 12)
        assert accepted(page, self.expect())

    def test_an_empty_reading_is_not(self):
        assert not accepted("", self.expect())

    def test_a_refusal_is_not(self):
        assert not accepted("I'm sorry, I can't help with that.", self.expect())
