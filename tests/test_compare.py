"""What two readers disagree about.

The comparison is the half of M6 that is deterministic and free, so it is the
half that can be tested exhaustively, and it is tested that way here. The
adjudicator gets its own file because it needs a reader.

The fixture below is a page of Bourbaki in the shape the corpus keeps it: a
running head with a folio, a display, inline mathematics, and a numbered
exercise list. Every test mutates one thing about it, which is what makes a
failure readable.
"""

from __future__ import annotations

from local_ocr.compare import (
    AGREE,
    Comparison,
    Difference,
    Severity,
    Where,
    compare,
    formulas,
    gaps,
    prose,
    structural,
    structure,
)

PAGE = """18 ALGEBRAIC STRUCTURES Ch. I

Let $G$ be a group with identity $e$. For all $x$ and $y$ in $G$ we have
$$
x \\cdot y = y \\cdot x
$$
whenever $G$ is commutative.

## EXERCISES

1. Show that the centre of $G$ is a subgroup.

2. Show that $\\aleph_0$ is the cardinal of $\\mathbf{N}$.

3. Let $H$ be a subgroup of index $2$. Show that $H$ is normal.
"""


class TestStructure:
    def test_reads_the_head(self):
        assert structure(PAGE).head == "18 ALGEBRAIC STRUCTURES Ch. I"

    def test_no_head_when_the_first_line_is_prose(self):
        assert structure("just a sentence about groups and nothing else.\n").head == ""

    def test_counts_headings(self):
        assert structure(PAGE).headings == ("EXERCISES",)

    def test_counts_exercises(self):
        assert structure(PAGE).exercises == (1, 2, 3)

    def test_a_mid_paragraph_number_is_not_an_exercise(self):
        # "see 7. above" inside a sentence is a cross reference, and counting it
        # would make every reference page look like it had lost exercises.
        text = "The result follows from 7. above, which is the lemma.\n"
        assert structure(text).exercises == ()

    def test_counts_spans_by_kind(self):
        got = structure(PAGE)
        assert got.displays == 1
        assert got.inlines > 1

    def test_typography_does_not_change_the_shape(self):
        curly = PAGE.replace('"', "“")
        assert structure(curly) == structure(PAGE)


class TestGaps:
    def test_none_in_a_consecutive_run(self):
        assert gaps((1, 2, 3, 4)) == []

    def test_finds_a_hole(self):
        assert gaps((5, 6, 8, 9)) == [7]

    def test_finds_several(self):
        assert gaps((1, 5)) == [2, 3, 4]

    def test_a_single_number_cannot_have_a_gap(self):
        assert gaps((9,)) == []

    def test_empty(self):
        assert gaps(()) == []

    def test_a_restarting_list_is_not_a_gap(self):
        # A page carrying the end of one section's exercises and the start of
        # the next prints 8, 9, 1, 2. Only the first ascending run counts.
        assert gaps((8, 9, 1, 2)) == []


class TestStructural:
    def test_identical_readings_agree(self):
        assert structural(PAGE, PAGE) == []

    def test_a_lost_head_is_high(self):
        without = PAGE.split("\n", 1)[1].lstrip()
        got = structural(PAGE, without)
        heads = [d for d in got if d.what == "running head"]
        assert len(heads) == 1
        assert heads[0].severity is Severity.HIGH

    def test_a_lost_heading_is_high(self):
        without = PAGE.replace("## EXERCISES\n\n", "")
        got = [d for d in structural(PAGE, without) if d.what == "heading count"]
        assert len(got) == 1
        assert got[0].severity is Severity.HIGH

    def test_a_misread_heading_is_medium(self):
        other = PAGE.replace("## EXERCISES", "## EXERCISE")
        got = [d for d in structural(PAGE, other) if d.what.startswith("heading ")]
        assert len(got) == 1
        assert got[0].severity is Severity.MEDIUM

    def test_different_exercise_numbers_are_high(self):
        other = PAGE.replace("3. Let $H$", "4. Let $H$")
        got = [d for d in structural(PAGE, other) if d.what == "exercise numbering"]
        assert len(got) == 1
        assert got[0].severity is Severity.HIGH

    def test_a_skipped_exercise_is_reported_even_against_itself(self):
        # This is the one signal that does not need a second reader at all, so
        # it has to fire when both readings are the same and both are wrong.
        holed = PAGE.replace("2. Show that", "4. Show that").replace("3. Let $H$", "5. Let $H$")
        got = [d for d in structural(holed, holed) if d.what == "exercise gap"]
        assert len(got) == 2  # one for each reading, and they are the same reading
        assert all(d.severity is Severity.HIGH for d in got)

    def test_a_lost_display_is_high(self):
        without = PAGE.replace("$$\nx \\cdot y = y \\cdot x\n$$\n", "")
        got = [d for d in structural(PAGE, without) if d.what == "span count"]
        assert len(got) == 1
        assert got[0].severity is Severity.HIGH

    def test_a_paragraph_break_is_low(self):
        other = PAGE.replace("whenever $G$ is commutative.", "whenever\n\n$G$ is commutative.")
        got = [d for d in structural(PAGE, other) if d.what == "paragraph count"]
        assert len(got) == 1
        assert got[0].severity is Severity.LOW


class TestFormulas:
    def test_identical_readings_agree(self):
        assert formulas(PAGE, PAGE) == []

    def test_a_synonym_is_not_a_disagreement(self):
        # The whole reason the metric is CDM. \cdot and \cdot are the same, but
        # so are \frac and \dfrac in a display, and a source comparison would
        # call the second one an error.
        left = "$$\n\\frac{a}{b}\n$$\n"
        right = "$$\n\\dfrac{a}{b}\n$$\n"
        assert formulas(left, right) == []

    def test_a_changed_subscript_is_high(self):
        other = PAGE.replace("\\aleph_0", "\\aleph_1")
        got = [d for d in formulas(PAGE, other) if d.severity is Severity.HIGH]
        assert got
        assert got[0].score is not None
        assert got[0].score < AGREE

    def test_an_unpaired_span_is_high(self):
        other = PAGE.replace("$$\nx \\cdot y = y \\cdot x\n$$", "the law is commutative")
        got = [d for d in formulas(PAGE, other) if d.severity is Severity.HIGH]
        assert got

    def test_the_score_is_carried(self):
        other = PAGE.replace("\\aleph_0", "\\aleph_1")
        got = [d for d in formulas(PAGE, other) if d.score is not None]
        assert got
        assert 0.0 <= got[0].score <= 1.0


class TestProse:
    def test_identical_readings_agree(self):
        assert prose(PAGE, PAGE) == []

    def test_normalised_typography_is_not_a_disagreement(self):
        other = PAGE.replace("identity", "identity")
        assert prose(PAGE, other) == []

    def test_a_changed_word_is_medium(self):
        other = PAGE.replace("commutative", "commutable")
        got = prose(PAGE, other)
        assert got
        assert got[0].severity is Severity.MEDIUM

    def test_a_hyphen_in_a_long_line_is_low(self):
        long = "x" * 400
        other = long[:200] + "-" + long[201:]
        got = prose(long + "\n", other + "\n")
        assert got
        assert got[0].severity is Severity.LOW

    def test_mathematics_is_not_compared_here(self):
        # formulas() owns the mathematics. If prose() saw it too, a changed
        # subscript would be reported twice and the budget would be spent twice.
        other = PAGE.replace("\\aleph_0", "\\aleph_1")
        assert prose(PAGE, other) == []


class TestCompare:
    def test_a_page_against_itself_agrees(self):
        got = compare(PAGE, PAGE)
        assert got.agreed
        assert got.worth_asking == []

    def test_low_severity_alone_still_agrees(self):
        other = PAGE.replace("whenever $G$ is commutative.", "whenever\n\n$G$ is commutative.")
        got = compare(PAGE, other)
        assert got.counts()["low"] >= 1
        assert got.counts()["high"] == 0
        assert got.agreed

    def test_worst_first(self):
        other = (
            PAGE.replace("\\aleph_0", "\\aleph_1")
            .replace("commutative", "commutable")
            .replace("## EXERCISES", "## EXERCISE")
        )
        got = compare(PAGE, other).worth_asking
        assert got
        assert got[0].severity is Severity.HIGH
        assert [d.severity for d in got] == sorted(
            (d.severity for d in got), key=lambda s: 0 if s is Severity.HIGH else 1
        )

    def test_counts_add_up(self):
        other = PAGE.replace("\\aleph_0", "\\aleph_1")
        got = compare(PAGE, other)
        assert sum(got.counts().values()) == len(got.differences)

    def test_an_empty_reading_disagrees_loudly(self):
        got = compare(PAGE, "")
        assert not got.agreed
        assert any(d.severity is Severity.HIGH for d in got.differences)

    def test_both_empty_agree(self):
        assert compare("", "").agreed


class TestComparisonObject:
    def test_empty_agrees(self):
        assert Comparison().agreed

    def test_counts_every_severity(self):
        got = Comparison(
            [
                Difference(Where.PROSE, "prose sentence 1", "a", "b", Severity.LOW, "why"),
                Difference(Where.FORMULA, "span 0", "a", "b", Severity.HIGH, "why"),
            ]
        )
        assert got.counts() == {"high": 1, "medium": 0, "low": 1}
        assert len(got.worth_asking) == 1

    def test_a_difference_prints_itself(self):
        one = Difference(Where.FORMULA, "span 3", "a", "b", Severity.HIGH, "CDM 0.500")
        assert one.line() == "formula span 3: high, CDM 0.500"
