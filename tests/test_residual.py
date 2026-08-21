"""The residual failure report, which decides what a fine tune is trained on.

Every test here builds its reference and its reading by hand, because the whole
module is a claim about how one particular difference between two texts should
be classified, and a fixture drawn from the corpus would hide which difference
is doing the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from local_ocr import residual


@dataclass
class FakePage:
    """Enough of a corpus page for `judge`, which wants an id and a body."""

    id: str
    body: str
    head: str = ""

    @property
    def page_label(self) -> str:
        return self.head


def write(root: Path, name: str, pages: dict[str, str]) -> Path:
    where = root / name
    for page_id, text in pages.items():
        path = where / f"{page_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return where


class TestSurvived:
    def test_a_head_that_lost_its_markup_still_opens_its_line(self):
        assert residual._survived("**Lemme 5.**", "Lemme 5. — Pour toute extension")

    def test_a_head_that_is_not_on_the_page_did_not_survive(self):
        assert not residual._survived("**Lemme 5.**", "Proposition 4. — Soient A et B")

    def test_a_head_buried_mid_sentence_is_not_a_head(self):
        """The reason this is not a substring test.

        `Symboles` inside a sentence about symbols is the word, not the
        heading, and calling it a surviving heading would report a page that
        lost its structure as a page that merely lost its hashes.
        """
        assert not residual._survived("Symboles", "on trouvera les Symboles au tome II")

    def test_the_hashes_themselves_do_not_have_to_be_there(self):
        assert residual._survived("### 1. Modules semi-simples", "1. Modules semi-simples")

    def test_an_empty_head_never_survives(self):
        assert not residual._survived("", "anything at all")

    def test_present_finds_a_passage_anywhere(self):
        assert residual._present("une famille finie", "cf. I, p. 36, une famille finie de modules")
        assert not residual._present("une famille finie", "cf. I, p. 36, deux modules")


class TestAttribution:
    def test_a_heading_written_as_running_text_is_convention(self):
        which, note = residual._attribute_headings(
            "A VIII.51\n\n### 1. Modules semi-simples\n\nOn dit qu'un module",
            "A VIII.51\n\n1. Modules semi-simples\n\nOn dit qu'un module",
        )
        assert which == residual.CONVENTION
        assert "hashes" in note

    def test_a_heading_that_is_not_on_the_page_is_capability(self):
        which, _ = residual._attribute_headings(
            "A VIII.51\n\n### 1. Modules semi-simples\n\nOn dit qu'un module",
            "A VIII.51\n\nOn dit qu'un module",
        )
        assert which == residual.CAPABILITY

    def test_a_statement_head_that_lost_its_bold_is_convention(self):
        which, note = residual._attribute_statement_heads(
            "**Définition 1.** — On dit qu'un module est semi-simple",
            "Définition 1. — On dit qu'un module est semi-simple",
        )
        assert which == residual.CONVENTION
        assert "unbolded" in note

    def test_a_running_head_missing_only_its_page_label_is_capability(self):
        """The margin failure, and the one the prompt was never going to reach."""
        which, note = residual._attribute_running_head(
            "ANNEAUX SEMI-SIMPLES A VIII.138\n\nbody",
            "ANNEAUX SEMI-SIMPLES\n\nbody",
        )
        assert which == residual.CAPABILITY
        assert "margin" in note

    def test_a_running_head_that_is_a_bare_label_is_convention(self):
        which, note = residual._attribute_running_head(
            "A VIII.51\n\n§ 4. MODULES SEMI-SIMPLES\n\nbody",
            "§ 4. MODULES SEMI-SIMPLES\n\nbody",
        )
        assert which == residual.CONVENTION
        assert "bare label" in note

    def test_a_running_head_pushed_off_the_first_line_is_convention(self):
        which, note = residual._attribute_running_head(
            "ANNEAUX SEMI-SIMPLES A VIII.144\n\nbody",
            "EXERCICES\n\nANNEAUX SEMI-SIMPLES A VIII.144\n\nbody",
        )
        assert which == residual.CONVENTION
        assert "first line" in note

    def test_a_footnote_kept_under_the_printed_mark_is_convention(self):
        which, note = residual._attribute_footnotes(
            "modules simples[^1].\n\n[^1]: Voir aussi le chapitre suivant sur ce point.",
            "modules simples(1).\n\n(1) Voir aussi le chapitre suivant sur ce point.",
        )
        assert which == residual.CONVENTION
        assert "printed mark" in note

    def test_a_footnote_that_is_gone_is_capability(self):
        which, _ = residual._attribute_footnotes(
            "modules simples[^1].\n\n[^1]: Voir aussi le chapitre suivant sur ce point.",
            "modules simples.",
        )
        assert which == residual.CAPABILITY

    def test_only_the_four_rules_with_a_test_are_attributed(self):
        """The rest are reported unattributed rather than guessed at."""
        assert set(residual.ATTRIBUTORS) == {
            "headings",
            "statement heads",
            "running head",
            "footnotes",
        }


class TestMoved:
    def _readings(self, rates: list[tuple[int, int]]) -> list[residual.Reading]:
        out = []
        for i, (obeyed, applicable) in enumerate(rates):
            reading = residual.Reading(f"r{i}")
            reading.verdicts = [residual.Verdict("headings", applicable, obeyed)]
            out.append(reading)
        return out

    def test_a_rule_the_prompt_moved_reports_its_spread(self):
        readings = self._readings([(0, 100), (74, 100), (80, 100)])
        assert residual.moved(readings, "headings") == 0.80

    def test_a_rule_the_prompt_never_moved_reports_zero(self):
        readings = self._readings([(0, 60), (0, 60), (0, 60)])
        assert residual.moved(readings, "headings") == 0.0

    def test_one_reading_has_no_spread_to_report(self):
        assert residual.moved(self._readings([(0, 60)]), "headings") is None

    def test_a_rule_nobody_measured_has_no_spread(self):
        assert residual.moved(self._readings([(0, 60), (1, 60)]), "footnotes") is None

    def test_the_table_says_which_rules_the_prompt_can_still_reach(self):
        readings = self._readings([(0, 60), (0, 60)])
        table = residual.table(readings)
        assert "| headings |" in table
        line = next(row for row in table.split("\n") if row.startswith("| headings |"))
        assert line.rstrip().endswith("| no |")

    def test_a_rule_that_never_applied_is_not_called_unreachable(self):
        reading = residual.Reading("r0")
        reading.verdicts = [residual.Verdict("forward references", 0, 0)]
        table = residual.table([reading, reading])
        line = next(row for row in table.split("\n") if row.startswith("| forward references |"))
        assert "did not apply" in line
        assert "| n/a | n/a |" in line


class TestReport:
    def test_the_report_carries_the_tables_and_the_notes(self):
        reading = residual.Reading("head3", pages=200)
        verdict = residual.Verdict("headings", 60, 0, convention=56, capability=4)
        verdict.notes["the heading is there as running text, without its hashes"] = 55
        reading.verdicts = [verdict]
        reading.formulas = residual.Formulas(
            scored=8000, exact=5522, unpaired=2730, total=10839, mean=0.7841
        )
        text = residual.report([reading], "golden-dev")
        assert "# Residual failures on golden-dev" in text
        assert "200 pages, 1 revisions of the prompt." in text
        assert "| headings | 60 of 60 | 56 | 4 | 0 |" in text
        assert "- 55 pages: the heading is there as running text" in text
        assert "0.7841" in text
        assert "2730" in text

    def test_a_rule_with_no_failures_is_left_out_of_the_attribution(self):
        reading = residual.Reading("head3", pages=200)
        reading.verdicts = [
            residual.Verdict("illegible", 200, 200),
            residual.Verdict("headings", 60, 0, convention=60),
        ]
        assert "| illegible |" not in residual.attribution(reading)
        assert "| headings |" in residual.attribution(reading)
