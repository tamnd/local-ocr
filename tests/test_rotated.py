"""The rotated small type set: what it draws, what it refuses, and how it scores."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from local_ocr import kvant, rotated

CREDIT = "Иллюстрации"
NAME = "Д.Гришуковой"


@dataclass
class Done:
    stdout: str


def bbox(pages: list[list[tuple[float, float, float, float, str]]]) -> str:
    """A `pdftotext -bbox-layout` document holding exactly these words."""
    out = ['<?xml version="1.0"?>', "<doc>"]
    for page in pages:
        out.append('<page width="595.00" height="842.00">')
        for x0, y0, x1, y1, text in page:
            out.append(
                f'<word xMin="{x0}" yMin="{y0}" xMax="{x1}" yMax="{y1}">'
                f"{text.replace('&', '&amp;').replace('<', '&lt;')}</word>"
            )
        out.append("</page>")
    out.append("</doc>")
    return "\n".join(out)


def test_the_parser_reads_a_word_and_its_box_off_each_page_in_order():
    doc = bbox(
        [
            [(10.0, 20.0, 30.0, 32.0, "первая")],
            [(11.0, 21.0, 19.9, 76.1, CREDIT), (40.0, 50.0, 60.0, 62.0, "вторая")],
        ]
    )
    read = rotated.boxes(Path("x.pdf"), run=lambda *_a, **_k: Done(doc))
    assert len(read) == 2
    assert read[0] == [(10.0, 20.0, 30.0, 32.0, "первая")]
    assert read[1][0][4] == CREDIT
    assert read[1][1][4] == "вторая"


def test_the_parser_puts_an_escaped_ampersand_back_the_way_the_page_had_it():
    doc = bbox([[(1.0, 2.0, 3.0, 4.0, "Тела & точки")]])
    read = rotated.boxes(Path("x.pdf"), run=lambda *_a, **_k: Done(doc))
    assert read[0][0][4] == "Тела & точки"


def test_a_page_with_no_words_at_all_is_still_a_page():
    """So that the PDF page numbers stay one based against `align`.

    A blank sheet that vanished from the list would shift every page after it
    by one, and the map into the corpus would then be silently off.
    """
    doc = bbox([[], [(11.0, 21.0, 19.9, 76.1, CREDIT)]])
    read = rotated.boxes(Path("x.pdf"), run=lambda *_a, **_k: Done(doc))
    assert read[0] == []
    assert read[1][0][4] == CREDIT


def test_a_tall_narrow_cyrillic_box_is_sideways():
    assert rotated.sideways(11.0, 21.0, 19.9, 77.0, CREDIT)


def test_an_ordinary_horizontal_word_in_a_slightly_tall_box_is_not():
    """The measured noise floor, which is why RATIO is where it is."""
    assert not rotated.sideways(100.0, 100.0, 112.5, 113.0, "А.И.")


def test_a_word_in_a_box_too_wide_across_is_not_taken():
    assert not rotated.sideways(0.0, 0.0, 48.3, 97.9, "тела")


def test_a_short_run_is_not_taken_however_upended_it_is():
    """`Рис.` at four characters, which is a real rotated run and a poor test item."""
    assert not rotated.sideways(0.0, 0.0, 15.8, 34.2, "Рис.")


def test_a_tall_narrow_latin_box_is_not_taken():
    assert not rotated.sideways(11.0, 21.0, 19.9, 77.0, "Figure")


def test_a_degenerate_box_is_refused_rather_than_dividing_by_zero():
    assert not rotated.sideways(11.0, 21.0, 11.0, 77.0, CREDIT)
    assert not rotated.sideways(11.0, 21.0, 19.9, 21.0, CREDIT)


def test_the_runs_of_an_issue_carry_the_pdf_page_they_were_found_on_one_based():
    doc = bbox(
        [
            [(1.0, 2.0, 3.0, 4.0, "обычный")],
            [(11.0, 21.0, 19.9, 77.0, CREDIT), (30.0, 21.0, 38.9, 79.0, NAME)],
        ]
    )
    found = rotated.runs(
        "kvant_2018_10",
        Path("x.pdf"),
        read=lambda _p: rotated.boxes(Path("x.pdf"), run=lambda *_a, **_k: Done(doc)),
    )
    assert [one.pdf_page for one in found] == [2, 2]
    assert [one.text for one in found] == [CREDIT, NAME]
    assert found[0].issue == "kvant_2018_10"
    assert found[0].width == pytest.approx(8.9)
    assert found[0].ratio == pytest.approx(56.0 / 8.9)


def test_a_run_with_a_zero_width_box_reports_a_ratio_of_zero_and_not_an_error():
    """Nothing in the set has one, because `sideways` refuses it first.

    The property is still total, because `Run` is what gets written to the
    manifest and read back, and a manifest edited by hand should not be able to
    make a card raise.
    """
    assert rotated.Run("i", 1, CREDIT, 0.0, 50.0).ratio == 0.0


def fake_align(mapping: dict[str, int]):
    return lambda chosen, pdf, text=None: dict(mapping)


def test_the_sheet_map_is_the_inverse_of_align(monkeypatch):
    monkeypatch.setattr(kvant, "align", fake_align({"kvant_2018_10/0016": 18}))
    pages = [
        kvant.Page("kvant_2018_10", 2018, 16, "14", "native", "тело", Path("a.md")),
        kvant.Page("kvant_2018_11", 2018, 16, "14", "native", "тело", Path("b.md")),
    ]
    assert rotated.sheets("kvant_2018_10", pages, Path("x.pdf"), text=lambda _p: []) == {
        18: "kvant_2018_10/0016"
    }


def test_a_page_align_left_out_is_not_in_the_map_and_so_not_in_the_set(monkeypatch):
    monkeypatch.setattr(kvant, "align", fake_align({}))
    pages = [kvant.Page("kvant_2018_10", 2018, 16, "14", "native", "тело", Path("a.md"))]
    assert rotated.sheets("kvant_2018_10", pages, Path("x.pdf"), text=lambda _p: []) == {}


def build(tmp_path: Path, monkeypatch, *, held: set[str] | None = None):
    """A corpus of one issue whose PDF page 18 carries both runs."""
    doc = bbox(
        [
            [],
            [(11.0, 21.0, 19.9, 77.0, CREDIT), (30.0, 21.0, 38.9, 79.0, NAME)],
        ]
        * 9
    )
    pages = [kvant.Page("kvant_2018_10", 2018, 16, "14", "native", "тело", tmp_path / "a.md")]
    monkeypatch.setattr(kvant, "pages", lambda _c: pages)
    monkeypatch.setattr(kvant, "scan", lambda _i, _s: tmp_path / "x.pdf")
    monkeypatch.setattr(kvant, "align", fake_align({"kvant_2018_10/0016": 18}))
    read = lambda _p: rotated.boxes(Path("x.pdf"), run=lambda *_a, **_k: Done(doc))  # noqa: E731
    return read, (lambda: set(held or ()))


def test_the_draw_keeps_the_page_align_placed_and_both_of_its_runs(tmp_path, monkeypatch):
    read, avoid = build(tmp_path, monkeypatch)
    chosen, where = rotated.draw(tmp_path, tmp_path, read=read, text=lambda _p: [], avoid=avoid)
    assert chosen == ["kvant_2018_10/0016"]
    assert [one.text for one in where["kvant_2018_10/0016"]] == [CREDIT, NAME]


def test_a_run_on_a_sheet_the_corpus_does_not_hold_is_named_and_left_out(tmp_path, monkeypatch):
    read, avoid = build(tmp_path, monkeypatch)
    said: list[str] = []
    chosen, _where = rotated.draw(
        tmp_path, tmp_path, read=read, text=lambda _p: [], avoid=avoid, say=said.append
    )
    assert chosen == ["kvant_2018_10/0016"]
    # Pages 2, 4, 6, 8, 10, 12, 14, 16 carry runs too, and align placed 18 only.
    assert len(said) == 16
    assert all("is not one of this issue's corpus pages" in line for line in said)


def test_a_held_out_page_is_named_and_left_out_so_the_set_needs_no_purpose_gate(
    tmp_path, monkeypatch
):
    read, avoid = build(tmp_path, monkeypatch, held={"kvant_2018_10/0016"})
    said: list[str] = []
    chosen, where = rotated.draw(
        tmp_path, tmp_path, read=read, text=lambda _p: [], avoid=avoid, say=said.append
    )
    assert chosen == []
    assert where == {}
    assert sum("held out in kvant-test" in line for line in said) == 2


def test_an_issue_whose_pdf_will_not_open_costs_that_issue_and_not_the_draw(tmp_path, monkeypatch):
    pages = [kvant.Page("kvant_2018_10", 2018, 16, "14", "native", "тело", tmp_path / "a.md")]
    monkeypatch.setattr(kvant, "pages", lambda _c: pages)
    monkeypatch.setattr(kvant, "scan", lambda _i, _s: tmp_path / "x.pdf")

    def boom(_pdf):
        raise subprocess.CalledProcessError(1, "pdftotext")

    said: list[str] = []
    chosen, _where = rotated.draw(
        tmp_path, tmp_path, read=boom, text=lambda _p: [], avoid=set, say=said.append
    )
    assert chosen == []
    assert len(said) == 1


def test_an_issue_with_no_cached_pdf_is_skipped_in_silence(tmp_path, monkeypatch):
    pages = [kvant.Page("kvant_2018_10", 2018, 16, "14", "native", "тело", tmp_path / "a.md")]
    monkeypatch.setattr(kvant, "pages", lambda _c: pages)
    monkeypatch.setattr(kvant, "scan", lambda _i, _s: None)
    said: list[str] = []
    chosen, _ = rotated.draw(
        tmp_path, tmp_path, read=lambda _p: [], text=lambda _p: [], avoid=set, say=said.append
    )
    assert chosen == []
    assert said == []


def run_of(text: str) -> rotated.Run:
    return rotated.Run("kvant_2018_10", 18, text, 8.9, 56.0)


def test_a_reading_that_has_the_run_verbatim_caught_it():
    assert rotated.caught(f"Какой то текст.\n\n{CREDIT} {NAME}\n", run_of(NAME))


def test_the_space_the_page_does_not_have_does_not_lose_the_reader_the_run():
    assert rotated.caught("Иллюстрации Д. Гришуковой", run_of(NAME))


def test_one_letter_wrong_is_not_caught_because_that_name_cannot_be_looked_up():
    assert not rotated.caught("Иллюстрации Д. Гришуновой", run_of(NAME))


def test_a_reading_that_dropped_the_run_entirely_did_not_catch_it():
    assert not rotated.caught("Обычный текст страницы без подписи.", run_of(NAME))


def test_case_does_not_decide_it():
    assert rotated.caught("иллюстрации д.гришуковой", run_of(NAME))


def test_a_run_with_no_letters_in_it_is_never_caught():
    assert not rotated.caught("что угодно", run_of("...."))


def test_the_card_counts_runs_and_not_pages():
    card = rotated.Card(model="reader-a")
    card.pages.append(rotated.judge("a/0001", f"{CREDIT} {NAME}", [run_of(CREDIT), run_of(NAME)]))
    card.pages.append(rotated.judge("a/0002", CREDIT, [run_of(CREDIT), run_of(NAME)]))
    assert card.total() == 4
    assert card.caught() == 3
    assert card.recall() == pytest.approx(0.75)
    assert card.clean() == 1


def test_a_page_with_no_reading_is_charged_with_every_run_on_it(tmp_path):
    where = {"kvant_2018_10/0016": [run_of(CREDIT), run_of(NAME)]}
    card = rotated.score(tmp_path, where, model="reader-a")
    assert card.total() == 2
    assert card.caught() == 0
    assert card.recall() == 0.0
    assert [page.id for page in card.failures] == ["kvant_2018_10/0016"]


def test_a_reading_found_in_the_issue_subdirectory_is_scored(tmp_path):
    (tmp_path / "kvant_2018_10").mkdir()
    (tmp_path / "kvant_2018_10" / "0016.md").write_text(f"{CREDIT}\n{NAME}\n", encoding="utf-8")
    where = {"kvant_2018_10/0016": [run_of(CREDIT), run_of(NAME)]}
    card = rotated.score(tmp_path, where, model="reader-a")
    assert card.recall() == 1.0
    assert card.failures == []


def test_an_empty_card_is_zero_recall_and_not_a_division_by_zero():
    assert rotated.Card(model="reader-a").recall() == 0.0


def test_the_worst_pages_come_first_and_ties_go_by_page_id():
    card = rotated.Card(model="reader-a")
    card.pages.append(rotated.PageCard("a/0002", (), (NAME,)))
    card.pages.append(rotated.PageCard("a/0001", (), (CREDIT, NAME)))
    card.pages.append(rotated.PageCard("a/0003", (), (NAME,)))
    assert [page.id for page in card.worst()] == ["a/0001", "a/0002", "a/0003"]


def test_the_markdown_names_every_run_the_reader_lost():
    card = rotated.Card(model="reader-a")
    card.pages.append(rotated.judge("kvant_2018_10/0016", CREDIT, [run_of(CREDIT), run_of(NAME)]))
    body = card.to_markdown()
    assert "reader-a on kvant-rotated" in body
    assert "50.0%" in body
    assert f"kvant_2018_10/0016: lost {NAME}" in body


def test_the_markdown_says_so_plainly_when_every_run_came_back():
    card = rotated.Card(model="reader-a")
    card.pages.append(rotated.judge("kvant_2018_10/0016", f"{CREDIT} {NAME}", [run_of(CREDIT)]))
    assert "None." in card.to_markdown()


def test_the_written_card_carries_the_recall_and_the_set_name(tmp_path):
    card = rotated.Card(model="reader-a")
    card.pages.append(rotated.judge("kvant_2018_10/0016", CREDIT, [run_of(CREDIT), run_of(NAME)]))
    out = tmp_path / "card.json"
    rotated.write(card, json_path=out, markdown_path=tmp_path / "card.md")
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["set"] == "kvant-rotated"
    assert body["runs"] == 2
    assert body["caught"] == 1
    assert body["recall"] == pytest.approx(0.5)


def test_the_manifest_round_trips_through_the_reference_file(tmp_path, monkeypatch):
    where = {"kvant_2018_10/0016": [run_of(CREDIT), run_of(NAME)]}
    out = tmp_path / "kvant-rotated.json"
    monkeypatch.setattr(rotated, "REFERENCE", out)
    monkeypatch.setattr(kvant, "MANIFESTS", tmp_path)
    rotated.write_manifests(["kvant_2018_10/0016"], where)
    back = rotated.reference(out)
    assert [one.text for one in back["kvant_2018_10/0016"]] == [CREDIT, NAME]
    assert back["kvant_2018_10/0016"][0].pdf_page == 18
    assert back["kvant_2018_10/0016"][0].issue == "kvant_2018_10"


def test_scoring_without_a_reference_file_says_how_to_draw_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="kvant rotated --draw"):
        rotated.reference(tmp_path / "nothing.json")


def test_the_manifest_header_records_the_three_thresholds_it_was_drawn_with():
    head = rotated.manifest_lines(["a/0001"], [run_of(CREDIT)])
    assert "2 times taller than it is wide" in head
    assert "no more than 18 points across" in head
    assert "at least 5 characters long" in head
    assert head.rstrip().endswith("a/0001")


def test_the_shipped_set_is_the_one_the_docstring_describes():
    """The manifest is committed, so this runs anywhere and needs no PDF cache."""
    where = rotated.reference()
    runs = [one for page in where.values() for one in page]
    assert len(where) == 63
    assert len(runs) == 126
    assert all(len(page) == 2 for page in where.values())
    assert all(one.ratio >= rotated.RATIO for one in runs)
    assert all(one.width <= rotated.WIDEST for one in runs)
    assert all(len(one.text) >= rotated.SHORTEST for one in runs)
    assert sorted(where) == kvant.read_manifest("kvant-rotated")


def test_the_shipped_set_does_not_touch_the_held_out_one():
    assert not set(rotated.reference()) & set(kvant.read_manifest("kvant-test"))


def reader(root: Path, name: str, pages: dict[str, str]) -> Path:
    """A directory of readings in the shape `ocr-batch` writes."""
    where = root / name
    for page_id, text in pages.items():
        issue, _, number = page_id.partition("/")
        (where / issue).mkdir(parents=True, exist_ok=True)
        (where / issue / f"{number}.md").write_text(text, encoding="utf-8")
    return where


def test_the_shared_set_is_the_pages_every_reader_produced(tmp_path):
    where = {
        "kvant_2018_10/0016": [run_of(NAME)],
        "kvant_2018_10/0017": [run_of(NAME)],
        "kvant_2018_10/0018": [run_of(NAME)],
    }
    a = reader(tmp_path, "a", dict.fromkeys(where, "текст"))
    b = reader(tmp_path, "b", {"kvant_2018_10/0016": "текст", "kvant_2018_10/0018": "текст"})
    assert sorted(rotated.shared(where, [a, b])) == [
        "kvant_2018_10/0016",
        "kvant_2018_10/0018",
    ]


def test_one_reader_shares_the_pages_it_read_and_nothing_else(tmp_path):
    where = {"kvant_2018_10/0016": [run_of(NAME)], "kvant_2018_10/0017": [run_of(NAME)]}
    a = reader(tmp_path, "a", {"kvant_2018_10/0017": "текст"})
    assert list(rotated.shared(where, [a])) == ["kvant_2018_10/0017"]


def test_the_shared_cut_is_what_turns_a_reader_that_answers_rarely_around(tmp_path):
    # The trap this exists for. b reads one page of two and reads it perfectly;
    # a reads both and gets one wrong. Over the whole set a leads, over the
    # shared page b does, and a report needs both numbers.
    where = {"kvant_2018_10/0016": [run_of(NAME)], "kvant_2018_10/0017": [run_of(NAME)]}
    a = reader(
        tmp_path,
        "a",
        {"kvant_2018_10/0016": f"Иллюстрации {NAME}", "kvant_2018_10/0017": "нет подписи"},
    )
    b = reader(tmp_path, "b", {"kvant_2018_10/0017": f"Иллюстрации {NAME}"})
    whole = [rotated.score(d, where, model=n) for n, d in (("a", a), ("b", b))]
    assert [card.recall() for card in whole] == [0.5, 0.5]
    assert [len(card.failures) for card in whole] == [0, 1]

    cut = rotated.shared(where, [a, b])
    only = [rotated.score(d, cut, model=n) for n, d in (("a", a), ("b", b))]
    assert [card.recall() for card in only] == [0.0, 1.0]


def test_the_table_ranks_by_recall_and_not_by_the_order_it_was_given(tmp_path):
    where = {"kvant_2018_10/0016": [run_of(NAME)]}
    poor = reader(tmp_path, "poor", {"kvant_2018_10/0016": "нет подписи"})
    good = reader(tmp_path, "good", {"kvant_2018_10/0016": f"Иллюстрации {NAME}"})
    text = rotated.table(
        [rotated.score(poor, where, model="poor"), rotated.score(good, where, model="good")]
    )
    lines = text.strip().split("\n")
    assert lines[2].startswith("| good |")
    assert lines[3].startswith("| poor |")


def test_the_table_says_how_many_pages_each_reader_never_answered(tmp_path):
    where = {"kvant_2018_10/0016": [run_of(NAME)], "kvant_2018_10/0017": [run_of(NAME)]}
    half = reader(tmp_path, "half", {"kvant_2018_10/0016": f"Иллюстрации {NAME}"})
    assert "| 1 |" in rotated.table([rotated.score(half, where, model="half")])
