"""The Russian bake off.

The one thing worth pinning hardest is at the top: a reading that got a two
column page right must not be punished for the reference's own column order.
That is not a nicety, it is the reason this harness exists instead of
`local-ocr eval`, and a regression there would quietly rank the readers upside
down while every number still looked plausible.

The rest is about what a rate is allowed to hide. A reader that emits one
perfect paragraph and drops the rest of the page has a matched rate of zero, so
the headline is the content rate, and the tests below are mostly the arithmetic
that keeps those two apart.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_ocr import bakeoff, kvant

LEFT = [
    "Рассмотрим последовательность точек на плоскости и докажем её сходимость.",
    "Каждая окружность пересекает прямую не более чем в двух различных точках.",
    "Медиана треугольника делит его на две части одинаковой площади всегда.",
]
RIGHT = [
    "Обозначим через радиус вписанной окружности величину, найденную выше.",
    "Многочлен с целыми коэффициентами имеет корень, делящий свободный член.",
    "Трапеция вписана в окружность тогда и только тогда, когда она равнобокая.",
]

READING = "\n\n".join(LEFT + RIGHT)
"""How a person reads the page, and how the vision path returns it."""

INTERLEAVED = "\n\n".join(one for pair in zip(LEFT, RIGHT, strict=True) for one in pair)
"""How the publisher's text layer hands it over, which is the reference here."""


def page(**over) -> kvant.Page:
    fields = {
        "issue": "kvant_2018_10",
        "year": 2018,
        "page_index": 16,
        "page_label": "14",
        "extraction": "native",
        "body": INTERLEAVED,
        "path": Path("/nowhere/0016.md"),
    }
    fields.update(over)
    return kvant.Page(**fields)


def card(*pages: bakeoff.PageCard, model: str = "reader-a") -> bakeoff.Card:
    return bakeoff.Card(set_name="kvant-dev", model=model, pages=list(pages))


# ---------------------------------------------------------------------------
# The reference is wrong about order, and the reading must not pay for it


def test_a_correct_reading_of_a_page_the_reference_scrambled_scores_no_errors():
    """§07 measured the scramble and this is the whole reason for the module.

    A straight edit distance against this reference charges the correct reading
    about the length of a column, so the better a reader is at two column pages
    the worse its number looks. Matching the blocks first takes that to zero.
    """
    got = bakeoff.judge(page(), READING, set())
    assert got.matched.edits == 0
    assert got.content.edits == 0
    assert got.order.matched == 6


def test_the_order_is_still_reported_and_is_not_folded_into_the_rate():
    """Reported, because a reader that scrambles columns itself has to be visible.

    It just is not charged as character errors, since against this reference
    there is no way to tell a reader that scrambled the page from one that read
    it the way the reference happens to be laid out.
    """
    got = bakeoff.judge(page(), READING, set())
    assert got.order.tau < 1.0
    assert got.order.inversions > 0


# ---------------------------------------------------------------------------
# What the reader dropped, against what it misread


def test_a_dropped_block_is_free_in_the_matched_rate_and_charged_in_the_content_rate():
    """The hole the matched rate leaves open, and what closes it.

    Half a page read perfectly is a matched rate of zero, which is a true
    sentence about the blocks that were paired and a useless one about the
    reader. Content charges every character of every block nothing matched.
    """
    got = bakeoff.judge(page(), "\n\n".join(LEFT), set())
    assert got.matched.edits == 0
    assert got.content.edits > 0
    assert 0.4 < got.content.rate < 0.6


def test_a_misread_block_is_charged_in_both():
    wrong = [LEFT[0].replace("плоскости", "плоскастi"), *LEFT[1:], *RIGHT]
    got = bakeoff.judge(page(), "\n\n".join(wrong), set())
    assert got.matched.edits > 0
    assert got.content.edits == got.matched.edits


def test_the_denominator_of_the_content_rate_is_the_whole_reference():
    whole = len("".join(LEFT + RIGHT).replace(" ", ""))
    got = bakeoff.judge(page(), "\n\n".join(LEFT), set())
    assert got.content.length > whole


def test_a_block_the_reference_never_had_is_not_charged_as_an_error():
    """A caption the text layer omitted is common and is not the reader's fault."""
    extra = READING + "\n\nПодпись под рисунком, которой нет в текстовом слое."
    got = bakeoff.judge(page(), extra, set())
    assert got.content.edits == 0


# ---------------------------------------------------------------------------
# The two Russian gates ride along


def test_the_word_gate_counts_what_the_page_repeats():
    words = {w.lower().strip(".,") for block in LEFT + RIGHT for w in block.split()}
    read = READING + "\n\nкривоая кривоая"
    assert bakeoff.judge(page(), read, words).oov == 1


def test_the_homoglyph_gate_counts_cyrillic_standing_in_for_a_variable():
    got = bakeoff.judge(page(), READING + "\n\nЗдесь $с^{2}$ участвует.", set())
    assert got.homoglyphs == 1


# ---------------------------------------------------------------------------
# A page with no reading at all


def test_a_page_with_no_reading_is_charged_in_full_rather_than_skipped():
    """§05 is blunt about why.

    A reader that refuses the pages it would have done badly on is the failure
    mode a benchmark that drops its hard cases rewards, so nothing is dropped.
    """
    got = bakeoff.missing(page())
    assert got.failed
    assert got.content.rate == 1.0
    assert got.content.length > 0


def test_a_page_with_no_reading_says_so_in_words():
    assert "no reading" in bakeoff.missing(page()).failure


def test_a_page_with_no_reading_contributes_no_order_score():
    """Zero and not a tau, because there is nothing to be in the wrong order."""
    got = bakeoff.missing(page())
    assert got.order.matched == 0
    assert got.order.expected == 0


# ---------------------------------------------------------------------------
# Where a reading is found on disk


class TestFindReading:
    def test_a_directory_per_issue(self, tmp_path: Path):
        (tmp_path / "kvant_2018_10").mkdir()
        want = tmp_path / "kvant_2018_10" / "0016.md"
        want.write_text("x", encoding="utf-8")
        assert bakeoff.find_reading(tmp_path, "kvant_2018_10/0016") == want

    def test_the_flat_name_ocr_batch_writes(self, tmp_path: Path):
        """`kvant pages` flattens the id to name the image and the reading follows."""
        want = tmp_path / "kvant_2018_10-0016.md"
        want.write_text("x", encoding="utf-8")
        assert bakeoff.find_reading(tmp_path, "kvant_2018_10/0016") == want

    def test_nothing_there_is_none_and_not_an_empty_file(self, tmp_path: Path):
        assert bakeoff.find_reading(tmp_path, "kvant_2018_10/0016") is None


# ---------------------------------------------------------------------------
# The card, and the arithmetic across pages


def test_a_run_charges_the_pages_it_found_nothing_for():
    got = bakeoff.run([page()], Path("/nowhere at all"), set(), model="r", set_name="kvant-dev")
    assert len(got.failures) == 1
    assert got.rate("content") == 1.0


def test_a_run_reads_what_is_there(tmp_path: Path):
    (tmp_path / "kvant_2018_10-0016.md").write_text(READING, encoding="utf-8")
    got = bakeoff.run([page()], tmp_path, set(), model="r", set_name="kvant-dev")
    assert got.failures == []
    assert got.rate("content") == 0.0


def test_the_rate_is_micro_averaged_so_a_short_page_cannot_outweigh_a_dense_one():
    """The mean of the per page rates would make a two line contents page
    count as much as a page of problems, and the contents pages are the ones a
    reader is most likely to get exactly right or entirely wrong.
    """
    from local_ocr.metrics import cer

    short = bakeoff.PageCard(
        id="a",
        matched=cer.Score(5, 5),
        content=cer.Score(5, 5),
        prose=cer.Score(0, 0),
        order=bakeoff.orderlib.Order(tau=1.0, inversions=0, matched=1, read=1, expected=1),
        oov=0,
        homoglyphs=0,
    )
    dense = bakeoff.PageCard(
        id="b",
        matched=cer.Score(0, 995),
        content=cer.Score(0, 995),
        prose=cer.Score(0, 0),
        order=bakeoff.orderlib.Order(tau=1.0, inversions=0, matched=1, read=1, expected=1),
        oov=0,
        homoglyphs=0,
    )
    assert card(short, dense).rate("content") == 0.005


def test_the_inversion_count_is_the_worst_page_and_not_the_mean():
    """A perfectly woven page still scores a tau above +0.5, so a run where a
    tenth of the pages are woven has a mean tau that looks fine. The worst
    page's inverted pair count is what says whether the reader can do columns.
    """
    clean = bakeoff.judge(page(), INTERLEAVED, set())
    woven = bakeoff.judge(page(), READING, set())
    assert card(clean, woven).inversions() == woven.order.inversions


def test_coverage_is_blocks_found_over_blocks_wanted_across_the_set():
    half = bakeoff.judge(page(), "\n\n".join(LEFT), set())
    whole = bakeoff.judge(page(), READING, set())
    assert card(half, whole).coverage() == 9 / 12


def test_the_flag_rates_count_pages_and_not_flags():
    """A page carrying several lookalikes is one page for somebody to look at."""
    many = bakeoff.judge(page(), READING + "\n\nЗдесь $с^{2} + р^{2} + х^{2}$ равно нулю.", set())
    none = bakeoff.judge(page(), READING, set())
    assert many.homoglyphs > 1
    assert card(many, none).flagged("homoglyphs") == 0.5


def test_the_worst_pages_come_back_worst_first_by_content():
    good = bakeoff.judge(page(), READING, set())
    bad = bakeoff.missing(page(page_index=17))
    assert [p.id for p in card(good, bad).worst(1)] == [bad.id]


def test_an_empty_card_answers_zero_rather_than_dividing_by_zero():
    empty = card()
    assert empty.rate("content") == 0.0
    assert empty.tau() == 0.0
    assert empty.coverage() == 0.0
    assert empty.flagged("oov") == 0.0
    assert empty.inversions() == 0


# ---------------------------------------------------------------------------
# The bake off itself


def test_the_table_ranks_by_content_and_does_not_leave_it_to_the_reader():
    """M8 item 5 ends in a default chosen by a number.

    A table that merely lists the readers invites the default to stay where it
    is, because moving it would need somebody to make the argument.
    """
    good = card(bakeoff.judge(page(), READING, set()), model="reader-c")
    poor = card(bakeoff.missing(page()), model="reader-a")
    lines = bakeoff.table([poor, good]).splitlines()
    assert lines[2].startswith("| reader-c ")
    assert lines[3].startswith("| reader-a ")


def test_every_reader_gets_a_row():
    ones = [card(bakeoff.judge(page(), READING, set()), model=f"reader-{n}") for n in "abc"]
    assert len(bakeoff.table(ones).splitlines()) == 5


# ---------------------------------------------------------------------------
# What gets written down


def test_the_json_carries_the_numbers_a_later_run_would_be_compared_against(tmp_path: Path):
    one = card(bakeoff.judge(page(), READING, set()))
    bakeoff.write(one, json_path=tmp_path / "a" / "card.json", markdown_path=None)
    got = json.loads((tmp_path / "a" / "card.json").read_text(encoding="utf-8"))
    assert got["set"] == "kvant-dev"
    assert got["model"] == "reader-a"
    assert got["pages"] == 1
    assert got["content_cer"] == 0.0
    assert got["failed"] == 0


def test_the_json_keeps_the_russian_readable(tmp_path: Path):
    """`ensure_ascii` would turn every page id and every flag into escapes, and
    these files are meant to be read by a person before they are read by
    anything else.
    """
    one = card(bakeoff.missing(page()))
    bakeoff.write(one, json_path=tmp_path / "card.json", markdown_path=None)
    assert "\\u" not in (tmp_path / "card.json").read_text(encoding="utf-8")


def test_the_markdown_says_what_the_reference_is_before_it_says_a_number(tmp_path: Path):
    """Anybody reading a tau off this file has to be told the reference's own
    column order is wrong, or they will read a tau below +1 as a mark against
    the reader.
    """
    one = card(bakeoff.judge(page(), READING, set()))
    bakeoff.write(one, json_path=None, markdown_path=tmp_path / "card.md")
    text = (tmp_path / "card.md").read_text(encoding="utf-8")
    assert "publisher's text layer" in text
    assert "Content CER" in text


def test_the_markdown_lists_the_pages_with_no_reading_by_name(tmp_path: Path):
    one = card(bakeoff.missing(page()))
    bakeoff.write(one, json_path=None, markdown_path=tmp_path / "card.md")
    assert "kvant_2018_10/0016" in (tmp_path / "card.md").read_text(encoding="utf-8")


def test_writing_neither_writes_nothing(tmp_path: Path):
    bakeoff.write(card(), json_path=None, markdown_path=None)
    assert list(tmp_path.iterdir()) == []
