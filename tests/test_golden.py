"""The manifests, and the guard on the one that must not be looked at.

None of this needs the corpus. That is deliberate: the properties worth pinning
are properties of the recorded sets and of the door into them, and those hold on
a machine that has never seen a page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_ocr import corpus as corpuslib
from local_ocr import golden


def test_every_set_has_a_manifest_of_the_size_it_claims():
    for name, entry in golden.SETS.items():
        ids = golden.read_manifest(name)
        assert len(ids) == entry.size, name


def test_dev_and_test_are_disjoint():
    """Not a nicety. An overlap makes the held out set report a tuned number."""
    dev = set(golden.read_manifest("golden-dev"))
    test = set(golden.read_manifest("golden-test"))
    assert dev & test == set()


def test_the_ids_are_page_ids_and_are_unique():
    for name in golden.SETS:
        ids = golden.read_manifest(name)
        assert len(set(ids)) == len(ids), name
        for page_id in ids:
            book, sep, number = page_id.partition("/")
            assert sep and book and number.isdigit() and len(number) == 4, page_id


def test_dev_and_test_are_stratified_the_same_way():
    """Two sets drawn to be comparable, so per volume they have to match.

    If they drift apart, a number from one is not a number about the other and
    the comparison the two exist for stops meaning anything.
    """

    def shape(name: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for page_id in golden.read_manifest(name):
            counts[page_id.split("/")[0]] = counts.get(page_id.split("/")[0], 0) + 1
        return counts

    dev, test = shape("golden-dev"), shape("golden-test")
    assert set(dev) == set(golden.TIER_B_VOLUMES)
    for book in dev:
        assert abs(dev[book] - test[book]) <= 1, book


def test_reading_the_held_out_set_for_development_is_refused():
    with pytest.raises(golden.Burned) as err:
        golden.load("golden-test", purpose=golden.Purpose.DEVELOPMENT)
    assert "held out" in str(err.value)


def test_reading_the_held_out_set_for_training_is_refused():
    """The one §08 makes a hard rule, because fine tuning breaks it by accident."""
    with pytest.raises(golden.Burned):
        golden.load("golden-test", purpose=golden.Purpose.TRAINING)


def test_a_set_that_is_not_held_out_needs_no_permission():
    """It still needs a corpus, so the guard is what has to be passed, not the read."""
    with pytest.raises(Exception) as err:
        golden.load("golden-dev", purpose=golden.Purpose.DEVELOPMENT, corpus="/nowhere")
    assert not isinstance(err.value, golden.Burned)


def test_an_unknown_set_says_what_there_is():
    with pytest.raises(KeyError) as err:
        golden.manifest("golden-easy")
    assert "golden-dev" in str(err.value)


def test_the_held_out_manifest_says_so_in_the_file():
    """Whoever opens the file next should not have to know the rule already."""
    text = golden.manifest("golden-test").read_text(encoding="utf-8")
    assert "HELD OUT" in text


def test_math_density_separates_prose_from_displays():
    assert golden.math_density("no mathematics at all here") == 0.0
    # The delimiters themselves are not inside the span, so a display of nothing
    # but mathematics comes out a little under one rather than at it.
    dense = golden.math_density("$$\\int_0^1 f(x)\\,dx = 1$$")
    assert dense > 0.8


def test_the_draw_key_is_stable():
    """A digest and not a random number generator, so a draw reproduces."""
    assert golden._rank("alg-viii/0042", "dense") == golden._rank("alg-viii/0042", "dense")
    assert golden._rank("alg-viii/0042", "dense") != golden._rank("alg-viii/0042", "plain")


def _page(
    page_id: str = "alg-viii/0118",
    method: str = "native",
    manual: bool = False,
    body: str = "Some text.",
) -> corpuslib.Page:
    book, _, number = page_id.partition("/")
    return corpuslib.Page(
        book=book,
        pdf_page=int(number),
        method=method,
        manual=manual,
        body=body,
        path=Path(f"/corpus/pages/{page_id}.md"),
    )


class TestWhetherAPageIsStillTheTierItWasDrawnAs:
    def test_a_native_tier_b_page_is_in_tier(self):
        assert golden.in_tier(_page(), "B")

    def test_a_tier_b_page_that_was_read_by_the_fleet_has_left_it(self):
        # The case this was written for. The extraction dropped a glyph, the
        # page went to the fleet, the id did not change, and the reference is a
        # model's reading now.
        assert not golden.in_tier(_page(method="ocr"), "B")

    def test_a_native_page_from_a_volume_with_no_text_layer_is_not_tier_b(self):
        assert not golden.in_tier(_page("ac-i-vii/0042"), "B")

    def test_an_empty_page_is_not_a_tier_b_reference(self):
        assert not golden.in_tier(_page(body="   \n"), "B")

    def test_tier_a_is_whether_a_person_read_it(self):
        assert golden.in_tier(_page(method="ocr", manual=True), "A")
        assert not golden.in_tier(_page(manual=False), "A")

    def test_tier_c_is_what_the_fleet_read(self):
        assert golden.in_tier(_page(method="ocr"), "C")
        assert not golden.in_tier(_page(method="native"), "C")

    def test_a_tier_nobody_has_defined_is_not_quietly_failed(self):
        # Returning False for an unknown tier would report every page of a new
        # set as stale, which reads as a corrupt set rather than as a gap here.
        assert golden.in_tier(_page(), "Z")


class TestWhichPagesAreStale:
    def test_a_page_that_left_its_tier_is_stale(self):
        pages = [_page(), _page("alg-viii/0311", method="ocr")]
        assert [p.id for p in golden.stale("golden-dev", pages)] == ["alg-viii/0311"]

    def test_a_page_a_person_read_is_never_stale(self):
        # The tier is about where the reference came from and a person is the
        # best source there is, so a hand read page improves on any tier.
        pages = [_page("alg-viii/0311", method="ocr", manual=True)]
        assert golden.stale("golden-dev", pages) == []

    def test_the_incumbent_set_has_nothing_to_lose(self):
        assert golden.stale("golden-incumbent", [_page(method="ocr")]) == []

    def test_a_set_with_no_drift_reports_none(self):
        assert golden.stale("golden-dev", [_page(), _page("alg-viii/0311")]) == []


class TestDriftReportsIt:
    def _drift(self, left: list[str]) -> golden.Drift:
        return golden.Drift(
            name="golden-test", recorded=200, would_draw=200, gone=[], arrived=[], left=left
        )

    def test_a_set_that_lost_a_tier_is_not_steady(self):
        assert not self._drift(["alg-viii/0353"]).steady

    def test_the_line_names_the_tier_that_was_left(self):
        assert "1 no longer tier B" in self._drift(["alg-viii/0353"]).line()

    def test_a_set_that_kept_its_tier_is_steady(self):
        assert self._drift([]).steady
        assert self._drift([]).line() == "golden-test: 200 pages, unchanged"
