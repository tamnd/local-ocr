"""The manifests, and the guard on the one that must not be looked at.

None of this needs the corpus. That is deliberate: the properties worth pinning
are properties of the recorded sets and of the door into them, and those hold on
a machine that has never seen a page.
"""

from __future__ import annotations

import pytest

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
