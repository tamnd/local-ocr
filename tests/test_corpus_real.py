"""The properties that can only be checked against the real corpus.

Skipped where `BOURBAKI_CORPUS` is not set, which includes CI, so nothing here
can be relied on to gate a pull request. They are here because two of them found
real defects the moment they were first run, and a check that only runs on the
one machine that has the pages is still worth more than the same check written
down in a comment.
"""

from __future__ import annotations

import pytest

from local_ocr import corpus as corpuslib
from local_ocr import evaluate, golden, pool
from local_ocr.metrics import conformance

pytestmark = pytest.mark.skipif(
    not corpuslib.available(), reason="needs a checkout of tamnd/bourbaki"
)


@pytest.fixture(scope="module")
def dev() -> list[corpuslib.Page]:
    return golden.load("golden-dev", purpose=golden.Purpose.DEVELOPMENT)


def test_the_set_is_the_size_it_says(dev):
    assert len(dev) == 200


def test_the_reference_obeys_every_house_rule_it_is_judged_by(dev):
    """The invariant the whole conformance metric rests on.

    A faithful reading of a page is the reference with its running head put back
    on the front. If that breaks a rule, the rule is wrong, and every reader
    including a perfect one loses the same points for it.
    """
    broken: dict[str, list[str]] = {}
    house = conformance.Conformance()
    for page in dev:
        faithful = evaluate.conformance_reference(page)
        for name in house.observe(faithful, faithful):
            broken.setdefault(name, []).append(page.id)
    assert broken == {}


def test_a_faithful_reading_scores_zero(dev):
    """End to end on 200 real pages, including the head the reference has not got."""
    house = conformance.Conformance()
    for page in dev:
        result = evaluate.judge(page, evaluate.conformance_reference(page), house)
        assert result.prose.edits == 0, page.id
        assert result.whole.edits == 0, page.id
        assert result.formulas.below_threshold == 0, page.id
        assert result.broke == [], page.id
        # The calibration that matters most. If the eight acceptance rules
        # reject a faithful reading of a tier B page, then first read acceptance
        # is measuring the harness rather than the model, and the throughput
        # number that comes out of it is meaningless. This was 84 per cent the
        # first time it was run, and every one of the 32 was a page from a
        # volume that prints no page label being asked for one.
        assert result.accepted, f"{page.id}: {result.problems}"


def test_the_running_head_is_stripped_before_the_text_is_compared(dev):
    """Because `extract` files it in the front matter and a model reads it off the page.

    Without this every reading pays for the head on every page, which on a page
    of this size is most of a per cent of character error rate before it has
    made a single mistake.
    """
    with_heads = [page for page in dev if page.has_head]
    assert len(with_heads) > 100, "the sample should be mostly pages that print a head"
    for page in with_heads:
        faithful = evaluate.conformance_reference(page)
        assert evaluate.without_head(page, faithful).strip() == page.body.strip(), page.id


def test_a_first_line_that_is_not_the_head_is_left_alone(dev):
    """The stripper has to be conservative or it eats a line of the text."""
    page = next(page for page in dev if page.has_head)
    reading = "The theorem below is proved in Chapter IV.\n\n" + page.body
    assert evaluate.without_head(page, reading) == reading


def test_the_pages_are_the_tier_they_claim_to_be(dev):
    assert {page.book for page in dev} <= set(golden.TIER_B_VOLUMES)
    # Not `{page.method} == {"native"}` any more, which is what this asserted
    # until the day it caught something. A set is a list of ids frozen once and
    # the pages under those ids go on being read: four of these had their text
    # layer found to have dropped a glyph and were re-read by the fleet, so
    # their reference is a model's reading now. The invariant that survives is
    # that no such page is scored, not that no such page exists, because the
    # corpus is allowed to improve a page and the set is not allowed to pretend
    # it did not.
    assert {page.method for page in dev if not page.manual} <= {"native", "ocr"}
    for page in golden.stale("golden-dev", dev):
        assert page.method != "native", page.id


@pytest.mark.parametrize("name", sorted(golden.SETS))
def test_a_reference_that_is_no_longer_ground_truth_is_not_scored_against(name):
    """The set may drift. A number measured against the drift may not.

    `golden-incumbent` has no ground truth by construction and every page in it
    is tier C, so it has nothing to drop and this passes vacuously there. That
    is the right shape: the check is over what a set promised, and that set
    promised nothing.
    """
    pages = golden.load(name, purpose=golden.Purpose.MILESTONE)
    stale = golden.stale(name, pages)
    scored = [page for page in pages if page not in stale]
    assert all(golden.in_tier(page, golden.SETS[name].tier) or page.manual for page in scored)


def test_the_head_is_ordered_by_the_page_number(dev):
    """The rule measured in `evaluate.head_line`, checked against the front matter.

    Not the measurement itself, which needed readings of the printed pages to
    make, but the property that measurement chose: on a page that prints both
    halves the label leads on an even number and follows on an odd one. Pinned
    here because the completion in the training pool is this line, so a page
    that comes out the wrong way round is 2 400 examples teaching a model to
    put the page reference on the wrong side of the head.
    """
    both = [page for page in dev if page.running_head and page.page_label]
    assert len(both) > 100, "the sample should be mostly pages that print both halves"
    for page in both:
        line = evaluate.head_line(page)
        number = evaluate.printed_number(page.page_label)
        if number is None:
            continue
        if number % 2 == 0:
            assert line.startswith(page.page_label), page.id
        else:
            assert line.endswith(page.page_label), page.id


def test_no_held_out_page_is_in_the_training_pool(dev):
    """§08's rule, on the real manifests and the real corpus.

    `build` asserts this itself on the way out, so this is the same check run
    against the pages rather than against a fixture, and it is the one that
    would catch a manifest redrawn under a pool that was built before it.
    """
    built = pool.build(require_image=False)
    got = {one.id for one in built.examples}
    for name in pool.FORBIDDEN:
        assert got.isdisjoint(golden.read_manifest(name)), name
    assert len(got) > 2000, "tier B is 2 800 pages and the pool should be most of it"
