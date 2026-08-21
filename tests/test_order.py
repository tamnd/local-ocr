"""Reading order.

The case this metric exists for is a two column page whose columns got
interleaved, so most of these build that page rather than shuffling blocks at
random. A metric that only handles a random shuffle would score the interleave
somewhere in the middle and say nothing useful about it.

The other half of the tests are about what the number is allowed to hide. A
reading that dropped half the page and put the rest in perfect order has a tau
of 1, and that is only not a lie because `coverage` is reported next to it.
"""

from __future__ import annotations

from local_ocr.metrics.order import MATCH, Order, blocks, order, paired

# Six paragraphs with no shared vocabulary beyond the function words, so that a
# block matches its own counterpart and nothing else.
LEFT = [
    "The tortoise sets out along the road at a steady pace and does not hurry.",
    "Achilles waits at the start and gives the tortoise a generous head start.",
    "Each stride of Achilles covers the ground the tortoise has already left.",
]
RIGHT = [
    "Zeno objects that the pursuer must first arrive where the pursued began.",
    "The objection dissolves once an infinite series is allowed a finite sum.",
    "Aristotle answered differently, by denying that time divides that finely.",
]

READING = "\n\n".join(LEFT + RIGHT)
"""How a person reads it: the whole left column, then the whole right column."""

INTERLEAVED = "\n\n".join(para for pair in zip(LEFT, RIGHT, strict=True) for para in pair)
"""How a PDF text layer often hands it over: line by line across the page.

Not a random shuffle. The blocks come out alternating, which is what a text
layer that runs in drawing order does to a two column page, and it is the
failure the born digital Kvant path actually has.
"""


# ---------------------------------------------------------------------------
# Cutting the page into blocks


def test_a_blank_line_is_what_separates_two_blocks():
    assert len(blocks(READING)) == 6


def test_a_display_formula_is_a_block_like_any_other():
    """An equation that lands in the wrong column is the failure being looked
    for, so its position has to be measured and not skipped.
    """
    body = "First the claim is stated.\n\n$$\na^2 + b^2 = c^2\n$$\n\nThen it is proved.\n"
    assert len(blocks(body)) == 3


def test_trailing_blank_lines_are_not_blocks():
    assert len(blocks("One paragraph here.\n\n\n\n")) == 1


def test_the_yaml_header_is_not_a_block():
    """It is metadata, it is written by the tool rather than read by the model,
    and it always matches its counterpart, so counting it would hand every page
    one free block and make both numbers flatter than the page deserves.
    """
    header = '---\nissue: kvant_1986_5\nyear: 1986\npage_label: "57"\nlang: ru\n---\n\n'
    assert blocks(header + READING) == blocks(READING)


def test_a_header_like_line_in_the_middle_is_left_alone():
    """Only the header at the very top goes. A rule further down the page is
    part of the page.
    """
    assert len(blocks("First block.\n\n---\n\nSecond block.")) == 3


# ---------------------------------------------------------------------------
# The order itself


def test_a_reading_that_agrees_with_the_reference():
    got = order(READING, READING)
    assert got.tau == 1.0
    assert got.inversions == 0
    assert got.matched == 6


def test_the_two_column_interleave_is_what_this_is_for():
    """Six blocks, alternating, and only three of the fifteen pairs swap.

    Worth writing down, because the number is smaller than it feels. A whole
    page assembled in the wrong order scores +0.600, which is nowhere near the
    -1 that reading the page backwards gives, and a run judged on tau alone
    with a threshold set by intuition would let the interleave straight through.
    Alternating blocks keep most pairs in the right relative order; what they
    destroy is the local order, three adjacent pairs of it.
    """
    got = order(INTERLEAVED, READING)
    assert got.matched == 6
    assert got.inversions == 3
    assert got.tau == (15 - 6) / 15 == 0.6


def test_a_page_read_backwards_is_minus_one():
    backwards = "\n\n".join(reversed(LEFT + RIGHT))
    got = order(backwards, READING)
    assert got.tau == -1.0
    assert got.inversions == 15


def test_one_block_out_of_place_is_one_inversion():
    swapped = LEFT[:1] + LEFT[2:3] + LEFT[1:2] + RIGHT
    got = order("\n\n".join(swapped), READING)
    assert got.inversions == 1
    assert got.tau == (15 - 2) / 15


def test_inversions_are_reported_because_tau_hides_the_size_of_the_page():
    """Two pages with the same tau and five times the work between them.

    A three block page with one swap and a six block page whose last block was
    read first both come out at +0.333. They are not the same problem: one is a
    swap somebody fixes in a second and the other is five pairs out of order.
    Only the raw count says which is which, which is why both are reported.
    """
    small = order("\n\n".join([LEFT[1], LEFT[0], LEFT[2]]), "\n\n".join(LEFT))
    moved = LEFT + RIGHT
    large = order("\n\n".join([moved[5], *moved[:5]]), READING)
    assert small.tau == large.tau
    assert (small.inversions, large.inversions) == (1, 5)


# ---------------------------------------------------------------------------
# What the number is not allowed to hide


def test_a_reading_that_dropped_half_the_page_still_has_a_perfect_tau():
    """Which is exactly why coverage is reported beside it.

    Order is a sentence about the blocks that are there. Folding what is missing
    into the same number would give a page with one block a middling score and
    make it indistinguishable from a page read in a middling order.
    """
    got = order("\n\n".join(LEFT), READING)
    assert got.tau == 1.0
    assert got.matched == 3
    assert got.expected == 6
    assert got.coverage == 0.5


def test_a_reading_with_nothing_in_common_matches_nothing():
    got = order("A wholly unrelated sentence about elephants and rainfall.", READING)
    assert got.matched == 0
    assert got.coverage == 0.0


def test_a_single_block_cannot_be_out_of_order():
    """One block has no pairs, so tau is one by definition and not by luck."""
    got = order(LEFT[0], "\n\n".join(LEFT))
    assert got.tau == 1.0
    assert got.inversions == 0
    assert got.matched == 1


def test_an_empty_reading_is_not_a_division_by_zero():
    got = order("", READING)
    assert got.matched == 0
    assert got.coverage == 0.0
    assert got.tau == 1.0


def test_an_empty_reference_is_not_a_division_by_zero():
    assert order(READING, "").coverage == 0.0


# ---------------------------------------------------------------------------
# Matching blocks to blocks


def test_a_block_read_imperfectly_still_matches_its_counterpart():
    """The two readings differ in markup, hyphenation and the odd misread word,
    and they are still the same paragraph. A matcher that needed them equal
    would report every page as entirely dropped.
    """
    rough = LEFT[0].replace("steady", "steddy").replace("hurry", "hurrry")
    got = order("\n\n".join([rough, *LEFT[1:]]), "\n\n".join(LEFT))
    assert got.matched == 3
    assert got.tau == 1.0


def test_two_different_paragraphs_do_not_match_each_other():
    """They share the function words and little else, and a set counts each of
    those once, so the overlap stays under the threshold.
    """
    got = order(LEFT[0], RIGHT[0])
    assert got.matched == 0


def test_a_reference_block_is_claimed_once():
    """The same paragraph read twice must not match the same reference twice.

    Otherwise a model that repeats itself, which is a real failure mode of a
    reading that loses its place, scores as though it read two blocks.
    """
    doubled = "\n\n".join([LEFT[0], LEFT[0], LEFT[1]])
    got = order(doubled, "\n\n".join(LEFT))
    assert got.matched == 2


def test_the_best_match_wins_rather_than_the_first_one():
    """Left to right matching would let an early block take a partial match that
    a later block matches better, and the later block would then count as
    dropped. One bad guess would become two wrong numbers.
    """
    partial = LEFT[1].split(" and ")[0]
    got = order("\n\n".join([partial, LEFT[1]]), "\n\n".join(LEFT))
    assert got.matched == 1


def test_a_block_with_no_words_matches_itself():
    """A folio marker or a bare rule has an empty word set and cannot be matched
    by vocabulary, so it falls back to the text. Before that fallback, 13 of 400
    real Kvant pages could not match themselves and the metric reported blocks
    the page plainly contained as dropped.
    """
    body = "⟦folio 45⟧\n\n" + LEFT[0] + "\n\n---"
    got = order(body, body)
    assert got.matched == 3
    assert got.coverage == 1.0


def test_two_different_wordless_blocks_do_not_match():
    got = order("---", "⟦column⟧")
    assert got.matched == 0


def test_two_short_markers_match_on_one_shared_word_and_that_is_the_known_limit():
    """The price of scoring over the smaller block rather than over the two
    together. `Рис. 1.` and `Рис. 2.` share one word of the two each of them has,
    which is half, which clears the threshold, so in isolation the matcher calls
    them the same block and it is wrong.

    It is paid because the alternative was worse. Scoring over the two together
    is what made a correct paragraph inside a fused column run unmatchable, and
    that cost 30 points of content CER on the dev set against a handful of
    characters here.

    On a page it mostly does not arise, because a block proposes only its best
    partner and an exact marker beats a marker off by a digit.
    """
    assert order("Рис. 1.", "Рис. 1.").matched == 1
    assert order("Рис. 1.", "Рис. 2.").matched == 1

    page = order("Рис. 1.\n\nРис. 2.", "Рис. 2.\n\nРис. 1.")
    assert page.matched == 2
    assert page.inversions == 1


# ---------------------------------------------------------------------------
# The two sides disagreeing about where a block ends
#
# Both directions happen on one real Kvant page. The publisher's text layer
# shatters a display heading into one block a word and fuses a column of
# paragraphs into one block, so the matching has to put several blocks of one
# side against several of the other.


def test_a_reference_block_split_into_words_still_matches_the_line_that_holds_them():
    """The shattered heading. The text layer writes a title one word a block."""
    shattered = "\n\n".join(["Achilles", "and", "the", "tortoise"])
    got = order("Achilles and the tortoise", shattered)
    assert got.expected == 4
    assert got.matched == 4
    assert got.coverage == 1.0


def test_the_shattered_blocks_come_back_as_one_pair_and_nothing_is_dropped():
    shattered = "\n\n".join(["Achilles", "and", "the", "tortoise"])
    pairs, dropped, extra = paired("Achilles and the tortoise", shattered)
    assert len(pairs) == 1
    assert pairs[0][0] == "Achilles and the tortoise"
    assert pairs[0][1] == "Achilles\n\nand\n\nthe\n\ntortoise"
    assert dropped == []
    assert extra == []


def test_paragraphs_fused_into_one_reference_block_still_match():
    """The other direction. A column run holds five paragraphs in one block.

    Under a rate over the two sides together each paragraph scored at most its
    own length over the run's, so a correct reading matched nothing at all.
    """
    fused = " ".join(LEFT)
    got = order("\n\n".join(LEFT), fused)
    assert got.expected == 1
    assert got.matched == 1

    pairs, dropped, extra = paired("\n\n".join(LEFT), fused)
    assert len(pairs) == 1
    assert pairs[0][0] == "\n\n".join(LEFT)
    assert dropped == []
    assert extra == []


def test_a_fused_run_counts_once_towards_the_tau_and_not_once_a_paragraph():
    """Otherwise the reference's segmentation would weigh on order as well.

    The reading here has the right column first and the left column fused into
    one reference block. That is one pair out of order, not three, because the
    three paragraphs of the run are one passage.
    """
    got = order("\n\n".join(RIGHT + LEFT), " ".join(LEFT) + "\n\n" + "\n\n".join(RIGHT))
    assert got.inversions == 3
    assert got.matched == 4


def test_a_block_the_reading_never_produced_is_still_reported_dropped():
    """The grouping must not quietly absorb what was not read."""
    pairs, dropped, extra = paired("\n\n".join(LEFT), "\n\n".join(LEFT + RIGHT))
    assert len(pairs) == 3
    assert dropped == RIGHT
    assert extra == []


def test_the_threshold_is_where_it_says_it_is():
    assert 0 < MATCH < 1


# ---------------------------------------------------------------------------
# Reporting


def test_the_line_carries_all_three_numbers():
    line = str(order(INTERLEAVED, READING))
    assert "tau +0.600" in line
    assert "3 inversions" in line
    assert "6 of 6 blocks matched" in line


def test_a_perfect_reading_prints_a_signed_tau():
    """Signed, because a negative tau is a different diagnosis from a small one
    and the sign is the fastest way to see it in a column of numbers.
    """
    assert str(order(READING, READING)).startswith("tau +1.000")


def test_the_result_is_hashable_so_a_run_can_be_grouped_by_it():
    assert isinstance(hash(order(READING, READING)), int)
    assert isinstance(order(READING, READING), Order)


# ---------------------------------------------------------------------------
# The pairing underneath, which the Russian bake off scores inside


def test_the_pairs_come_back_in_reference_order_however_they_were_read():
    """The point of exposing this at all.

    A reading whose columns are interleaved is matched block for block against
    the reference and handed back in the reference's order, so a character rate
    computed over the pairs is a rate about the characters and not about the
    order. The order question is answered separately by the tau.
    """
    pairs, dropped, extra = paired(INTERLEAVED, READING)
    assert [want for _, want in pairs] == LEFT + RIGHT
    assert [read for read, _ in pairs] == LEFT + RIGHT
    assert dropped == []
    assert extra == []


def test_a_reference_block_nothing_matched_comes_back_as_dropped():
    pairs, dropped, extra = paired("\n\n".join(LEFT), READING)
    assert len(pairs) == 3
    assert dropped == RIGHT
    assert extra == []


def test_a_block_the_reading_invented_comes_back_as_extra():
    """Separately from the dropped ones, because they cost different things.

    A dropped reference block is text the reader did not produce and the bake
    off charges every character of it. An extra block is text the reference does
    not have, which on a Kvant page is usually a caption the text layer omitted,
    so charging it in full would punish the better reading.
    """
    invented = "\n\n".join([*LEFT, "A caption under the figure that the text layer never had."])
    pairs, dropped, extra = paired(invented, "\n\n".join(LEFT))
    assert len(pairs) == 3
    assert dropped == []
    assert extra == ["A caption under the figure that the text layer never had."]


def test_the_pairs_and_the_tau_agree_about_which_block_is_which():
    """One matcher underneath both, which is why this holds.

    They were two before, and two greedy passes over the same blocks can differ
    on a page where a block matches two others nearly as well, which would leave
    a run reporting a character rate and an order score computed against
    different pairings of the same page.
    """
    pairs, _, _ = paired(INTERLEAVED, READING)
    assert len(pairs) == order(INTERLEAVED, READING).matched


def test_pairing_nothing_against_a_page_drops_the_whole_page():
    pairs, dropped, extra = paired("", READING)
    assert pairs == []
    assert dropped == LEFT + RIGHT
    assert extra == []
