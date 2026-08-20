"""The adjudicator, whose job is to shrink the residue and not to hide it."""

from __future__ import annotations

from local_ocr import corpus as corpuslib
from local_ocr import disagree

HEAD = "A VIII.13 § 2.5\n\n"
BODY = (
    "**Proposition 4.** — Let $A$ be a ring whose radical is $\\mathfrak{r}$. Then the "
    "canonical map is surjective and its kernel is the radical, which is what was to be "
    "proved, and this sentence runs on so that the page is not a short one under rule 1.\n"
)


def page(body: str, book: str = "alg-viii", number: int = 42) -> corpuslib.Page:
    return corpuslib.Page(
        book=book,
        pdf_page=number,
        method="native",
        manual=False,
        body=body,
        path=None,  # type: ignore[arg-type]
    )


def test_two_identical_readings_disagree_about_nothing():
    assert disagree.classify(page(HEAD + BODY), HEAD + BODY) == []


def test_a_page_the_corpus_already_flagged_goes_to_extraction():
    """The corpus said so before this model existed, so the model does not get the blame."""
    reference = page(HEAD + BODY, book="alg-viii-fr", number=448)
    items = disagree.classify(
        reference, HEAD + BODY.replace("surjective", "injective"), drift=["alg-viii-fr/0448"]
    )
    assert items
    assert all(item.kind is disagree.Kind.EXTRACTION for item in items)


def test_a_difference_that_normalisation_removes_is_a_variation():
    reference = page(HEAD + BODY)
    items = disagree.classify(reference, HEAD + BODY.replace("radical", "radical"))
    assert all(item.kind is disagree.Kind.VARIATION for item in items)


def test_a_real_difference_is_left_for_a_person():
    """Never automatically the model's fault. That verdict has a cost attached."""
    items = disagree.classify(page(HEAD + BODY), HEAD + BODY.replace("surjective", "injective"))
    assert items
    assert items[0].kind is disagree.Kind.UNDECIDED


def test_a_reading_the_rules_reject_while_the_reference_passes_is_the_model():
    items = disagree.classify(page(HEAD + BODY), "I'm sorry, I can't help with that.")
    assert items
    assert any(item.kind is disagree.Kind.MODEL for item in items)


def test_the_work_list_names_every_pile():
    work = disagree.Work()
    work.items.extend(disagree.classify(page(HEAD + BODY), HEAD + BODY.replace("kernel", "image")))
    counts = work.counts()
    assert set(counts) == {"variation", "extraction", "model", "undecided"}
    assert sum(counts.values()) == len(work.items)
    assert "Disagreements" in work.to_markdown()
