"""The training pool, on a corpus of a handful of pages made up for the purpose.

Made up rather than read out of `tamnd/bourbaki`, so these run in CI where there
is no corpus. The ids are not made up: they are real ids off the real manifests,
because the property worth pinning is that a page named in `golden-test` does
not reach the pool, and a made up id would pin nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_ocr import golden, pool

BODY = (
    "## § 2. THE RADICAL\n"
    "\n"
    "**Proposition 4.** — Let $A$ be a ring whose radical is $\\mathfrak{r}$.\n"
)


def write(corpus: Path, page_id: str, body: str = BODY, method: str = "native") -> Path:
    book, _, number = page_id.partition("/")
    path = corpus / "pages" / book / f"{number}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"book: {book}\n"
        f"pdf_page: {int(number)}\n"
        f"method: {method}\n"
        "manual: false\n"
        "page_label: A VIII.13\n"
        "running_head: THE RADICAL\n"
        "---\n"
        "\n" + body,
        encoding="utf-8",
    )
    image = corpus / "images" / book / f"{int(number):04d}.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"not a png, and nothing here opens it")
    return path


@pytest.fixture
def held() -> dict[str, list[str]]:
    """Real ids off the real manifests, a few of each."""
    return {
        "golden-test": golden.read_manifest("golden-test")[:3],
        "golden-hard": [
            page_id
            for page_id in golden.read_manifest("golden-hard")
            if page_id.split("/")[0] in golden.TIER_B_VOLUMES
        ][:2],
        "golden-dev": golden.read_manifest("golden-dev")[:2],
    }


@pytest.fixture
def corpus(tmp_path: Path, held: dict[str, list[str]]) -> Path:
    root = tmp_path / "bourbaki"
    for name in held:
        for page_id in held[name]:
            write(root, page_id)
    for page_id in ("alg-viii/9001", "alg-viii/9002", "lie-vii-ix/9003"):
        write(root, page_id)
    return root


class TestExclusion:
    def test_no_held_out_page_reaches_the_pool(self, corpus, held):
        """The assertion §08 asks for, and the reason this module exists."""
        built = pool.build(corpus)
        got = {one.id for one in built.examples}
        for name in ("golden-test", "golden-hard"):
            assert got.isdisjoint(held[name]), name

    def test_the_pages_that_were_kept_out_are_counted(self, corpus, held):
        built = pool.build(corpus)
        assert built.excluded["golden-test"] == len(held["golden-test"])
        assert built.excluded["golden-hard"] == len(held["golden-hard"])

    def test_the_check_catches_a_pool_somebody_built_by_hand(self):
        """`build` is not the only way to make a Pool, so the guard is separate."""
        page_id = golden.read_manifest("golden-test")[0]
        by_hand = pool.Pool(examples=[pool.Example(page_id, "alg-viii", "x.png", "text", "train")])
        with pytest.raises(pool.Contaminated) as err:
            pool.check(by_hand)
        assert page_id in str(err.value)

    def test_golden_dev_is_in_the_pool_and_is_reported(self, corpus, held):
        """§08 allows it. What it does not allow is it happening quietly."""
        built = pool.build(corpus)
        got = {one.id for one in built.examples}
        assert set(held["golden-dev"]) <= got
        assert built.dev == len(held["golden-dev"])
        assert f"{built.dev} of the pages are `golden-dev`" in pool.report(built)


class TestWhatGoesIn:
    def test_the_completion_is_the_head_and_the_body(self, corpus):
        built = pool.build(corpus)
        one = next(e for e in built.examples if e.id == "alg-viii/9001")
        assert one.completion.startswith("THE RADICAL A VIII.13\n\n")
        assert "Proposition 4." in one.completion

    def test_the_prompt_is_not_in_the_file(self, corpus):
        """1 400 identical tokens a line, and §08 trains on the completion only."""
        line = json.loads(pool.to_jsonl(pool.build(corpus)).splitlines()[0])
        assert set(line) == {"id", "book", "image", "completion", "split"}

    def test_a_page_with_no_image_is_left_out_and_named(self, corpus):
        (corpus / "images" / "alg-viii" / "9001.png").unlink()
        built = pool.build(corpus)
        assert "alg-viii/9001" in built.no_image
        assert "alg-viii/9001" not in {one.id for one in built.examples}

    def test_a_page_with_no_image_can_be_kept_on_purpose(self, corpus):
        (corpus / "images" / "alg-viii" / "9001.png").unlink()
        built = pool.build(corpus, require_image=False)
        assert "alg-viii/9001" in {one.id for one in built.examples}

    def test_a_blank_page_is_left_out(self, corpus):
        write(corpus, "alg-viii/9004", body="\n")
        built = pool.build(corpus)
        assert "alg-viii/9004" in built.empty

    def test_a_page_the_fleet_read_is_not_ground_truth(self, corpus):
        """Tier B is the native extraction. An OCR page has no verified answer."""
        write(corpus, "alg-viii/9005", method="ocr")
        built = pool.build(corpus)
        assert "alg-viii/9005" not in {one.id for one in built.examples}


class TestSplit:
    def test_the_same_page_lands_on_the_same_side_every_time(self, corpus):
        first = {one.id: one.split for one in pool.build(corpus).examples}
        second = {one.id: one.split for one in pool.build(corpus).examples}
        assert first == second

    def test_the_split_can_be_switched_off(self, corpus):
        built = pool.build(corpus, validation=0.0)
        assert built.val == []
        assert len(built.train) == len(built.examples)

    def test_the_fraction_is_roughly_what_was_asked_for(self):
        """Not a property of the corpus, so it is measured over ten thousand ids."""
        ids = [f"alg-viii/{n:04d}" for n in range(10000)]
        held = sum(1 for page_id in ids if pool._split(page_id, 0.05) == "val")
        assert 0.04 < held / len(ids) < 0.06


class TestCommand:
    def test_it_writes_the_examples_and_the_summary(self, corpus, tmp_path):
        from local_ocr import cli

        jsonl, markdown = tmp_path / "pool.jsonl", tmp_path / "pool.md"
        code = cli.main(
            [
                "pool",
                "--corpus",
                str(corpus),
                "--jsonl",
                str(jsonl),
                "--markdown",
                str(markdown),
            ]
        )
        assert code == 0
        assert len(jsonl.read_text(encoding="utf-8").splitlines()) == 5
        assert "# Training pool" in markdown.read_text(encoding="utf-8")

    def test_a_corpus_that_is_not_there_is_an_argument_error(self, tmp_path, capsys):
        from local_ocr import cli

        assert cli.main(["pool", "--corpus", str(tmp_path / "nowhere")]) == 2
        assert "local-ocr:" in capsys.readouterr().err
