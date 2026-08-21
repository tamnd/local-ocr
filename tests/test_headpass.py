"""The second look at the top of the page, with the reader stubbed.

The pass is the answer to the one acceptance rule that rejected 177 of 200 pages
on the first bake off run, so what it must never do matters as much as what it
does: never rewrite a line the reader produced, never turn a read page into a
refused one, never put a paragraph on the front of a page and call it a head.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from local_ocr.batch import Refused
from local_ocr.headpass import HeadPass, band, missing, usable

HEADED = "ARTINIAN MODULES AND NOETHERIAN MODULES A VIII.10\n\nPROPOSITION 7. Let A be a ring.\n"
BARE = "PROPOSITION 7. Let A be a ring, and let X be an indeterminate over it.\n"


class Stub:
    """A reader that answers the page one way and the strip another."""

    def __init__(self, page: str = BARE, head: str = "A VIII.10 ARTINIAN MODULES") -> None:
        self.page = page
        self.head = head
        self.seen: list[Path] = []
        self.prompts: list[str] = []

    async def read(self, image: Path, prompt: str) -> str:
        self.seen.append(image)
        self.prompts.append(prompt)
        return self.head if len(self.seen) > 1 else self.page


class Declines:
    """A reader that reads the page and refuses the strip."""

    def __init__(self) -> None:
        self.calls = 0

    async def read(self, image: Path, prompt: str) -> str:
        self.calls += 1
        if self.calls == 1:
            return BARE
        raise Refused("no")


@pytest.fixture
def page(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "0027.png"
    Image.new("L", (1831, 2776), color=255).save(path)
    return path


class TestMissing:
    def test_a_reading_that_starts_with_a_running_head_needs_nothing(self) -> None:
        assert not missing(HEADED)

    def test_a_reading_that_starts_with_prose_needs_a_head(self) -> None:
        assert missing(BARE)

    def test_a_page_label_on_the_first_line_is_a_head(self) -> None:
        assert not missing("A VIII.10\n\nPROPOSITION 7. Let A be a ring.\n")

    def test_leading_blank_lines_are_not_the_first_line(self) -> None:
        assert not missing("\n\n" + HEADED)

    def test_an_empty_reading_counts_as_missing(self) -> None:
        assert missing("   \n\n")


class TestUsable:
    def test_a_head_comes_back_as_itself(self) -> None:
        assert usable("ARTINIAN MODULES AND NOETHERIAN MODULES A VIII.10\n") is not None

    def test_none_means_the_page_prints_no_head(self) -> None:
        assert usable("NONE") is None

    def test_a_paragraph_is_not_a_head(self) -> None:
        # The failure this whole module exists to undo, arriving from the other
        # direction: a model that transcribes the body of the strip instead.
        assert usable("Let A be a ring and let X be an indeterminate over it. " * 3) is None

    def test_a_sentence_is_not_a_head(self) -> None:
        assert usable("The strip shows the top of a page.") is None

    def test_nothing_at_all_is_not_a_head(self) -> None:
        assert usable("   ") is None

    def test_a_fenced_answer_is_unwrapped(self) -> None:
        assert usable("`A VIII.10`") == "A VIII.10"


class TestBand:
    def test_the_strip_is_the_top_of_the_page_and_its_full_width(
        self, page: Path, tmp_path: Path
    ) -> None:
        from PIL import Image

        out = band(page, tmp_path / "strip.png", 0.12)
        with Image.open(out) as strip, Image.open(page) as whole:
            assert strip.width == whole.width
            assert strip.height == int(whole.height * 0.12)

    def test_a_tiny_page_still_yields_a_strip(self, tmp_path: Path) -> None:
        from PIL import Image

        small = tmp_path / "small.png"
        Image.new("L", (4, 4), color=255).save(small)
        out = band(small, tmp_path / "strip.png", 0.01)
        with Image.open(out) as strip:
            assert strip.height == 1


class TestPass:
    def test_a_page_with_a_head_is_read_once_and_left_alone(self, page: Path) -> None:
        stub = Stub(page=HEADED)
        reader = HeadPass(stub)
        assert asyncio.run(reader.read(page, "read this")) == HEADED
        assert len(stub.seen) == 1
        assert reader.asked == 0

    def test_a_page_without_one_gets_the_head_asked_for_and_prepended(self, page: Path) -> None:
        stub = Stub()
        reader = HeadPass(stub)
        out = asyncio.run(reader.read(page, "read this"))
        assert out.splitlines()[0] == "A VIII.10 ARTINIAN MODULES"
        assert BARE.strip() in out
        assert reader.asked == 1 and reader.fixed == 1

    def test_the_second_look_sees_a_strip_and_not_the_page(self, page: Path) -> None:
        from PIL import Image

        # Measured inside the read, because the strip lives in a temporary
        # directory that is gone by the time the call returns, which is the
        # point of it: a crop is not something to leave lying beside a corpus.
        sizes: list[tuple[int, int]] = []

        class Measuring(Stub):
            async def read(self, image: Path, prompt: str) -> str:
                with Image.open(image) as seen:
                    sizes.append(seen.size)
                return await super().read(image, prompt)

        asyncio.run(HeadPass(Measuring()).read(page, "read this"))
        assert len(sizes) == 2
        assert sizes[1][0] == sizes[0][0]
        assert sizes[1][1] < sizes[0][1]

    def test_the_second_look_carries_its_own_instruction(self, page: Path) -> None:
        stub = Stub()
        asyncio.run(HeadPass(stub).read(page, "the fleet prompt"))
        assert stub.prompts[0] == "the fleet prompt"
        assert "running head" in stub.prompts[1]

    def test_the_second_look_asks_for_the_folio(self, page: Path) -> None:
        # Without this sentence the model answered TABLE DES MATIERES where the
        # page prints 496 TABLE DES MATIERES, and the rule rejected it.
        stub = Stub()
        asyncio.run(HeadPass(stub).read(page, "the fleet prompt"))
        assert "page number" in stub.prompts[1]

    def test_a_strip_with_no_head_leaves_the_reading_as_it_was(self, page: Path) -> None:
        stub = Stub(head="NONE")
        reader = HeadPass(stub)
        assert asyncio.run(reader.read(page, "read this")) == BARE
        assert reader.asked == 1 and reader.fixed == 0

    def test_a_refused_second_look_does_not_lose_the_page(self, page: Path) -> None:
        stub = Declines()
        assert asyncio.run(HeadPass(stub).read(page, "read this")) == BARE
        assert stub.calls == 2

    def test_a_refused_first_look_is_still_a_refusal(self, page: Path) -> None:
        class Never:
            async def read(self, image: Path, prompt: str) -> str:
                raise Refused("the answer hit the token limit and is truncated")

        with pytest.raises(Refused):
            asyncio.run(HeadPass(Never()).read(page, "read this"))


class TestBatchLine:
    """What a run says about the second look, which is the only account of it."""

    def run(self, tmp_path: Path, reader: HeadPass, capsys: pytest.CaptureFixture[str]) -> str:
        from PIL import Image

        from local_ocr.cli import ocr_batch

        src, dst = tmp_path / "in", tmp_path / "out"
        src.mkdir()
        Image.new("L", (1831, 2776), color=255).save(src / "0027.png")
        code = ocr_batch(
            [str(src), str(dst), "--ext", "png", "--prompt", "read this"],
            reader=reader,
        )
        assert code == 0
        return capsys.readouterr().out

    def test_a_run_that_repaired_a_page_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self.run(tmp_path, HeadPass(Stub()), capsys)
        assert "head pass: asked on 1 pages, put a head on 1" in out

    def test_a_run_that_needed_nothing_stays_quiet(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = self.run(tmp_path, HeadPass(Stub(page=HEADED)), capsys)
        assert "head pass" not in out


class Counting(Stub):
    """A reader that reports what each call cost, the way a vLLM server does."""

    def __init__(self, page: str = BARE, head: str = "A VIII.10 ARTINIAN MODULES") -> None:
        super().__init__(page=page, head=head)
        self.spent: dict[Path, tuple[int, int]] = {}

    async def read(self, image: Path, prompt: str) -> str:
        text = await super().read(image, prompt)
        # The strip is a tenth of the page, and it is charged like one.
        self.spent[image] = (1400, 700) if len(self.seen) == 1 else (160, 12)
        return text

    def usage(self, image: Path) -> tuple[int, int] | None:
        return self.spent.pop(image, None)


class TestUsage:
    """The head pass costs tokens on most pages, and they were going nowhere.

    The strip is written into a temporary directory that is deleted before
    anybody asks the wrapper what the page cost, so the strip's counts have to be
    moved onto the page's key while the strip still exists. On the M6 run the
    second look ran on 90 of 122 pages, so this is most of the difference between
    a sidecar that adds up and one that does not.
    """

    def test_a_page_that_needed_no_head_costs_what_the_page_cost(self, page: Path) -> None:
        reader = HeadPass(Counting(page=HEADED))
        asyncio.run(reader.read(page, "read this"))
        assert reader.usage(page) == (1400, 700)

    def test_a_page_that_needed_one_costs_the_page_and_the_strip(self, page: Path) -> None:
        reader = HeadPass(Counting())
        asyncio.run(reader.read(page, "read this"))
        assert reader.usage(page) == (1560, 712)

    def test_the_strip_is_not_charged_to_the_next_page(self, page: Path) -> None:
        reader = HeadPass(Counting())
        asyncio.run(reader.read(page, "read this"))
        assert reader.usage(page) == (1560, 712)
        assert reader.usage(page) is None

    def test_a_reader_that_does_not_count_is_still_a_reader(self, page: Path) -> None:
        # codex is a subprocess against a subscription and reports nothing, and
        # requiring a usage of every reader would break it for a sidecar field.
        reader = HeadPass(Stub())
        asyncio.run(reader.read(page, "read this"))
        assert reader.usage(page) is None

    def test_a_reader_whose_usage_raises_does_not_lose_the_page(self, page: Path) -> None:
        class Angry(Stub):
            def usage(self, image: Path) -> tuple[int, int] | None:
                raise RuntimeError("no")

        reader = HeadPass(Angry())
        assert "A VIII.10" in asyncio.run(reader.read(page, "read this"))
        assert reader.usage(page) is None
