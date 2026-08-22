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
from local_ocr.headpass import (
    PROMPT,
    HeadPass,
    band,
    completes,
    extends,
    fragment,
    heading,
    missing,
    usable,
)

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

    def test_a_paragraph_citing_a_page_label_is_not_a_head(self) -> None:
        """The gate's own failure, and the one that kept the pass off the pages that needed it.

        `parse_page_label` searches the line, so a paragraph of body text that
        cites another page answered yes to `is this line a running head` and the
        second look was never taken. Nine of the 200 golden-dev readings open
        with a paragraph that does this.
        """
        body = (
            "(II, §5, No. 1, p. 278, Proposition 2) that the homomorphisms are those "
            "of A VIII.202, and the argument of the preceding paragraph applies to each "
            "of them without change, which is what was to be shown here.\n"
        )
        assert len(body.strip()) > 90
        assert missing(body)

    def test_a_paragraph_citing_a_section_is_not_a_head_either(self) -> None:
        body = (
            "b) On suppose desormais que A est la seule sous-A-algebre de K qui soit un "
            "corps, et on applique le resultat du § 3 a la famille consideree ci-dessus.\n"
        )
        assert missing(body)

    def test_a_display_opener_is_not_a_head(self) -> None:
        """A letterless line reads as a bare folio to the capitals test, and `\\[` is not one."""
        assert missing("\\[\n= \\prod_{x \\in H} g_1 c(s(x)^{-1})\n")
        assert missing("\\(\n")

    def test_a_bare_folio_still_is_one(self) -> None:
        assert not missing("496\n\nTABLE DES MATIERES\n")

    def test_a_head_that_ends_in_a_full_stop_is_still_a_head(self) -> None:
        """hist prints the stop, and the veto on it cost 56 pages of that volume.

        Every one of them was read three times and went dead on a running head
        the page really carries. Across the 4809 raw readings on disk there are
        112 first lines of 90 characters or fewer that end in a stop, and 97 of
        them read as a head under the rule as it stands now, over 60 distinct
        pages. The other 15 are mixed case sentences.
        """
        for head in (
            "234  23. HAAR MEASURE. CONVOLUTION.",
            "17. INFINITESIMAL CALCULUS.",
            "PREFACE.",
            "TABLE OF CONTENTS.",
        ):
            assert not missing(f"{head}\n\nThe body of the page follows here.\n"), head


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

    def test_a_capitals_head_with_a_stop_comes_back(self) -> None:
        """The full stop is not what tells a head from a sentence. The case is."""
        assert usable("23. HAAR MEASURE. CONVOLUTION.") == "23. HAAR MEASURE. CONVOLUTION."

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

    def test_a_head_the_page_already_opens_with_is_not_prepended_twice(self, page: Path) -> None:
        """What keeps a gate that fires more often from being worse than one that fires less.

        The gate can be wrong the other way round: a head that is genuinely up
        there and that fails the test for some small reason, here the small
        capitals of a title come back lower case from the page and in capitals
        from the strip. The strip then hands back the line the page already opens
        with, and putting that on the front would give the page two heads, which
        is a worse reading than the one that arrived.

        Passed through rather than corrected, because this module never rewrites
        a line the reader produced. The page is still rejected by rule 4 and read
        again, which is the outcome the module promises for a page it cannot fix.
        """
        body = "Table des matieres\n\nCHAPITRE VIII. Modules et anneaux semi-simples\n"
        stub = Stub(page=body, head="TABLE DES MATIERES")
        reader = HeadPass(stub)
        assert asyncio.run(reader.read(page, "read this")) == body
        assert reader.asked == 1 and reader.fixed == 0

    def test_a_refused_second_look_does_not_lose_the_page(self, page: Path) -> None:
        stub = Declines()
        assert asyncio.run(HeadPass(stub).read(page, "read this")) == BARE
        assert stub.calls == 2

    def test_a_page_image_the_cropper_cannot_open_does_not_lose_its_reading(
        self, tmp_path: Path
    ) -> None:
        """The crop opens the page a second time and it can fail on its own.

        Five pages of the contract run were read and then thrown away, because
        an image Pillow will not decode raised out of the second look and the
        batch recorded a refusal for a page it had in hand. The pages this
        happens to are the ones whose reading is worth most: a page image
        truncated by an interrupted copy has nothing else left that came off it.
        """
        broken = tmp_path / "0027.png"
        broken.write_bytes(b"\x89PNG and then nothing that decodes")
        stub = Stub()
        reader = HeadPass(stub)
        assert asyncio.run(reader.read(broken, "read this")) == BARE
        assert reader.asked == 1 and reader.fixed == 0
        assert len(stub.seen) == 1, "the strip was never asked for"

    def test_a_second_look_that_dies_on_the_wire_does_not_lose_the_page(self, page: Path) -> None:
        class Drops:
            def __init__(self) -> None:
                self.calls = 0

            async def read(self, image: Path, prompt: str) -> str:
                self.calls += 1
                if self.calls == 1:
                    return BARE
                raise ConnectionError("all connection attempts failed")

        assert asyncio.run(HeadPass(Drops()).read(page, "read this")) == BARE

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


HALF = "ANNEAUX\n\nPROPOSITION 7. Let A be a ring, and let X be an indeterminate over it.\n"
WHOLE = (
    "ANNEAUX A I.109\n\nPROPOSITION 7. Let A be a ring, and let X be an indeterminate over it.\n"
)


class Volume:
    """A reader for a volume that prints a page label, with pages that lost it.

    The page answer is chosen by the caller, one to a call, so a batch of pages
    can be walked past a single wrapper the way a real batch is. The strip
    answers the same head every time, with the label on it.
    """

    def __init__(self, pages: list[str], head: str = "ANNEAUX A I.109") -> None:
        self.pages = list(pages)
        self.head = head
        self.strips = 0

    async def read(self, image: Path, prompt: str) -> str:
        if prompt == PROMPT:
            self.strips += 1
            return self.head
        return self.pages.pop(0)


class TestTheLostPageLabel:
    """A head that is there and short of its label, on a volume that prints one.

    71 pages of four volumes are this, every one read three times and then dead,
    and none of them was ever asked about because none of them looked wrong.
    """

    def walk(self, reader: HeadPass, page: Path, count: int) -> list[str]:
        return [asyncio.run(reader.read(page, "read this")) for _ in range(count)]

    def test_nothing_is_believed_about_a_volume_from_too_few_pages(self, page: Path) -> None:
        """Seven labelled pages is not yet a volume that prints labels."""
        reader = HeadPass(Volume([WHOLE] * 7))
        self.walk(reader, page, 7)
        assert not reader.wants_label()
        assert reader.asked == 0

    def test_a_volume_that_labels_its_pages_is_learned_from_its_pages(self, page: Path) -> None:
        reader = HeadPass(Volume([WHOLE] * 8))
        self.walk(reader, page, 8)
        assert reader.wants_label()
        assert reader.asked == 0, "every one of them already had its label"

    def test_the_label_is_put_back_on_the_page_that_lost_it(self, page: Path) -> None:
        inner = Volume([WHOLE] * 8 + [HALF])
        reader = HeadPass(inner)
        out = self.walk(reader, page, 9)
        assert inner.strips == 1, "one strip, for the one page that needed it"
        assert out[-1].startswith("ANNEAUX A I.109\n")
        assert reader.completed == 1
        assert reader.fixed == 0, "the head was there, so nothing was prepended"

    def test_the_body_is_left_exactly_as_it_arrived(self, page: Path) -> None:
        reader = HeadPass(Volume([WHOLE] * 8 + [HALF]))
        out = self.walk(reader, page, 9)[-1]
        assert out.split("\n", 1)[1] == HALF.split("\n", 1)[1]

    def test_a_volume_that_prints_no_label_is_never_asked(self, page: Path) -> None:
        """foot-number volumes carry a title and a locator and nothing else."""
        reader = HeadPass(Volume([HALF] * 12))
        self.walk(reader, page, 12)
        assert not reader.wants_label()
        assert reader.asked == 0

    def test_a_strip_that_answers_a_different_head_changes_nothing(self, page: Path) -> None:
        """The guard against putting one page's head on another page."""
        reader = HeadPass(Volume([WHOLE] * 8 + [HALF], head="POLYNOMES A IV.7"))
        out = self.walk(reader, page, 9)[-1]
        assert out == HALF
        assert reader.completed == 0
        assert reader.asked == 9 - 8, "asked once, and the answer was not usable"

    def test_a_strip_with_no_label_on_it_changes_nothing(self, page: Path) -> None:
        reader = HeadPass(Volume([WHOLE] * 8 + [HALF], head="ANNEAUX"))
        assert self.walk(reader, page, 9)[-1] == HALF
        assert reader.completed == 0

    def test_a_page_with_no_head_at_all_still_takes_the_other_path(self, page: Path) -> None:
        """Both repairs live here and the missing head is the older one."""
        reader = HeadPass(Volume([WHOLE] * 8 + [BARE]))
        out = self.walk(reader, page, 9)[-1]
        assert out.startswith("ANNEAUX A I.109\n\n")
        assert out.endswith(BARE)
        assert reader.fixed == 1 and reader.completed == 0


class TestCompletes:
    """The guard, on its own, because it is what makes the edit safe."""

    def test_the_page_head_has_to_be_inside_the_strip_head(self) -> None:
        assert completes("ANNEAUX A I.109", HALF)
        assert not completes("POLYNOMES A IV.7", HALF)

    def test_the_strip_head_has_to_carry_a_label(self) -> None:
        assert not completes("ANNEAUX", HALF)

    def test_a_page_that_already_has_a_label_is_left_alone(self) -> None:
        assert not completes("ANNEAUX A I.109", WHOLE)

    def test_the_spacing_and_the_case_do_not_decide_it(self) -> None:
        """The two came out of two different requests, so they differ in both."""
        assert completes("Anneaux  A I.109", HALF)

    def test_the_label_digits_do_not_answer_for_the_head(self) -> None:
        """The commonest half a head is one digit long, and the label has four.

        On ac-viii-ix-fr the reader keeps "§ 2" and drops the rest. Run the
        containment over the whole strip answer and the 2 in "AC VIII.32" is
        enough to accept a head off another page, which is how a page ends up
        carrying a section it is not in. The label comes out first.
        """
        page = "§ 2\n\nPROPOSITION 7. Soit A un anneau.\n"
        assert completes("AC VIII.14  DIMENSION  § 2", page)
        assert not completes("AC VIII.32  DIMENSION  § 3", page)

    def test_the_kept_fragment_sits_at_either_end(self) -> None:
        """Body pages print the label first, exercise pages print it last.

        Both lines below were read off ac-viii-ix-fr, which is why the guard
        cannot be a prefix test or a suffix test.
        """
        body = "§ 2\n\nSoit A un anneau noetherien.\n"
        exercises = "§ 4\n\n1) Soit k un corps.\n"
        assert completes("AC VIII.14  DIMENSION  § 2", body)
        assert completes("§ 4  EXERCICES  AC VIII.93", exercises)


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


class TestThePrompt:
    """The strip instruction is the whole of what the pass can control.

    Nothing here checks wording for its own sake. Each of these is a shape the
    prompt lost a measurement to, so a rewrite that drops one of them is a
    rewrite that puts back a defect somebody already paid for.
    """

    def test_it_asks_for_both_ends_of_the_line(self) -> None:
        """A Bourbaki head has three parts and the middle one is the trap."""
        assert "left edge to its right edge" in PROMPT
        assert "either end" in PROMPT

    def test_it_gives_an_example_of_each_head_shape(self) -> None:
        """The two-part English head and the three-part French one."""
        assert "18 ALGEBRAIC STRUCTURES Ch. I" in PROMPT
        assert "EXTENSIONS GALOISIENNES" in PROMPT

    def test_it_still_offers_a_way_out(self) -> None:
        """A front matter page prints no head and must be allowed to say so."""
        assert "NONE" in PROMPT
        assert usable("NONE") is None


PIECE = "§ 2\n\nPROPOSITION 7. Let A be a ring, and let X be an indeterminate over it.\n"


class TestFragment:
    """One end of the head and nothing else, which every other test here takes for a head."""

    def test_a_section_marker_on_its_own_is_a_piece_of_a_head(self) -> None:
        for line in ("§ 2", "§ 10", "§ 5.", "N° 2", "§ 3, n° 1"):
            assert fragment(line), line

    def test_a_head_with_a_title_in_it_is_not(self) -> None:
        for line in ("§ 2 EXERCICES TG I.91", "ANNEAUX", "Exercices", "TABLE DES MATIERES"):
            assert not fragment(line), line

    def test_a_line_carrying_a_page_label_is_not(self) -> None:
        """It is already enough to file the page under, and completes owns the rest."""
        assert not fragment("A I.109")

    def test_a_paragraph_is_not_a_fragment_either(self) -> None:
        """Whatever is wrong with a page of prose, prepending a head is the repair for it."""
        assert not fragment(BARE.strip())

    def test_the_reading_that_cost_the_measurement(self) -> None:
        """323 readings on disk open with one of these and the volume rule asked about 13."""
        assert missing(PIECE) is False, "which is why nothing else here catches it"
        assert fragment("§ 2")


class TestExtends:
    """The guard on the fragment repair, which is the whole of what makes it safe."""

    def test_the_fragment_has_to_be_one_end_of_the_strip_head(self) -> None:
        assert extends("§ 2 EXERCICES TG I.91", PIECE)
        assert extends("AC VIII.84 DIMENSION § 2", PIECE)

    def test_a_head_belonging_to_another_section_is_refused(self) -> None:
        assert not extends("§ 5 EXERCICES TG I.97", PIECE)

    def test_a_digit_buried_in_the_middle_does_not_count(self) -> None:
        """A one character key is inside almost any head, so anywhere is not enough."""
        assert not extends("§ 5 EXERCICES 2 BIS TG I.97", PIECE)

    def test_a_strip_that_hands_back_the_fragment_extends_nothing(self) -> None:
        assert not extends("§ 2", PIECE)

    def test_the_label_digits_do_not_answer_for_the_fragment(self) -> None:
        """TG I.92 ends in a 2 and that says nothing about the section."""
        assert not extends("SOMETHING ELSE TG I.92", PIECE)

    def test_the_spacing_and_the_case_do_not_decide_it(self) -> None:
        assert extends("§ 2   Exercices   TG I.91", PIECE)


class TestTheFragmentRepair:
    """The repair end to end, on a volume the old gate would never have opened for."""

    def walk(self, reader: HeadPass, page: Path, count: int) -> list[str]:
        return [asyncio.run(reader.read(page, "read this")) for _ in range(count)]

    def test_the_rest_of_the_head_is_put_back(self, page: Path) -> None:
        inner = Volume([PIECE], head="§ 2 EXERCICES TG I.91")
        reader = HeadPass(inner)
        out = self.walk(reader, page, 1)[-1]
        assert inner.strips == 1
        assert out.startswith("§ 2 EXERCICES TG I.91\n")
        assert reader.mended == 1
        assert reader.fixed == 0 and reader.completed == 0

    def test_it_does_not_wait_for_the_volume_to_show_anything(self, page: Path) -> None:
        """The point of it. The volume rule needs eight pages and never opens on these.

        Half of an exercises batch comes back with no page label, so the share
        sits at 0.5625 against a threshold of 0.6 and the pages that need the
        second look are the votes keeping it shut.
        """
        reader = HeadPass(Volume([PIECE], head="§ 2 EXERCICES TG I.91"))
        self.walk(reader, page, 1)
        assert not reader.wants_label(), "one page, so it has shown nothing at all"
        assert reader.mended == 1

    def test_the_body_is_left_exactly_as_it_arrived(self, page: Path) -> None:
        reader = HeadPass(Volume([PIECE], head="§ 2 EXERCICES TG I.91"))
        out = self.walk(reader, page, 1)[-1]
        assert out.split("\n", 1)[1] == PIECE.split("\n", 1)[1]

    def test_a_strip_answering_a_different_head_changes_nothing(self, page: Path) -> None:
        reader = HeadPass(Volume([PIECE], head="§ 5 EXERCICES TG I.97"))
        assert self.walk(reader, page, 1)[-1] == PIECE
        assert reader.mended == 0
        assert reader.asked == 1

    def test_a_strip_with_no_head_on_it_changes_nothing(self, page: Path) -> None:
        reader = HeadPass(Volume([PIECE], head="NONE"))
        assert self.walk(reader, page, 1)[-1] == PIECE
        assert reader.mended == 0

    def test_a_head_that_is_whole_is_never_asked_about(self, page: Path) -> None:
        reader = HeadPass(Volume([WHOLE]))
        assert self.walk(reader, page, 1)[-1] == WHOLE
        assert reader.asked == 0 and reader.mended == 0

    def test_a_foot_number_volume_keeps_the_repair(self, page: Path) -> None:
        """No page label anywhere, so completes can never fire and this still can.

        13 of the 189 fragment pages are on volumes that print a bare folio or
        nothing at all, and the label test refuses every one of them.
        """
        reader = HeadPass(Volume([PIECE], head="§ 2 EXERCICES 67"))
        out = self.walk(reader, page, 1)[-1]
        assert out.startswith("§ 2 EXERCICES 67\n")
        assert reader.mended == 1

    def test_the_run_line_says_what_it_did(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reader = HeadPass(Volume([PIECE], head="§ 2 EXERCICES TG I.91"))
        out = TestBatchLine().run(tmp_path, reader, capsys)
        assert "put the rest of the head back on 1" in out


OPENER = (
    "§ 5. APPLICATIONS OUVERTES ET APPLICATIONS FERMÉES\n\n"
    "DÉFINITION 1. Soient X et Y deux espaces topologiques.\n"
)


class TestTheBodyHeading:
    """The line that looks more like a running head than the running head does.

    On a page that opens a section the body's heading is set a third of the way
    down and the reader brings that back instead of the head. It is in capitals,
    it is the right length, and `parse_section_locator` finds a marker on it, so
    every other test in this module says the page has its head.
    """

    def test_a_section_number_with_a_stop_and_a_title_is_the_body_heading(self) -> None:
        for line in (
            "§ 5. APPLICATIONS OUVERTES ET APPLICATIONS FERMÉES",
            "§ 10. APPLICATIONS PROPRES",
            "§ 2. Relèvement des idéaux premiers.",
            "§ 3. Corps de représentants ...................................",
        ):
            assert heading(line), line

    def test_a_running_head_is_not(self) -> None:
        """The volumes print the stop in the body and leave it out of the head."""
        for line in (
            "TG I.30 STRUCTURES TOPOLOGIQUES § 5",
            "§ 2 EXERCICES TG I.91",
            "§ 4, N° 1 ALGÈBRES DE LIE NILPOTENTES 55",
            "§ 6, No 1 ESPACES POLONAIS; ESPACES SOUSLINIENS",
        ):
            assert not heading(line), line

    def test_a_bare_marker_with_a_stop_is_a_fragment_and_not_a_heading(self) -> None:
        """Something has to follow the stop.

        `§ 5.` on its own is a piece of a head with the stop misread onto it, and
        the repair for that is the rest of the head in its place, not a head in
        front of it.
        """
        assert not heading("§ 5.")
        assert fragment("§ 5.")

    def test_the_page_is_treated_as_having_no_head(self) -> None:
        assert missing(OPENER)

    def test_the_head_goes_in_front_and_the_heading_stays_in_the_body(self, page: Path) -> None:
        reader = HeadPass(Volume([OPENER], head="TG I.30 STRUCTURES TOPOLOGIQUES § 5"))
        out = asyncio.run(reader.read(page, "read this"))
        assert out.startswith("TG I.30 STRUCTURES TOPOLOGIQUES § 5\n\n")
        assert out.endswith(OPENER)
        assert reader.fixed == 1 and reader.mended == 0

    def test_a_page_that_prints_no_head_is_left_alone(self, page: Path) -> None:
        """Nine of the 42 are ac-x-fr, whose section opening pages print none."""
        reader = HeadPass(Volume([OPENER], head="NONE"))
        assert asyncio.run(reader.read(page, "read this")) == OPENER
        assert reader.asked == 1 and reader.fixed == 0

    def test_a_strip_that_read_past_the_band_is_refused(self) -> None:
        """The same line arriving from the other direction, which is worse.

        Prepending it would give the page the body heading twice and still no
        head.
        """
        assert usable("§ 5. APPLICATIONS OUVERTES ET APPLICATIONS FERMÉES") is None
