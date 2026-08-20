"""Rasterising a golden set, without a PDF and without pdftoppm.

The renderer takes whatever it runs commands with, so everything below is about
the two things that can be wrong on a machine where pdftoppm works fine: the
command built for a page, and the name the image ends up under. A page written
as `42.png` instead of `0042.png` is invisible to the batch reader, which finds
nothing and reports success.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from local_ocr import pageimages


def png(path: Path, dpi: int | None, width: int = 8, height: int = 8) -> None:
    """The smallest PNG that says what it was rendered at, or refuses to."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    out = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
    ]
    if dpi is not None:
        metre = round(dpi / 0.0254)
        out.append(chunk(b"pHYs", struct.pack(">IIB", metre, metre, 1)))
    out.append(chunk(b"IDAT", zlib.compress(b"\x00" * (width + 1) * height)))
    out.append(chunk(b"IEND", b""))
    path.write_bytes(b"".join(out))


BOOKS = """
books:
  - id: alg-viii
    title: Algebra VIII
    pdf: pdf/en/algebra-viii.pdf
  - id: ts-i-ii-fr
    title: Theorie des ensembles I-II
    pdf: pdf/fr/ts-i-ii.pdf
  - id: nopdf
    title: A volume nobody has a scan of
"""


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "books.yaml").write_text(BOOKS, encoding="utf-8")
    for name in ("pdf/en", "pdf/fr"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "pdf" / "en" / "algebra-viii.pdf").write_bytes(b"%PDF-1.4\n")
    (tmp_path / "pdf" / "fr" / "ts-i-ii.pdf").write_bytes(b"%PDF-1.4\n")
    return tmp_path


class Fake:
    """Stands in for pdftoppm, writing what pdftoppm would write."""

    def __init__(self, width: int = 3) -> None:
        self.width = width
        self.commands: list[list[str]] = []

    def run(self, command: list[str]) -> None:
        self.commands.append(command)
        page = int(command[command.index("-f") + 1])
        dpi = int(command[command.index("-r") + 1])
        prefix = Path(command[-1])
        png(prefix.with_name(f"{prefix.name}-{page:0{self.width}d}.png"), dpi)


class TestSources:
    def test_every_volume_with_a_pdf_is_found(self, corpus: Path) -> None:
        found = pageimages.sources(corpus)
        assert found["alg-viii"] == corpus / "pdf/en/algebra-viii.pdf"
        assert found["ts-i-ii-fr"] == corpus / "pdf/fr/ts-i-ii.pdf"

    def test_a_volume_with_no_pdf_is_absent_rather_than_empty(self, corpus: Path) -> None:
        # The caller reports it as a page it could not render, which is true, and
        # an empty path would instead be a pdftoppm error about a missing file.
        assert "nopdf" not in pageimages.sources(corpus)


class TestImagePath:
    def test_the_page_number_is_four_digits(self, corpus: Path) -> None:
        # Which is what the fleet writes and therefore what the batch reader
        # globs for. Three digits is a page nothing ever reads.
        assert pageimages.image_path(corpus, "alg-viii", 42).name == "0042.png"

    def test_a_volume_over_a_thousand_pages_is_not_truncated(self, corpus: Path) -> None:
        assert pageimages.image_path(corpus, "alg-viii", 1234).name == "1234.png"


class TestCommand:
    def test_it_is_the_command_the_corpus_renders_with(self, corpus: Path) -> None:
        # pdfsrc.Source.Render on the Go side, flag for flag. A bake off page
        # rendered differently from the fleet's pages measures the renderer.
        render = pageimages.Renderer(corpus=corpus, dpi=300)
        command = render.command(Path("book.pdf"), 42, Path("out"))
        assert command == [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-gray",
            "-f",
            "42",
            "-l",
            "42",
            "book.pdf",
            "out",
        ]

    def test_one_page_is_asked_for_at_both_ends(self, corpus: Path) -> None:
        render = pageimages.Renderer(corpus=corpus)
        command = render.command(Path("book.pdf"), 7, Path("out"))
        assert command[command.index("-f") + 1] == command[command.index("-l") + 1] == "7"

    def test_the_dpi_reaches_the_command(self, corpus: Path) -> None:
        # The 600 dpi benchmark is this and nothing else.
        render = pageimages.Renderer(corpus=corpus, dpi=600)
        command = render.command(Path("book.pdf"), 1, Path("out"))
        assert command[command.index("-r") + 1] == "600"

    def test_colour_can_be_asked_for(self, corpus: Path) -> None:
        render = pageimages.Renderer(corpus=corpus, gray=False)
        assert "-gray" not in render.command(Path("book.pdf"), 1, Path("out"))


class TestRender:
    def test_the_padding_pdftoppm_chose_is_not_the_padding_kept(self, corpus: Path) -> None:
        # pdftoppm pads to the width of the volume's page count, so the same page
        # of a 90 page volume and a 900 page one come out named differently. The
        # file is found and renamed rather than predicted.
        fake = Fake(width=3)
        render = pageimages.Renderer(corpus=corpus, run=fake)
        out = render.render("alg-viii", 42, corpus / "pdf/en/algebra-viii.pdf")
        assert out == corpus / "images" / "alg-viii" / "0042.png"
        assert pageimages.dpi_of(out) == pageimages.DEFAULT_DPI

    def test_a_volume_wide_enough_to_need_no_rename_still_lands_right(self, corpus: Path) -> None:
        fake = Fake(width=4)
        render = pageimages.Renderer(corpus=corpus, run=fake)
        out = render.render("alg-viii", 42, corpus / "pdf/en/algebra-viii.pdf")
        assert out.exists()

    def test_nothing_of_the_working_name_is_left_behind(self, corpus: Path) -> None:
        render = pageimages.Renderer(corpus=corpus, run=Fake())
        render.render("alg-viii", 42, corpus / "pdf/en/algebra-viii.pdf")
        assert list((corpus / "images" / "alg-viii").iterdir()) == [
            corpus / "images" / "alg-viii" / "0042.png"
        ]

    def test_a_page_that_produced_nothing_is_an_error_and_not_a_missing_file(
        self, corpus: Path
    ) -> None:
        # pdftoppm exits zero when asked for a page past the end of a volume and
        # writes no image. Silence there would be a page the bake off scores as
        # unread by every model at once.
        class Nothing:
            def run(self, command: list[str]) -> None:
                return None

        render = pageimages.Renderer(corpus=corpus, run=Nothing())
        with pytest.raises(pageimages.NoSource):
            render.render("alg-viii", 9999, corpus / "pdf/en/algebra-viii.pdf")


class TestDpiOf:
    def test_a_rendered_page_says_what_it_was_rendered_at(self, tmp_path: Path) -> None:
        png(tmp_path / "p.png", 300)
        assert pageimages.dpi_of(tmp_path / "p.png") == 300

    def test_six_hundred_is_read_as_six_hundred(self, tmp_path: Path) -> None:
        # pixels per metre is not an integer number of dpi, so this is a rounding
        # question and 599 would be a page re-rendered on every run.
        png(tmp_path / "p.png", 600)
        assert pageimages.dpi_of(tmp_path / "p.png") == 600

    def test_a_page_that_does_not_say_admits_it(self, tmp_path: Path) -> None:
        png(tmp_path / "p.png", None)
        assert pageimages.dpi_of(tmp_path / "p.png") is None

    def test_something_that_is_not_a_png_is_not_guessed_at(self, tmp_path: Path) -> None:
        (tmp_path / "p.png").write_bytes(b"not a png at all")
        assert pageimages.dpi_of(tmp_path / "p.png") is None


class TestBuild:
    def test_pages_already_on_disk_at_this_dpi_are_not_rendered_again(self, corpus: Path) -> None:
        there = pageimages.image_path(corpus, "alg-viii", 42)
        there.parent.mkdir(parents=True)
        png(there, 300)
        fake = Fake()
        built = pageimages.build(
            ["alg-viii/0042"], corpus, renderer=pageimages.Renderer(corpus=corpus, run=fake)
        )
        assert built.had == ["alg-viii/0042"]
        assert fake.commands == []

    def test_a_page_on_disk_at_another_dpi_is_rendered_again_and_reported(
        self, corpus: Path
    ) -> None:
        # This is not hypothetical. The corpus held two lie-vii-ix pages at
        # 600 dpi from a fleet retry, in a directory of 300 dpi pages, and they
        # were in golden-dev.
        there = pageimages.image_path(corpus, "alg-viii", 42)
        there.parent.mkdir(parents=True)
        png(there, 600)
        built = pageimages.build(
            ["alg-viii/0042"], corpus, renderer=pageimages.Renderer(corpus=corpus, run=Fake())
        )
        assert built.made == ["alg-viii/0042"]
        assert built.redone == ["alg-viii/0042"]
        assert pageimages.dpi_of(there) == 300

    def test_a_page_that_will_not_say_what_it_is_is_rendered_again(self, corpus: Path) -> None:
        there = pageimages.image_path(corpus, "alg-viii", 42)
        there.parent.mkdir(parents=True)
        png(there, None)
        built = pageimages.build(
            ["alg-viii/0042"], corpus, renderer=pageimages.Renderer(corpus=corpus, run=Fake())
        )
        assert built.redone == ["alg-viii/0042"]

    def test_a_page_counted_once_however_it_got_there(self, corpus: Path) -> None:
        # redone pages are in made as well, so a total that added both would
        # report more pages than the set has.
        there = pageimages.image_path(corpus, "alg-viii", 42)
        there.parent.mkdir(parents=True)
        png(there, 600)
        built = pageimages.build(
            ["alg-viii/0042"], corpus, renderer=pageimages.Renderer(corpus=corpus, run=Fake())
        )
        assert built.total == 1

    def test_overwrite_renders_them_anyway(self, corpus: Path) -> None:
        there = pageimages.image_path(corpus, "alg-viii", 42)
        there.parent.mkdir(parents=True)
        png(there, 300)
        built = pageimages.build(
            ["alg-viii/0042"],
            corpus,
            overwrite=True,
            renderer=pageimages.Renderer(corpus=corpus, run=Fake()),
        )
        assert built.made == ["alg-viii/0042"]
        assert built.redone == []

    def test_a_volume_with_no_pdf_is_reported_and_the_rest_still_run(self, corpus: Path) -> None:
        built = pageimages.build(
            ["nopdf/0001", "alg-viii/0042"],
            corpus,
            renderer=pageimages.Renderer(corpus=corpus, run=Fake()),
        )
        assert built.made == ["alg-viii/0042"]
        assert [page for page, _ in built.failed] == ["nopdf/0001"]
        assert "nopdf" in built.failed[0][1]

    def test_every_page_asked_for_is_accounted_for(self, corpus: Path) -> None:
        ids = ["alg-viii/0042", "ts-i-ii-fr/0007", "nopdf/0001"]
        built = pageimages.build(
            ids, corpus, renderer=pageimages.Renderer(corpus=corpus, run=Fake())
        )
        assert built.total == len(ids)

    def test_the_page_number_of_the_id_is_the_page_asked_of_pdftoppm(self, corpus: Path) -> None:
        # `alg-viii/0042` is a leading zero away from being octal, and 0042 read
        # that way is page 34.
        fake = Fake()
        pageimages.build(
            ["alg-viii/0042"], corpus, renderer=pageimages.Renderer(corpus=corpus, run=fake)
        )
        command = fake.commands[0]
        assert command[command.index("-f") + 1] == "42"
