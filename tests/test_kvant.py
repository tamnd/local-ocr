"""The Russian tier B set: its manifests, its gate, and the arithmetic of the draw.

Most of this needs no corpus, for the same reason `test_golden.py` needs none.
What is worth pinning is the shape of the recorded sets and the behaviour of the
gate, and both hold on a machine that has never seen a page of Kvant.

The two tests at the end do need the corpus and skip without it. They are here
because the gate's whole value is the number it prints on real data, and a gate
that has only ever been shown handwritten pages is a gate nobody has checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pdftoppm import Fake, png

from local_ocr import kvant, pageimages
from local_ocr.golden import Burned, Purpose

RUSSIAN = "Рассмотрим последовательность точек на плоскости и докажем, что она сходится. " * 6


def page(**over) -> kvant.Page:
    fields = {
        "issue": "kvant_2018_10",
        "year": 2018,
        "page_index": 16,
        "page_label": "14",
        "extraction": "native",
        "body": RUSSIAN,
        "path": Path("/nowhere/0016.md"),
    }
    fields.update(over)
    return kvant.Page(**fields)


def test_every_set_has_a_manifest_of_the_size_it_claims():
    for name, entry in kvant.SETS.items():
        assert len(kvant.read_manifest(name)) == entry.size, name


def test_dev_and_test_are_disjoint():
    """An overlap makes the held out set report a number that was tuned."""
    dev = set(kvant.read_manifest("kvant-dev"))
    test = set(kvant.read_manifest("kvant-test"))
    assert dev & test == set()


def test_the_ids_are_issue_and_sheet_and_are_unique():
    for name in kvant.SETS:
        ids = kvant.read_manifest(name)
        assert len(set(ids)) == len(ids), name
        for page_id in ids:
            issue, sep, number = page_id.partition("/")
            assert sep and issue.startswith("kvant_"), page_id
            assert number.isdigit() and len(number) == 4, page_id


def test_the_two_sets_cover_the_same_issues():
    """They are drawn to be compared, so a difference in coverage is a defect.

    A test set that is missing the issues dev is heaviest in would report a
    number about a different decade of the magazine than the one that was
    developed against.
    """
    dev = {page_id.partition("/")[0] for page_id in kvant.read_manifest("kvant-dev")}
    test = {page_id.partition("/")[0] for page_id in kvant.read_manifest("kvant-test")}
    assert dev == test


def test_the_held_out_set_says_so_in_its_own_header():
    """Somebody who opens the file and never reads the code should still know."""
    text = kvant.manifest("kvant-test").read_text(encoding="utf-8")
    assert "HELD OUT" in text
    assert "HELD OUT" not in kvant.manifest("kvant-dev").read_text(encoding="utf-8")


def test_the_headers_say_the_reference_is_the_publisher_text_layer():
    """The one thing a reader of a number off this set has to be told."""
    for name in kvant.SETS:
        assert "text layer" in kvant.manifest(name).read_text(encoding="utf-8")


def test_reading_the_held_out_set_for_development_raises(tmp_path):
    with pytest.raises(Burned):
        kvant.load("kvant-test", purpose=Purpose.DEVELOPMENT, corpus=tmp_path)


def test_an_unknown_set_is_named_along_with_the_ones_there_are():
    with pytest.raises(KeyError) as err:
        kvant.manifest("kvant-hard")
    assert "kvant-dev" in str(err.value)


def test_a_vision_page_is_not_tier_b():
    assert "vision" in (kvant.usable(page(extraction="vision")) or "")


def test_an_empty_text_layer_is_rejected():
    """The April 2023 file is this, if it ever reaches the corpus as native."""
    reason = kvant.usable(page(body="⟦folio 12⟧\n"))
    assert reason and "characters of body" in reason


def test_mojibake_is_rejected():
    """A missing ToUnicode map turns every Cyrillic byte into a Latin one.

    So the page still has a plausible length and a plausible shape, and the
    only thing that gives it away is that none of the letters are Russian.
    """
    reason = kvant.usable(page(body="Ïðèâåò, ýòî íå ðóññêèé òåêñò. " * 20))
    assert reason and "Cyrillic" in reason


def test_a_page_of_ordinary_russian_passes():
    assert kvant.usable(page()) is None


def test_a_page_thick_with_latin_variables_still_passes():
    """The measured floor of the corpus is 0.72, which is a page like this.

    A threshold that rejected it would be rejecting the mathematics rather than
    the encoding, which is the opposite of what the gate is for.
    """
    body = RUSSIAN + " ".join(f"$abc_{{{n}}}$" for n in range(40))
    assert kvant.cyrillic_share(body) < 0.85
    assert kvant.usable(page(body=body)) is None


def test_a_page_with_no_letters_at_all_is_not_called_russian():
    """Zero over zero has to fall to the rejecting side and not the passing one."""
    assert kvant.cyrillic_share("123 456 789") == 0.0


def test_the_sheet_ordinal_is_not_the_pdf_page():
    """Off by one here renders the whole set one sheet early."""
    assert page(page_index=0).pdf_page == 1
    assert page(page_index=16).pdf_page == 17


def test_the_printed_number_and_the_sheet_ordinal_are_kept_apart():
    """They differ by the covers and the inserts, and the id uses the ordinal."""
    one = page(page_index=16, page_label="14")
    assert one.id == "kvant_2018_10/0016"
    assert one.page_label == "14"


def test_a_quota_adds_up_to_what_was_asked_for():
    """The reason `_quota` exists at all.

    Rounding each issue's share on its own gave 175 pages of a nominal 200 and
    left four issues out entirely, and the manifest header would still have
    said 200.
    """
    sizes = {f"kvant_{n}": 16 for n in range(130)}
    assert sum(kvant._quota(sizes, 200).values()) == 200


def test_a_quota_gives_the_leftover_to_the_biggest_remainders():
    sizes = {"a": 10, "b": 1, "c": 1}
    got = kvant._quota(sizes, 4)
    assert sum(got.values()) == 4
    assert got["a"] > got["b"]


def test_a_quota_of_nothing_is_nothing_rather_than_an_error():
    assert sum(kvant._quota({"a": 5}, 0).values()) == 0
    assert sum(kvant._quota({}, 200).values()) == 0


def test_a_missing_corpus_names_the_variable_to_set(monkeypatch):
    monkeypatch.delenv(kvant.CORPUS_ENV, raising=False)
    with pytest.raises(kvant.NoKvant) as err:
        kvant.root()
    assert kvant.CORPUS_ENV in str(err.value)


def test_a_directory_that_is_not_the_corpus_is_refused(tmp_path):
    with pytest.raises(kvant.NoKvant):
        kvant.root(tmp_path)


def test_a_missing_cache_names_its_own_variable(monkeypatch):
    monkeypatch.delenv(kvant.CACHE_ENV, raising=False)
    with pytest.raises(kvant.NoKvant) as err:
        kvant.cache()
    assert kvant.CACHE_ENV in str(err.value)


def test_the_scan_of_an_issue_with_no_manifest_is_none(tmp_path):
    (tmp_path / "blobs").mkdir()
    assert kvant.scan("kvant_1970_1", tmp_path) is None


def test_the_scan_is_found_through_the_digest_in_the_manifest(tmp_path):
    digest = "c9" + "8fc9ca11f2f777dc9b207566452c49d67a6ae944b33b8bce1130592fc30e25"
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "kvant_2018_10.yaml").write_text(
        "sheets:\n  - ord: 0\n    file: '0000'\npdf:\n"
        f"  url: https://example.invalid/x.pdf\n  sha256: {digest}\n  bytes: 4407970\n",
        encoding="utf-8",
    )
    blobs = tmp_path / "blobs" / digest[:2]
    blobs.mkdir(parents=True)
    (blobs / digest[2:]).write_bytes(b"%PDF-1.4\n")
    assert kvant.scan("kvant_2018_10", tmp_path) == blobs / digest[2:]


def test_a_manifest_whose_blob_is_not_cached_is_none(tmp_path):
    digest = "ab" * 32
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "kvant_2018_10.yaml").write_text(
        f"pdf:\n  sha256: {digest}\n", encoding="utf-8"
    )
    (tmp_path / "blobs").mkdir()
    assert kvant.scan("kvant_2018_10", tmp_path) is None


def test_a_sheet_digest_is_not_mistaken_for_the_pdf_digest(tmp_path):
    """The sheets carry their own hashes at four spaces and the pdf at two.

    Matching the first `sha256:` in the file would take a sheet's and look up
    a blob that is one page of the issue, which would render as a set of one
    page repeated two hundred times and would still produce a number.
    """
    sheet = "11" * 32
    pdf = "22" * 32
    (tmp_path / "pages").mkdir()
    (tmp_path / "pages" / "kvant_2018_10.yaml").write_text(
        f"sheets:\n  - ord: 0\n    file: '0000'\n    sha256: {sheet}\npdf:\n  sha256: {pdf}\n",
        encoding="utf-8",
    )
    blobs = tmp_path / "blobs" / pdf[:2]
    blobs.mkdir(parents=True)
    (blobs / pdf[2:]).write_bytes(b"%PDF-1.4\n")
    assert kvant.scan("kvant_2018_10", tmp_path) == blobs / pdf[2:]


def test_a_page_file_is_read_into_the_fields_the_draw_uses(tmp_path):
    path = tmp_path / "0016.md"
    path.write_text(
        "---\n"
        "issue: kvant_2018_10\n"
        "year: 2018\n"
        "page_index: 16\n"
        'page_label: "14"\n'
        "extraction: native\n"
        "---\n"
        "\n" + RUSSIAN,
        encoding="utf-8",
    )
    got = kvant.read_page(path)
    assert got.issue == "kvant_2018_10"
    assert got.page_index == 16
    assert got.page_label == "14"
    assert got.extraction == "native"
    assert got.body.startswith("Рассмотрим")


def test_a_page_file_with_a_word_where_the_index_goes_says_which_file(tmp_path):
    path = tmp_path / "0016.md"
    path.write_text("---\npage_index: sixteen\n---\nx\n", encoding="utf-8")
    with pytest.raises(ValueError) as err:
        kvant.read_page(path)
    assert "0016.md" in str(err.value)


SHEETS = "sheets:\n  - ord: 0\n    file: '0000'\n    sha256: {sheet}\n"


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """A scan cache holding one issue, the way `kvant-cache` holds a hundred."""
    cache = tmp_path / "cache"
    (cache / "pages").mkdir(parents=True)
    digest = "22" * 32
    (cache / "pages" / "kvant_2018_10.yaml").write_text(
        SHEETS.format(sheet="11" * 32) + f"pdf:\n  sha256: {digest}\n  bytes: 4407970\n",
        encoding="utf-8",
    )
    blob = cache / "blobs" / digest[:2]
    blob.mkdir(parents=True)
    (blob / digest[2:]).write_bytes(b"%PDF-1.4\n")
    return cache


class TestRender:
    def test_the_image_is_named_by_the_sheet_and_the_page_asked_for_is_the_next_one(
        self, store: Path, tmp_path: Path
    ) -> None:
        """The one off by one this module can make, in the one place it makes it.

        The corpus files sheet 16 as `0016.md` and the same sheet is page 17 of
        the PDF, so the image has to be named from one number and rendered from
        the other. Getting it wrong renders the whole set one sheet early and
        every page still reads, so nothing downstream would complain.
        """
        fake = Fake()
        out = tmp_path / "img"
        built = kvant.render(
            [page(page_index=16)],
            store,
            out,
            renderer=pageimages.Renderer(corpus=out, run=fake),
        )
        assert built.made == ["kvant_2018_10/0016"]
        assert (out / "kvant_2018_10" / "0016.png").exists()
        command = fake.commands[0]
        assert command[command.index("-f") + 1] == "17"

    def test_the_source_is_the_blob_the_issue_manifest_points_at(
        self, store: Path, tmp_path: Path
    ) -> None:
        fake = Fake()
        out = tmp_path / "img"
        kvant.render([page()], store, out, renderer=pageimages.Renderer(corpus=out, run=fake))
        assert fake.commands[0][-2] == str(kvant.scan("kvant_2018_10", store))

    def test_the_images_go_where_the_caller_said_and_into_neither_tree(
        self, store: Path, tmp_path: Path
    ) -> None:
        # The Bourbaki side writes into `images/` inside the corpus checkout,
        # which is gitignored there. The Kvant checkout has no such directory,
        # and a set of 200 scans landing in it is a `git add -A` away from being
        # committed.
        out = tmp_path / "somewhere else"
        kvant.render([page()], store, out, renderer=pageimages.Renderer(corpus=out, run=Fake()))
        assert (out / "kvant_2018_10" / "0016.png").exists()
        assert list(store.rglob("*.png")) == []

    def test_an_issue_with_no_cached_scan_is_reported_and_not_raised(
        self, store: Path, tmp_path: Path
    ) -> None:
        # 130 issues, and the cache is 9.8 GB. One missing is a page short, not
        # a run that dies two thirds of the way through a rasterisation.
        built = kvant.render([page(issue="kvant_1970_1")], store, tmp_path / "img")
        assert built.made == []
        assert built.failed == [("kvant_1970_1/0016", "no cached scan for kvant_1970_1")]

    def test_a_page_that_produced_no_image_is_a_failure_and_not_a_missing_file(
        self, store: Path, tmp_path: Path
    ) -> None:
        class Nothing:
            def run(self, command: list[str]) -> None:
                return None

        out = tmp_path / "img"
        built = kvant.render(
            [page()], store, out, renderer=pageimages.Renderer(corpus=out, run=Nothing())
        )
        assert built.made == []
        assert len(built.failed) == 1
        assert "0 images" in built.failed[0][1]

    def test_a_page_already_there_at_this_dpi_is_left_alone(
        self, store: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "img"
        (out / "kvant_2018_10").mkdir(parents=True)
        png(out / "kvant_2018_10" / "0016.png", 300)
        fake = Fake()
        built = kvant.render(
            [page()], store, out, dpi=300, renderer=pageimages.Renderer(corpus=out, run=fake)
        )
        assert built.had == ["kvant_2018_10/0016"]
        assert fake.commands == []

    def test_a_page_there_at_another_dpi_is_rendered_again_and_says_so(
        self, store: Path, tmp_path: Path
    ) -> None:
        # A 300 dpi page in a 600 dpi set is a reader scored on a different image
        # from the rest, which is the one thing a bake off cannot survive.
        out = tmp_path / "img"
        (out / "kvant_2018_10").mkdir(parents=True)
        png(out / "kvant_2018_10" / "0016.png", 300)
        built = kvant.render(
            [page()],
            store,
            out,
            dpi=600,
            renderer=pageimages.Renderer(corpus=out, dpi=600, run=Fake()),
        )
        assert built.made == ["kvant_2018_10/0016"]
        assert built.redone == ["kvant_2018_10/0016"]
        assert pageimages.dpi_of(out / "kvant_2018_10" / "0016.png") == 600

    def test_overwrite_renders_a_page_that_is_already_right(
        self, store: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "img"
        (out / "kvant_2018_10").mkdir(parents=True)
        png(out / "kvant_2018_10" / "0016.png", 300)
        fake = Fake()
        built = kvant.render(
            [page()],
            store,
            out,
            dpi=300,
            overwrite=True,
            renderer=pageimages.Renderer(corpus=out, run=fake),
        )
        assert built.made == ["kvant_2018_10/0016"]
        assert len(fake.commands) == 1

    def test_an_issue_is_looked_up_once_however_many_of_its_pages_are_drawn(
        self, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every lookup parses a manifest off disk, and the draw is 200 pages over
        # 127 issues, so this is small. It is here because the dict that makes it
        # true also caches the misses, which is the part worth not losing.
        looked: list[str] = []
        real = kvant.scan
        monkeypatch.setattr(
            kvant, "scan", lambda issue, s: (looked.append(issue), real(issue, s))[1]
        )
        out = tmp_path / "img"
        chosen = [page(page_index=n) for n in (16, 17, 18)]
        built = kvant.render(
            chosen, store, out, renderer=pageimages.Renderer(corpus=out, run=Fake())
        )
        assert len(built.made) == 3
        assert looked == ["kvant_2018_10"]

    def test_the_command_is_the_one_the_fleet_builds(self, store: Path, tmp_path: Path) -> None:
        # Same flags as `bourbaki render`, so a number measured off these images
        # is a number about the reader and not about the rasteriser.
        fake = Fake()
        out = tmp_path / "img"
        kvant.render(
            [page()],
            store,
            out,
            dpi=400,
            renderer=pageimages.Renderer(corpus=out, dpi=400, run=fake),
        )
        command = fake.commands[0]
        assert command[:5] == ["pdftoppm", "-png", "-r", "400", "-gray"]

    def test_nothing_of_the_working_name_is_left_behind(self, store: Path, tmp_path: Path) -> None:
        out = tmp_path / "img"
        kvant.render([page()], store, out, renderer=pageimages.Renderer(corpus=out, run=Fake()))
        assert [p.name for p in (out / "kvant_2018_10").iterdir()] == ["0016.png"]


corpus_needed = pytest.mark.skipif(
    not kvant.available(), reason="KVANT_CORPUS is not set on this machine"
)


@corpus_needed
def test_every_recorded_page_is_still_in_the_corpus():
    """A set of 200 that silently loads 198 is what makes a benchmark flatter."""
    here = {one.id for one in kvant.pages(kvant.root())}
    for name in kvant.SETS:
        gone = [page_id for page_id in kvant.read_manifest(name) if page_id not in here]
        assert gone == [], name


@corpus_needed
def test_the_gate_passes_everything_the_upstream_routing_sent_it():
    """Measured, and the number is the point.

    2 063 native pages, none rejected. That is the evidence that the four
    mojibake files and the empty April 2023 one never reach the native lane, so
    the exclusion the milestone asks for is already happening upstream and this
    gate is guarding it rather than performing it. If this ever fails, the
    routing changed and the set got quietly smaller.
    """
    native = [one for one in kvant.pages(kvant.root()) if one.extraction == "native"]
    assert len(native) > 2_000
    refused = {one.id: kvant.usable(one) for one in native if kvant.usable(one)}
    assert refused == {}
