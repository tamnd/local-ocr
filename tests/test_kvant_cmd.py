"""`local-ocr kvant`, without the Russian corpus and without the scan cache.

The command is the only place the Kvant tree and the Bourbaki tree can be
confused for one another, and it is the place that decides whether the held out
set gets read. Both are worth pinning on a machine that has neither tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_ocr import cli, kvant, pageimages
from local_ocr.golden import Purpose


@pytest.fixture
def trees(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A corpus and a cache that exist, so the command gets past its own guard."""
    monkeypatch.setattr(kvant, "root", lambda path=None: tmp_path / "corpus")
    monkeypatch.setattr(kvant, "cache", lambda path=None: tmp_path / "cache")
    return tmp_path


class TestShow:
    def test_with_no_name_it_lists_the_sets_and_says_which_is_held_out(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.kvant_cmd(["show"]) == 0
        out = capsys.readouterr().out
        assert "kvant-dev: tier B, 200 pages" in out
        assert "kvant-test: tier B, 200 pages, held out" in out

    def test_with_a_name_it_prints_the_page_ids(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert cli.kvant_cmd(["show", "--name", "kvant-dev"]) == 0
        lines = capsys.readouterr().out.split("\n")
        assert len(lines) == 201
        assert lines[0].startswith("kvant_")

    def test_it_needs_neither_tree(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The manifests ship in the package, so somebody who wants to see what is
        # in a set should not have to have 9.8 GB of scans on the machine.
        monkeypatch.delenv(kvant.CORPUS_ENV, raising=False)
        monkeypatch.delenv(kvant.CACHE_ENV, raising=False)
        assert cli.kvant_cmd(["show"]) == 0


class TestMissingTrees:
    def test_no_corpus_is_a_usage_error_naming_its_own_variable(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # KVANT_CORPUS and not BOURBAKI_CORPUS. Pointing this command at the
        # Bourbaki tree draws a set of nothing and says so nowhere.
        monkeypatch.delenv(kvant.CORPUS_ENV, raising=False)
        assert cli.kvant_cmd(["draw"]) == 2
        assert kvant.CORPUS_ENV in capsys.readouterr().err

    def test_no_cache_is_a_usage_error_too(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(kvant, "root", lambda path=None: tmp_path)
        monkeypatch.delenv(kvant.CACHE_ENV, raising=False)
        assert cli.kvant_cmd(["pages", "--out", str(tmp_path / "img")]) == 2
        assert kvant.CACHE_ENV in capsys.readouterr().err


class TestDraw:
    def test_the_gate_yield_is_printed_even_when_it_is_zero(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The line that is the evidence the gate ran at all.

        On today's corpus the gate rejects nothing, because the mojibake files
        never reach the native lane. Printing that zero is the difference between
        a check that passed and a check somebody deleted.
        """
        monkeypatch.setattr(kvant, "draw", lambda c, s: kvant.Draw(dev=[], test=[], rejected={}))
        monkeypatch.setattr(kvant, "write_manifests", lambda drawn: [])
        assert cli.kvant_cmd(["draw"]) == 0
        assert "gate rejected 0 native pages" in capsys.readouterr().out

    def test_a_rejected_page_is_named_with_the_reason(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rejected = {"kvant_2023_4/0002": "0.01 Cyrillic of the letters"}
        monkeypatch.setattr(
            kvant, "draw", lambda c, s: kvant.Draw(dev=[], test=[], rejected=rejected)
        )
        monkeypatch.setattr(kvant, "write_manifests", lambda drawn: [])
        cli.kvant_cmd(["draw"])
        out = capsys.readouterr().out
        assert "kvant_2023_4/0002" in out
        assert "Cyrillic" in out


class TestCheck:
    def test_a_steady_set_says_so(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        drift = kvant.Drift(name="kvant-dev", recorded=200, would_draw=200, gone=[], arrived=[])
        monkeypatch.setattr(kvant, "check", lambda c, s: [drift])
        assert cli.kvant_cmd(["check"]) == 0
        assert "200 pages, unchanged" in capsys.readouterr().out


class TestPages:
    def test_without_an_output_directory_it_refuses_rather_than_guesses(
        self, trees: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # There is no right default. The Bourbaki side writes into the corpus
        # checkout because `images/` is gitignored there, and the Kvant checkout
        # has no such directory, so a guess would put 200 scans somewhere a
        # `git add -A` would commit them.
        assert cli.kvant_cmd(["pages"]) == 2
        assert "--out" in capsys.readouterr().err

    def test_it_renders_the_dev_set_by_default_and_says_what_it_did(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: dict[str, object] = {}

        def load(name, *, purpose, corpus=None):
            seen["name"] = name
            seen["purpose"] = purpose
            return []

        monkeypatch.setattr(kvant, "load", load)
        monkeypatch.setattr(
            kvant,
            "render",
            lambda *a, **k: pageimages.Built(made=["kvant_2018_10/0016"]),
        )
        assert cli.kvant_cmd(["pages", "--out", str(trees / "img")]) == 0
        assert seen["name"] == "kvant-dev"
        assert "1 rendered, 0 already there, 0 failed" in capsys.readouterr().out

    def test_rendering_the_held_out_set_is_not_reading_it(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`kvant-test` can be rasterised without burning it.

        Turning a page into an image looks at nothing and decides nothing, so
        the purpose is MILESTONE and the guard lets it through. It is stated
        rather than defaulted because the next caller would inherit the default
        without ever choosing it.
        """
        seen: dict[str, object] = {}

        def load(name, *, purpose, corpus=None):
            seen["purpose"] = purpose
            return []

        monkeypatch.setattr(kvant, "load", load)
        monkeypatch.setattr(kvant, "render", lambda *a, **k: pageimages.Built())
        assert cli.kvant_cmd(["pages", "--name", "kvant-test", "--out", str(trees / "img")]) == 0
        assert seen["purpose"] is Purpose.MILESTONE

    def test_a_page_that_could_not_be_rendered_is_named_and_fails(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Exit status matters: a set that is short an issue scores every reader
        # on a different set of pages from the one the manifest names.
        monkeypatch.setattr(kvant, "load", lambda name, *, purpose, corpus=None: [])
        monkeypatch.setattr(
            kvant,
            "render",
            lambda *a, **k: pageimages.Built(
                failed=[("kvant_1970_1/0016", "no cached scan for kvant_1970_1")]
            ),
        )
        assert cli.kvant_cmd(["pages", "--out", str(trees / "img")]) == 1
        assert "kvant_1970_1/0016" in capsys.readouterr().err

    def test_a_page_replaced_at_another_dpi_is_reported_as_such(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(kvant, "load", lambda name, *, purpose, corpus=None: [])
        monkeypatch.setattr(
            kvant,
            "render",
            lambda *a, **k: pageimages.Built(made=["a/0001"], redone=["a/0001"]),
        )
        assert cli.kvant_cmd(["pages", "--out", str(trees / "img")]) == 0
        assert "1 of them replacing an image at another dpi" in capsys.readouterr().out

    def test_the_dpi_and_the_overwrite_reach_the_renderer(
        self, trees: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, object] = {}

        def render(chosen, store, out, dpi=0, overwrite=False, renderer=None):
            seen["dpi"] = dpi
            seen["overwrite"] = overwrite
            seen["out"] = out
            return pageimages.Built()

        monkeypatch.setattr(kvant, "load", lambda name, *, purpose, corpus=None: [])
        monkeypatch.setattr(kvant, "render", render)
        argv = ["pages", "--out", str(trees / "img"), "--dpi", "600", "--overwrite"]
        assert cli.kvant_cmd(argv) == 0
        assert seen["dpi"] == 600
        assert seen["overwrite"] is True
        assert seen["out"] == trees / "img"


class TestReadingsLabels:
    """`--readings`, which is now repeatable and optionally labelled."""

    def test_one_bare_directory_is_labelled_with_the_model_flag(self) -> None:
        assert cli._readings(["/tmp/out"], "reader-a") == [("reader-a", Path("/tmp/out"))]

    def test_several_bare_directories_take_their_own_names(self) -> None:
        # --model cannot name four of them, and a table whose rows are all
        # called reader is worse than no table.
        assert cli._readings(["/tmp/reader-a", "/tmp/reader-c"], "reader") == [
            ("reader-a", Path("/tmp/reader-a")),
            ("reader-c", Path("/tmp/reader-c")),
        ]

    def test_an_explicit_label_wins_over_both(self) -> None:
        assert cli._readings(["c32=/tmp/out"], "reader-a") == [("c32", Path("/tmp/out"))]

    def test_labelled_and_bare_can_be_mixed(self) -> None:
        got = cli._readings(["reader-a=/tmp/one", "/tmp/reader-c"], "reader")
        assert got == [("reader-a", Path("/tmp/one")), ("reader-c", Path("/tmp/reader-c"))]

    def test_nothing_given_is_nothing_back(self) -> None:
        assert cli._readings([], "reader-a") == []


class TestRotatedComparison:
    """`kvant rotated` with more than one reader, which needs neither tree."""

    def make(self, root: Path, name: str, ids: list[str]) -> Path:
        from local_ocr import rotated

        where = root / name
        for page_id in ids:
            issue, _, number = page_id.partition("/")
            (where / issue).mkdir(parents=True, exist_ok=True)
            runs = " ".join(run.text for run in rotated.reference()[page_id])
            (where / issue / f"{number}.md").write_text(f"текст {runs}", encoding="utf-8")
        return where

    def test_two_readers_print_one_ranked_table_and_not_two_cards(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        from local_ocr import rotated

        ids = sorted(rotated.reference())[:3]
        good = self.make(tmp_path, "good", ids)
        poor = tmp_path / "poor"
        poor.mkdir()
        code = cli.kvant_cmd(
            ["rotated", "--readings", f"good={good}", "--readings", f"poor={poor}"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert out.startswith("| Reader |")
        assert "# good on kvant-rotated" not in out
        assert out.strip().split("\n")[2].startswith("| good |")

    def test_shared_cuts_the_set_to_the_pages_both_produced(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        from local_ocr import rotated

        ids = sorted(rotated.reference())[:4]
        both = self.make(tmp_path, "both", ids)
        half = self.make(tmp_path, "half", ids[:2])
        code = cli.kvant_cmd(
            ["rotated", "--shared", "--readings", f"both={both}", "--readings", f"half={half}"]
        )
        got = capsys.readouterr()
        assert code == 0
        assert "2 pages every reader produced, out of 63" in got.err
        # Both caught every run on the two shared pages, so the set is 4 runs
        # and neither reader is charged for a page it never answered.
        assert "| 4 of 4 |" in got.out

    def test_one_reader_still_prints_the_card_it_always_did(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        from local_ocr import rotated

        where = self.make(tmp_path, "solo", sorted(rotated.reference())[:1])
        assert cli.kvant_cmd(["rotated", "--readings", str(where), "--model", "solo"]) == 0
        assert capsys.readouterr().out.startswith("# solo on kvant-rotated")

    def test_no_readings_at_all_is_still_a_usage_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert cli.kvant_cmd(["rotated"]) == 2
        assert "--draw" in capsys.readouterr().err
