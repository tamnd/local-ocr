"""`local-ocr pages`, without a corpus and without pdftoppm."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_ocr import cli, pageimages


@pytest.fixture
def nothing_to_do(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(pageimages, "read_ids", lambda name: ["alg-viii/0042"])
    monkeypatch.setattr(cli.corpuslib, "root", lambda path=None: tmp_path)
    return tmp_path


def test_a_set_that_renders_completely_reports_nothing_missing(
    nothing_to_do: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        pageimages, "build", lambda *a, **k: pageimages.Built(made=["alg-viii/0042"])
    )
    assert cli.pages_cmd([]) == 0
    assert "1 rendered" in capsys.readouterr().out


def test_a_page_that_could_not_be_rendered_is_named_and_fails(
    nothing_to_do: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Exit status matters: this runs before a bake off, and a set that is short
    # a volume scores every model on a different corpus than it says.
    monkeypatch.setattr(
        pageimages,
        "build",
        lambda *a, **k: pageimages.Built(failed=[("nopdf/0001", "no pdf for nopdf")]),
    )
    assert cli.pages_cmd([]) == 1
    assert "nopdf/0001" in capsys.readouterr().err


def test_no_corpus_is_a_usage_error_rather_than_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("BOURBAKI_CORPUS", raising=False)
    assert cli.pages_cmd([]) == 2
    assert "BOURBAKI_CORPUS" in capsys.readouterr().err


def test_the_dpi_reaches_the_renderer(nothing_to_do: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def build(ids, corpus, dpi=0, overwrite=False, renderer=None):
        seen["dpi"] = dpi
        seen["overwrite"] = overwrite
        return pageimages.Built()

    monkeypatch.setattr(pageimages, "build", build)
    assert cli.pages_cmd(["--dpi", "600", "--overwrite"]) == 0
    assert seen == {"dpi": 600, "overwrite": True}


def test_the_default_set_is_the_development_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # golden-test is held back for a milestone, and rendering it is the first
    # step of reading it.
    asked: list[str] = []
    monkeypatch.setattr(pageimages, "read_ids", lambda name: asked.append(name) or [])
    monkeypatch.setattr(cli.corpuslib, "root", lambda path=None: tmp_path)
    monkeypatch.setattr(pageimages, "build", lambda *a, **k: pageimages.Built())
    cli.pages_cmd([])
    assert asked == ["golden-dev"]


def test_the_usage_mentions_pages() -> None:
    assert "pages" in cli._usage()
