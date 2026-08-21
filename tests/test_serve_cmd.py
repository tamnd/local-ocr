"""`local-ocr serve`, without starting anything.

The exec is injected, so these run on a laptop and still cover the part that can
be wrong: what would have been executed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_ocr import cli, serving


class Exec:
    """Stands in for os.execvp, which does not return and cannot be tested."""

    def __init__(self) -> None:
        self.command: list[str] | None = None

    def __call__(self, binary: str, argv: list[str]) -> None:
        self.command = argv


def test_serving_a_reader_hands_the_process_to_vllm() -> None:
    ran = Exec()
    assert cli.serve_cmd(["reader-a"], exec_=ran) == 1
    assert ran.command is not None
    assert ran.command[:3] == ["vllm", "serve", "allenai/olmOCR-2-7B-1025-FP8"]


def test_print_starts_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    ran = Exec()
    assert cli.serve_cmd(["reader-a", "--print"], exec_=ran) == 0
    assert ran.command is None
    assert "--served-model-name reader-a" in capsys.readouterr().out


def test_a_name_that_is_not_in_the_shortlist_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ran = Exec()
    assert cli.serve_cmd(["reader-q"], exec_=ran) == 2
    assert ran.command is None
    assert "reader-a" in capsys.readouterr().err


def test_no_name_at_all_is_refused_rather_than_defaulted() -> None:
    # There is a default reader, but starting it because somebody typed half a
    # command is how the wrong weights end up reading a book.
    ran = Exec()
    assert cli.serve_cmd([], exec_=ran) == 2
    assert ran.command is None


def test_an_unpinned_reader_starts_and_says_it_is_unpinned(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Every entry in the shipped file is pinned, which is the point of the file,
    # so the unpinned case needs a shortlist written for it.
    shortlist = tmp_path / "models.toml"
    shortlist.write_text('[reader-x]\nrepo = "somewhere/something"\n', encoding="utf-8")
    monkeypatch.setattr(serving, "MODELS", shortlist)
    ran = Exec()
    assert cli.serve_cmd(["reader-x"], exec_=ran) == 1
    assert ran.command is not None
    assert "cannot be reproduced" in capsys.readouterr().err


def test_a_pinned_reader_says_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    cli.serve_cmd(["reader-a"], exec_=Exec())
    assert capsys.readouterr().err == ""


def test_the_list_names_every_reader(capsys: pytest.CaptureFixture[str]) -> None:
    ran = Exec()
    assert cli.serve_cmd(["--list"], exec_=ran) == 0
    out = capsys.readouterr().out
    for name in ("reader-a", "reader-b", "reader-c", "reader-d", "reader-e"):
        assert name in out
    assert ran.command is None


def test_a_sweep_flag_is_appended_after_the_entrys_own(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # vLLM takes the last occurrence of a repeated option, so appending is what
    # makes --max-num-seqs 32 win over the 16 in the entry.
    ran = Exec()
    cli.serve_cmd(["reader-a", "--extra", "--max-num-seqs 32"], exec_=ran)
    assert ran.command is not None
    assert ran.command[-2:] == ["--max-num-seqs", "32"]
    assert ran.command.count("--max-num-seqs") == 2


def test_a_sweep_says_out_loud_that_it_is_not_the_shortlist(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.serve_cmd(["reader-a", "--extra", "--max-num-seqs 32"], exec_=Exec())
    assert "--max-num-seqs 32" in capsys.readouterr().err


def test_the_usage_mentions_serve() -> None:
    assert "serve" in cli._usage()
