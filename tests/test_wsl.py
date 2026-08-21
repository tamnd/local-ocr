"""What `local-ocr serve` puts in the environment on a box behind Windows.

None of this can be tested by starting a server, because the laptop these run
on is not WSL and the box that is has one card and a queue on it. So the two
inputs are injected: what uname says, and what the environment already holds.
"""

from __future__ import annotations

import pytest

from local_ocr import cli, wsl


def uname(release: str):
    """A stand in for platform.uname(), which returns a tuple of five fields."""
    return lambda: ("Linux", "gamingpc", release, "#1 SMP", "x86_64")


def test_a_wsl_kernel_is_recognised_by_the_string_vllm_looks_for() -> None:
    assert wsl.in_wsl(uname=uname("6.6.87.2-microsoft-standard-WSL2"))


def test_the_test_is_case_folded_the_way_vllms_own_is() -> None:
    assert wsl.in_wsl(uname=uname("6.6.87.2-Microsoft-standard-WSL2"))


def test_an_ordinary_linux_kernel_is_not_wsl() -> None:
    assert not wsl.in_wsl(uname=uname("6.8.0-45-generic"))


def test_a_mac_is_not_wsl() -> None:
    assert not wsl.in_wsl(uname=lambda: ("Darwin", "laptop", "24.6.0", "", "arm64"))


def test_wsl_turns_the_v2_model_runner_off() -> None:
    added, notes = wsl.additions({}, wsl=True)
    assert added == {wsl.RUNNER: "0"}
    assert len(notes) == 1


def test_the_note_says_why_and_not_just_what() -> None:
    # The value of this whole module is the explanation. A line that says
    # VLLM_USE_V2_MODEL_RUNNER=0 and nothing else is a line somebody deletes.
    _added, notes = wsl.additions({}, wsl=True)
    said = notes[0]
    assert "pin" in said
    assert "V1" in said


def test_nothing_is_added_anywhere_else() -> None:
    added, notes = wsl.additions({}, wsl=False)
    assert added == {}
    assert notes == []


def test_a_setting_already_in_the_environment_is_left_alone() -> None:
    # Somebody sweeping V1 against V2 typed this and meant it.
    added, notes = wsl.additions({wsl.RUNNER: "1"}, wsl=True)
    assert added == {}
    assert "leaves it alone" in notes[0]


def test_that_holds_even_when_they_typed_the_same_value() -> None:
    added, _notes = wsl.additions({wsl.RUNNER: "0"}, wsl=True)
    assert added == {}


class Exec:
    def __init__(self) -> None:
        self.command: list[str] | None = None

    def __call__(self, binary: str, argv: list[str]) -> None:
        self.command = argv


def test_serve_says_out_loud_that_it_moved_the_runner(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wsl, "in_wsl", lambda **_kw: True)
    monkeypatch.delenv(wsl.RUNNER, raising=False)
    cli.serve_cmd(["reader-a"], exec_=Exec())
    assert wsl.RUNNER in capsys.readouterr().err


def test_serve_puts_it_in_the_environment_the_exec_will_inherit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # os.execvp carries os.environ across, so setting it here is what the
    # server actually starts with.
    monkeypatch.setattr(wsl, "in_wsl", lambda **_kw: True)
    monkeypatch.delenv(wsl.RUNNER, raising=False)
    cli.serve_cmd(["reader-a"], exec_=Exec())
    import os

    assert os.environ[wsl.RUNNER] == "0"


def test_print_shows_the_assignment_in_front_of_the_command(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wsl, "in_wsl", lambda **_kw: True)
    monkeypatch.delenv(wsl.RUNNER, raising=False)
    cli.serve_cmd(["reader-a", "--print"], exec_=Exec())
    out = capsys.readouterr().out
    assert out.startswith(f"{wsl.RUNNER}=0 vllm serve ")


def test_print_off_wsl_is_the_command_and_nothing_else(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(wsl, "in_wsl", lambda **_kw: False)
    cli.serve_cmd(["reader-a", "--print"], exec_=Exec())
    assert capsys.readouterr().out.startswith("vllm serve ")
