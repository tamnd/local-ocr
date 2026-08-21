"""The six serving benchmarks.

Nothing here starts a server or touches a card. What is worth pinning is the
arithmetic and the honesty: that the headline is accepted pages an hour and not
throughput, that an arm which failed to run is named rather than dropped, and
that a question every arm failed has no winner instead of the least broken one.

The one number this file will not let go of is `accepted_per_hour`. Every
decision in M4 comes off it, and every way it could quietly become a number
about something else has a test here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_ocr import bench


def variant(
    name: str = "one", questions: tuple[str, ...] = (bench.ENGINE,), **over
) -> bench.Variant:
    return bench.Variant(name=name, questions=questions, **over)


def measured(
    name: str = "one",
    questions: tuple[str, ...] = (bench.ENGINE,),
    *,
    pages: int = 40,
    read: int = 40,
    accepted: int = 40,
    seconds: float = 360.0,
    joules: float | None = None,
    note: str = "",
) -> bench.Measured:
    return bench.Measured(
        variant=variant(name, questions),
        pages=pages,
        read=read,
        accepted=accepted,
        seconds=seconds,
        joules=joules,
        note=note,
    )


def board(*ms: bench.Measured) -> bench.Board:
    return bench.Board(model="reader-a", set_name="golden-dev", measured=list(ms))


# ---------------------------------------------------------------------------
# The plan


def test_the_plan_covers_every_one_of_the_six():
    answered = {q for one in bench.PLAN for q in one.questions}
    assert answered == set(bench.QUESTIONS)


def test_the_shipping_configuration_is_measured_once_and_answers_five():
    """Ten runs for thirteen arms.

    Not only a saving. Measuring the same configuration five times and printing
    five slightly different numbers for it invites the spread between them to be
    read as a difference between the questions.
    """
    assert len(bench.BASELINE.questions) == 5
    assert sum(1 for one in bench.PLAN if one is bench.BASELINE) == 1
    assert len(bench.PLAN) == 10


def test_the_concurrency_sweep_keeps_its_own_four_arms():
    """Folding the baseline in as the 16 would make the one row that was
    measured on a different afternoon the row the sweep is read off.
    """
    swept = [one for one in bench.PLAN if bench.CONCURRENCY in one.questions]
    assert [one.serve_extra[-1] for one in swept] == ["4", "8", "16", "32"]
    assert bench.CONCURRENCY not in bench.BASELINE.questions


def test_the_client_concurrency_follows_the_server_queue_depth():
    """A sweep of the server's queue depth run at one request in flight measures
    nothing, because the server is never given anything to batch, and every arm
    comes back the same.
    """
    for one in bench.PLAN:
        if bench.CONCURRENCY in one.questions:
            assert str(one.jobs) == one.serve_extra[-1]


def test_the_arm_that_can_fail_to_run_is_scheduled_early():
    """Finding out SGLang will not start after eight other runs have taken the
    card is finding out too late to do anything else with the window.
    """
    names = [one.name for one in bench.PLAN]
    assert names.index("sglang") == 1


def test_an_arm_that_answers_nothing_is_refused():
    with pytest.raises(ValueError):
        bench.Variant(name="stray", questions=())


def test_an_arm_that_answers_a_question_nobody_asked_is_refused():
    with pytest.raises(ValueError) as err:
        bench.Variant(name="stray", questions=("latency",))
    assert "latency" in str(err.value)


def test_the_plan_can_be_printed_without_taking_the_card():
    """Every arm takes the whole GPU for the length of a batch, so the window
    has to be asked for before it is taken.
    """
    lines = bench.plan_lines().splitlines()
    assert len(lines) == len(bench.PLAN) + 2
    assert "sglang" in bench.plan_lines()


# ---------------------------------------------------------------------------
# The arithmetic


def test_the_headline_is_accepted_pages_and_not_pages_read():
    """The whole reason this file does not report tokens a second.

    Two arms that read the same 40 pages in the same six minutes, one of which
    produced 20 pages somebody has to read again, are not the same arm.
    """
    whole = measured(read=40, accepted=40, seconds=360)
    half = measured(read=40, accepted=20, seconds=360)
    assert whole.read_per_hour == half.read_per_hour
    assert whole.accepted_per_hour == 2 * half.accepted_per_hour


def test_accepted_pages_an_hour_is_what_it_says():
    assert measured(accepted=40, seconds=360).accepted_per_hour == 400.0


def test_acceptance_is_over_the_pages_given_and_not_the_pages_returned():
    """A refused page is a page the setting cost somebody, so it stays in the
    denominator. Scoring acceptance over what came back would let an arm that
    refuses everything it would have failed report 100 per cent.
    """
    assert measured(pages=40, read=20, accepted=20).acceptance == 0.5


def test_a_run_of_no_time_is_zero_rather_than_a_division_by_zero():
    assert measured(seconds=0).accepted_per_hour == 0.0
    assert measured(seconds=0).read_per_hour == 0.0
    assert measured(read=0).seconds_per_page == 0.0
    assert measured(pages=0).acceptance == 0.0


def test_energy_is_charged_per_accepted_page():
    """Energy spent on a page that comes back refused has to be spent again, so
    an arm that is efficient per request and refuses a fifth of them is not
    efficient.
    """
    whole = measured(accepted=40, joules=360_000.0)
    half = measured(accepted=20, joules=360_000.0)
    assert half.wh_per_accepted == pytest.approx(2 * whole.wh_per_accepted)


def test_the_mean_watts_come_off_the_energy_and_the_clock():
    assert measured(seconds=360, joules=90_000.0).watts == pytest.approx(250.0)


def test_an_unmetered_run_says_nothing_about_power_rather_than_zero():
    one = measured(joules=None)
    assert one.watts is None
    assert one.wh_per_accepted is None


def test_an_arm_that_accepted_nothing_has_no_energy_per_page():
    """Dividing by zero accepted pages would print an infinity, and an arm that
    produced nothing usable has no cost per usable page to report.
    """
    assert measured(accepted=0, joules=90_000.0).wh_per_accepted is None


# ---------------------------------------------------------------------------
# Sampling the board


class Fixed:
    def __init__(self, *readings: float | None) -> None:
        self.readings = list(readings)
        self.asked = 0

    def __call__(self) -> float | None:
        self.asked += 1
        return self.readings[min(self.asked - 1, len(self.readings) - 1)]


def test_one_sample_is_not_an_energy_figure():
    """A run with one reading has an instantaneous number and no duration, and
    turning that into joules would be inventing the part that matters.
    """
    meter = bench.Power(Fixed(300.0), every=0.01)
    with meter:
        pass
    if meter.samples < 2:
        assert meter.joules is None


def test_samples_integrate_to_watt_seconds():
    meter = bench.Power(Fixed(300.0), every=0.01)
    with meter:
        while meter.samples < 6:
            pass
    assert meter.joules is not None
    assert meter.joules > 0


def test_a_sample_that_failed_is_dropped_and_not_counted_as_an_idle_card():
    """The card is behind an ssh hop into WSL. Counting a timed out sample as
    zero watts would report the arm as more efficient the worse the link was.
    """
    meter = bench.Power(Fixed(None), every=0.01)
    with meter:
        while meter._sample.asked < 4:  # type: ignore[attr-defined]
            pass
    assert meter.samples == 0
    assert meter.joules is None


def test_a_sampler_that_raises_does_not_take_the_run_with_it():
    def angry() -> float | None:
        raise RuntimeError("ssh went away")

    meter = bench.Power(angry, every=0.01)
    with meter:
        pass
    assert meter.joules is None


def test_the_nvidia_reading_is_parsed_off_the_first_line():
    sample = bench.nvidia_smi(lambda _cmd: "269.87\n")
    assert sample() == pytest.approx(269.87)


def test_a_card_that_answered_with_words_is_no_reading_rather_than_zero():
    sample = bench.nvidia_smi(lambda _cmd: "[N/A]\n")
    assert sample() is None


def test_a_card_that_could_not_be_reached_is_no_reading():
    def dead(_cmd):
        raise OSError("no route")

    assert bench.nvidia_smi(dead)() is None


# ---------------------------------------------------------------------------
# Measuring one arm


def test_the_clock_excludes_the_model_load_and_the_judging():
    """Thirty to ninety seconds of weights coming off disk belongs to the model
    and not to the setting under test, and the judging happens on the laptop and
    would put the laptop inside a number about the card.
    """
    ticks = iter([100.0, 110.0, 130.0, 400.0])
    tick = lambda: next(ticks)  # noqa: E731
    got = bench.measure(
        variant(),
        # Both of these take the clock forward, the way a 90 second model load
        # and a judging pass over 40 pages really do.
        stage=lambda _v: tick(),
        read=lambda _v: (40, 40),
        accept=lambda _v: tick() and 40,
        pages=40,
        clock=tick,
    )
    assert got.seconds == 20.0
    assert got.accepted == 40


def test_an_arm_that_would_not_start_is_a_row_and_not_an_exception():
    """The one arm expected to fail is SGLang, and losing the eight behind it to
    that would waste the window.
    """

    def refuse(_v):
        raise RuntimeError("no sglang on this machine")

    got = bench.measure(
        variant("sglang"),
        stage=refuse,
        read=lambda _v: (40, 40),
        accept=lambda _v: 40,
        pages=40,
    )
    assert got.skipped
    assert "no sglang" in got.note
    assert got.accepted == 0


def test_an_arm_that_died_mid_batch_keeps_the_clock_it_had():
    def die(_v):
        raise RuntimeError("the server went away")

    got = bench.measure(
        variant(),
        stage=lambda _v: None,
        read=die,
        accept=lambda _v: 40,
        pages=40,
        clock=iter([0.0, 5.0]).__next__,
    )
    assert got.skipped
    assert got.seconds == 5.0


def test_the_meter_runs_only_while_the_batch_does():
    seen: list[str] = []

    class Watching(bench.Power):
        def __enter__(self):
            seen.append("on")
            return self

        def __exit__(self, *_):
            seen.append("off")

    def read(_v):
        seen.append("reading")
        return (40, 40)

    bench.measure(
        variant(),
        stage=lambda _v: seen.append("stage"),
        read=read,
        accept=lambda _v: seen.append("judge") or 40,
        pages=40,
        power=lambda _v: Watching(lambda: 300.0),
    )
    assert seen == ["stage", "on", "reading", "off", "judge"]


def test_a_whole_run_says_what_each_arm_did_as_it_goes():
    said: list[str] = []
    bench.run(
        [variant("a"), variant("b")],
        stage=lambda _v: None,
        read=lambda _v: (40, 40),
        accept=lambda _v: 36,
        pages=40,
        say=said.append,
    )
    assert any("a: staging" in line for line in said)
    assert any("36 of 40 accepted" in line for line in said)


# ---------------------------------------------------------------------------
# The board


def test_the_arms_of_a_question_come_back_best_first():
    slow = measured("slow", accepted=20)
    fast = measured("fast", accepted=40)
    assert [m.variant.name for m in board(slow, fast).arms(bench.ENGINE)] == ["fast", "slow"]


def test_an_arm_that_did_not_run_sorts_last_however_fast_it_looks():
    dead = measured("dead", accepted=0, seconds=0.1, note="would not start")
    live = measured("live", accepted=40)
    assert [m.variant.name for m in board(dead, live).arms(bench.ENGINE)] == ["live", "dead"]


def test_an_arm_only_appears_under_the_questions_it_answers():
    one = measured("one", questions=(bench.ENGINE, bench.POWER))
    assert len(board(one).arms(bench.ENGINE)) == 1
    assert board(one).arms(bench.CONCURRENCY) == []


def test_a_question_where_nothing_ran_has_no_winner():
    """None and not the least broken arm. Printing a winner for a question
    nobody measured is how a benchmark comes to recommend a configuration that
    was never run.
    """
    dead = measured("dead", accepted=0, note="would not start")
    assert board(dead).winner(bench.ENGINE) is None


def test_an_arm_that_ran_and_accepted_nothing_does_not_win():
    empty = measured("empty", read=40, accepted=0)
    assert board(empty).winner(bench.ENGINE) is None


def test_the_margin_is_over_the_runner_up():
    assert board(measured("a", accepted=44), measured("b", accepted=40)).margin(
        bench.ENGINE
    ) == pytest.approx(0.1)


def test_a_question_with_one_arm_has_no_margin():
    assert board(measured("a")).margin(bench.ENGINE) is None


# ---------------------------------------------------------------------------
# What gets written down


def test_a_close_result_is_reported_as_close_rather_than_as_a_decision():
    """Most of these are expected to come out near each other, and a 2 per cent
    difference measured once on 40 pages is not a reason to change anything.
    """
    text = board(measured("a", accepted=41), measured("b", accepted=40)).to_markdown()
    assert "not a reason to change anything" in text


def test_a_real_difference_is_reported_as_a_win():
    text = board(measured("a", accepted=40), measured("b", accepted=20)).to_markdown()
    assert "wins by 100.0%" in text


def test_the_markdown_says_why_tokens_a_second_is_not_here():
    text = board(measured()).to_markdown()
    assert "Tokens a second is not reported" in text


def test_an_arm_that_did_not_run_is_named_in_the_markdown():
    """An arm missing from a table reads as an arm that lost."""
    text = board(measured("sglang", accepted=0, note="would not start")).to_markdown()
    assert "Arms that did not run" in text
    assert "sglang: would not start" in text


def test_a_question_with_no_arms_gets_no_empty_section():
    text = board(measured("a", questions=(bench.ENGINE,))).to_markdown()
    assert bench.ASKED[bench.CONCURRENCY] not in text


def test_the_json_carries_the_flags_that_produced_each_number(tmp_path: Path):
    """A row that says 400 accepted pages an hour and not what was appended to
    the command line is a row nobody can reproduce or argue with.
    """
    one = bench.Measured(
        variant=bench.Variant("kv-fp8", (bench.KV,), serve_extra=("--kv-cache-dtype", "fp8")),
        pages=40,
        read=40,
        accepted=38,
        seconds=360.0,
        joules=90_000.0,
    )
    bench.write(board(one), json_path=tmp_path / "b.json", markdown_path=None)
    got = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    arm = got["arms"][0]
    assert arm["serve_extra"] == ["--kv-cache-dtype", "fp8"]
    assert arm["accepted_pages_per_hour"] == 380.0
    assert got["answers"][bench.KV]["winner"] == "kv-fp8"


def test_the_json_says_a_question_is_unanswered_rather_than_leaving_it_out(tmp_path: Path):
    bench.write(board(measured()), json_path=tmp_path / "b.json", markdown_path=None)
    got = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert set(got["answers"]) == set(bench.QUESTIONS)
    assert got["answers"][bench.CONCURRENCY] is None


def test_writing_makes_the_directory(tmp_path: Path):
    bench.write(board(measured()), json_path=None, markdown_path=tmp_path / "deep" / "b.md")
    assert (tmp_path / "deep" / "b.md").is_file()
