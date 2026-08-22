"""Whether the reward a trainer would compute has anything in it to learn from.

Nothing here touches a card either, and for the same reason `test_finetune`
does not: the decision this feeds is whether to book one at all.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_ocr import rlvr
from local_ocr.rules.validate import Expect, Problem, Rule, validate


def job(target: str, *reasons: str | None) -> dict:
    """A queue job whose answers are the reasons given, None meaning accepted."""
    return {
        "target": target,
        "history": [
            {"ts": "2026-08-22T00:00:00Z", "host": "gpc", "ok": r is None}
            | ({} if r is None else {"reason": r})
            for r in reasons
        ],
    }


def queue(tmp_path: Path, *jobs: dict, state: str = "done") -> Path:
    root = tmp_path / "ocr"
    (root / state).mkdir(parents=True, exist_ok=True)
    for index, one in enumerate(jobs):
        (root / state / f"{index:04d}.json").write_text(json.dumps(one), encoding="utf8")
    return root


class TestTheTermsOfTheReward:
    def test_the_gates_are_the_rules_and_are_not_written_out_twice(self):
        assert tuple(Rule) == rlvr.GATES
        assert len(rlvr.GATES) == 8

    def test_a_gate_that_never_rejects_is_named_as_constant(self):
        fired = rlvr.Fires()
        fired.observe([])
        fired.observe([Problem(Rule.HEAD, "no head")])
        assert Rule.HEAD not in fired.silent
        assert Rule.LATEX in fired.silent
        assert len(fired.silent) == 7

    def test_every_gate_firing_leaves_nothing_constant(self):
        fired = rlvr.Fires()
        for gate in rlvr.GATES:
            fired.observe([Problem(gate, "so")])
        assert fired.silent == ()

    def test_one_page_failing_one_gate_twice_counts_the_term_once(self):
        # A binary term is failed or it is not. Rule 2 reports an unbalanced
        # total and an unclosed run off the same page and that is still one
        # sample scoring 7 rather than one scoring 6.
        fired = rlvr.Fires()
        fired.observe([Problem(Rule.MATH, "odd total"), Problem(Rule.MATH, "unclosed", 4)])
        assert fired.by_rule[Rule.MATH] == 1

    def test_the_pass_rate_is_over_answers_and_not_over_rejections(self):
        fired = rlvr.Fires()
        for _ in range(9):
            fired.observe([])
        fired.observe([Problem(Rule.HEAD, "no head")])
        assert fired.answers == 10
        assert fired.rate == 0.9

    def test_the_largest_term_is_reported_as_a_share_of_rejections(self):
        fired = rlvr.Fires()
        for _ in range(3):
            fired.observe([Problem(Rule.HEAD, "no head")])
        fired.observe([Problem(Rule.SHORT, "thin")])
        assert fired.carries == 0.75

    def test_nothing_rejected_leaves_the_largest_term_at_nought(self):
        fired = rlvr.Fires()
        fired.observe([])
        assert fired.carries == 0.0
        assert fired.rate == 1.0

    def test_no_answers_at_all_does_not_divide_by_nought(self):
        assert rlvr.Fires().rate == 0.0
        assert rlvr.Fires().carries == 0.0

    def test_a_real_page_is_scored_by_the_same_rules_the_fleet_uses(self):
        # Not a stub of validate. The reward has to be the gate suite itself or
        # the measurement is of something nobody runs.
        fired = rlvr.Fires()
        fired.observe(validate("too short", Expect(book="alg-viii", pdf_page=9)))
        assert fired.by_rule[Rule.SHORT] == 1


class TestWhichAnswersCount:
    def test_an_accepted_answer_is_an_answer(self):
        assert rlvr.attempt({"ok": True}) == rlvr.Attempt(True, None)

    def test_a_gate_rejection_names_its_gate(self):
        got = rlvr.attempt({"ok": False, "reason": 'head: no page label in the first line: "x"'})
        assert got == rlvr.Attempt(False, Rule.HEAD)

    def test_a_connection_that_never_opened_is_not_an_answer(self):
        # The one place the infrastructure's bad night must not land. Counted,
        # it shows a group disagreeing with itself when what happened is the
        # model was asked twice because the first ask never arrived.
        assert rlvr.attempt({"ok": False, "reason": "handed back without an attempt: Conn"}) is None

    def test_a_lease_that_expired_is_not_an_answer(self):
        assert rlvr.attempt({"ok": False, "reason": "lease expired, the worker did not"}) is None

    def test_a_call_that_timed_out_is_not_an_answer(self):
        assert rlvr.attempt({"ok": False, "reason": "no answer came back for this page"}) is None

    def test_a_reason_that_only_mentions_a_gate_is_not_a_gate_rejection(self):
        # The gate name has to be the head of the reason. A transport error that
        # happens to say the word head is not rule 4.
        assert rlvr.attempt({"ok": False, "reason": "gave up while reading the head"}) is None

    def test_something_that_is_not_a_record_is_not_an_answer(self):
        assert rlvr.attempt("ok") is None
        assert rlvr.attempt(None) is None


class TestWhetherTheGroupsDiffer:
    def test_a_page_read_once_is_a_group_of_one_and_has_no_spread(self):
        moved = rlvr.Spread()
        moved.observe([rlvr.Attempt(True, None)])
        assert (moved.groups, moved.singles, moved.varied) == (1, 1, 0)

    def test_answers_that_all_agree_are_flat(self):
        moved = rlvr.Spread()
        moved.observe([rlvr.Attempt(False, Rule.HEAD), rlvr.Attempt(False, Rule.HEAD)])
        assert (moved.flat, moved.varied) == (1, 0)

    def test_answers_that_disagree_are_the_trainable_case(self):
        moved = rlvr.Spread()
        moved.observe([rlvr.Attempt(False, Rule.HEAD), rlvr.Attempt(True, None)])
        assert (moved.flat, moved.varied) == (0, 1)

    def test_two_different_gates_still_both_reject_and_that_is_flat(self):
        # The reward here is whether the suite passed, because the history
        # records one reason and not the other seven terms. Two rejections are
        # the same reward whichever gates they were.
        moved = rlvr.Spread()
        moved.observe([rlvr.Attempt(False, Rule.HEAD), rlvr.Attempt(False, Rule.MATH)])
        assert (moved.flat, moved.varied) == (1, 0)

    def test_singletons_are_in_the_denominator(self):
        # The number that would flatter is varied over varied plus flat, taken
        # off a set selected for having varied.
        moved = rlvr.Spread()
        for _ in range(9):
            moved.observe([rlvr.Attempt(True, None)])
        moved.observe([rlvr.Attempt(False, Rule.HEAD), rlvr.Attempt(True, None)])
        assert moved.usable == 0.1

    def test_a_page_with_no_answer_at_all_is_not_a_group(self):
        moved = rlvr.Spread()
        moved.observe([])
        assert moved.groups == 0
        assert moved.usable == 0.0


class TestReadingTheQueue:
    def test_a_retried_page_comes_back_as_one_group(self, tmp_path: Path):
        root = queue(tmp_path, job("alg-viii-fr/0185", "head: no page label", None))
        got = dict(rlvr.history(root))
        assert list(got) == ["alg-viii-fr/0185"]
        assert [a.ok for a in got["alg-viii-fr/0185"]] == [False, True]

    def test_the_transport_failures_are_dropped_before_the_group_is_formed(self, tmp_path: Path):
        root = queue(
            tmp_path,
            job("alg-viii-fr/0185", "handed back without an attempt: Conn", "lease expired", None),
        )
        assert rlvr.spread(root) == rlvr.Spread(groups=1, singles=1, flat=0, varied=0)

    def test_a_job_that_never_ran_is_not_a_group(self, tmp_path: Path):
        root = queue(tmp_path, job("alg-viii-fr/0186"))
        assert list(rlvr.history(root)) == []

    def test_every_state_directory_is_read_and_not_just_done(self, tmp_path: Path):
        root = queue(tmp_path, job("a/0001", None))
        queue(tmp_path, job("b/0002", None), state="dead")
        assert sorted(t for t, _ in rlvr.history(root)) == ["a/0001", "b/0002"]

    def test_a_file_that_is_not_json_is_stepped_over_rather_than_fatal(self, tmp_path: Path):
        root = queue(tmp_path, job("a/0001", None))
        (root / "done" / "broken.json").write_text("{not json", encoding="utf8")
        assert [t for t, _ in rlvr.history(root)] == ["a/0001"]

    def test_a_record_without_a_history_is_stepped_over(self, tmp_path: Path):
        root = queue(tmp_path, {"target": "a/0001"})
        assert list(rlvr.history(root)) == []

    def test_an_empty_queue_reports_no_groups_rather_than_a_clean_nought(self, tmp_path: Path):
        # The caller treats this as the format having moved. It only works if
        # nothing here quietly invents a group.
        root = tmp_path / "ocr"
        (root / "done").mkdir(parents=True)
        assert rlvr.spread(root).groups == 0


class TestTheReport:
    def test_it_names_the_gates_that_never_fired(self):
        fired = rlvr.Fires()
        fired.observe([Problem(Rule.HEAD, "no head")])
        text = rlvr.report(fired, None)
        assert "7 of the 8 gates rejected nothing" in text
        assert "latex" in text and "exercise" in text

    def test_it_says_so_when_every_gate_fired(self):
        fired = rlvr.Fires()
        for gate in rlvr.GATES:
            fired.observe([Problem(gate, "so")])
        assert "Every gate rejected something" in rlvr.report(fired, None)

    def test_it_reports_the_spread_as_a_share_of_every_page(self):
        moved = rlvr.Spread(groups=10, singles=8, flat=1, varied=1)
        assert "10.0% of pages" in rlvr.report(None, moved)

    def test_either_half_stands_on_its_own(self):
        assert "Which terms are constant" not in rlvr.report(None, rlvr.Spread(groups=1))
        assert "Whether the groups differ" not in rlvr.report(rlvr.Fires(), None)

    def test_a_gate_that_never_fired_still_gets_a_row(self):
        # A table that lists only what fired reads like a table of the gates.
        fired = rlvr.Fires()
        fired.observe([Problem(Rule.HEAD, "no head")])
        text = rlvr.report(fired, None)
        for gate in rlvr.GATES:
            assert f"| {gate.value} |" in text

    def test_a_gate_that_was_never_asked_is_not_reported_as_having_passed(self):
        # Rule 7 needs a TeX installation and the fleet does not give it one.
        # Silence there is a question nobody put, and reading it as a result
        # would say the model's LaTeX always compiles on evidence of nothing.
        fired = rlvr.Fires()
        fired.observe([Problem(Rule.HEAD, "no head")])
        text = rlvr.report(fired, None)
        assert "latex is opt in and was not asked" in text

    def test_a_gate_that_ran_everywhere_and_stayed_quiet_gets_no_such_excuse(self):
        # Rule 5 runs on every page. Its silence is a measurement and the report
        # must not soften it the way it softens rule 7's.
        fired = rlvr.Fires()
        fired.observe([Problem(Rule.HEAD, "no head")])
        assert "illegible is opt in" not in rlvr.report(fired, None)

    def test_a_run_that_did_ask_rule_seven_says_nothing_about_it_being_withheld(self):
        fired = rlvr.Fires()
        for gate in rlvr.GATES:
            fired.observe([Problem(gate, "so")])
        assert "was not asked" not in rlvr.report(fired, None)


class TestBothHalvesOffOneWalk:
    def test_the_queue_answers_both_questions(self, tmp_path: Path):
        root = queue(
            tmp_path,
            job("a/0001", "head: no page label", None),
            job("a/0002", None),
        )
        fired, moved = rlvr.queue(root)
        assert (fired.answers, fired.passed) == (3, 2)
        assert fired.by_rule[Rule.HEAD] == 1
        assert (moved.groups, moved.varied, moved.singles) == (2, 1, 1)

    def test_counts_off_a_queue_are_marked_as_such(self, tmp_path: Path):
        root = queue(tmp_path, job("a/0001", None))
        fired, _ = rlvr.queue(root)
        assert fired.recorded
        assert "counted off the reasons a queue recorded" in rlvr.report(fired, None)

    def test_counts_off_the_text_are_marked_as_such(self):
        fired = rlvr.Fires()
        fired.observe([])
        assert not fired.recorded
        assert "scored from the text" in rlvr.report(fired, None)

    def test_a_transport_failure_is_not_an_answer_in_either_half(self, tmp_path: Path):
        root = queue(tmp_path, job("a/0001", "handed back without an attempt: Conn", None))
        fired, moved = rlvr.queue(root)
        assert fired.answers == 1
        assert moved.singles == 1
