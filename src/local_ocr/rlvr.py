"""Whether the gate suite can be a reward, measured before a card is spent on it.

    local-ocr rlvr-signal --set golden-dev --readings out/head3
    local-ocr rlvr-signal --history $BOURBAKI_CORPUS/work/queue/ocr

Section 08 puts RLVR after the style LoRA and argues that the ingredients are
unusually well aligned, because the reward does not have to be invented: the
gate suite is eight binary checks and this project has been running them in
production for months. That is true, and it is not the whole precondition. This
measures the half that section 08 does not state.

GRPO learns from the spread of reward inside a group of samples of one prompt.
An advantage is a sample's reward less its group's mean, so a group whose
samples all score the same contributes exactly nothing to the update, whatever
that score is. A reward can be correct, cheap, aligned with the thing you
actually want, and still be untrainable, because it is already saturated on the
pages you have.

Two questions follow from that, and neither of them needs the card.

**Which terms are constant.** A gate that never fires over a corpus of readings
takes the value 1 in every sample of every group. It cannot enter an advantage,
in any group, ever. That is not a weak term. It is not a term.

**Whether the groups differ.** A page the queue read more than once holds real
resamples of one prompt over one image, drawn by the machine that would be
trained rather than by an assumption about it. Whether those answers agreed is
the group spread, observed rather than hoped for.

Neither number is an argument against RLVR. They are the two numbers that say
whether a run would learn anything, and they are cheap enough that going without
them would be a choice rather than an oversight.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from local_ocr.rules.validate import Problem, Rule, validate

GATES: tuple[Rule, ...] = tuple(Rule)
"""The terms of the reward, in the order `validate.Rule` declares them.

Taken from the enum and not written out again, so that a ninth rule joins the
reward by existing rather than by somebody remembering to come back here.
"""

WITHHELD: frozenset[Rule] = frozenset({Rule.LATEX})
"""Gates that can be silent because nobody asked them, not because they pass.

Rule 7 is opt in. `validate` runs it only when handed something with a `check`,
because it costs a TeX installation and a subprocess per page, and the fleet
does not hand it one. So a run where it rejected nothing is a run where it was
never consulted.

The distinction does not change the arithmetic. A term that is 1 in every sample
of every group is a constant whether it passed or was never put, and it drops
out of the advantage either way. It changes what the number licenses: a gate
that is silent because it was withheld is a term the reward could be widened
with, and a gate that is silent after running on every page is not.
"""


@dataclass
class Fires:
    """How often each gate rejected, over readings scored here.

    Scored here and not counted out of a log, because the question is about the
    reward a trainer would compute, and a trainer computes it from the text it
    just sampled. A log says what the fleet did with a different prompt on a
    different day.

    One caveat that belongs on the number and not in a footnote: `validate`
    returns as soon as rule 3 fires, because a refusal is short and headless and
    has no mathematics, and listing four symptoms of one event helps nobody. So
    a page that leaks is counted against `leak` alone and the gates it would
    also have failed are not seen. It moves this in the direction of reporting
    more signal than there is, which is the safe direction for a measurement
    whose conclusion is that there is little.
    """

    answers: int = 0
    passed: int = 0
    by_rule: Counter[Rule] = field(default_factory=Counter)
    recorded: bool = False
    """Whether these came out of a queue's reasons rather than off the text.

    Carried so the report can say which it was. The two are not interchangeable
    and a table that does not say which one it is will be read as the stronger
    of the two by whoever finds it next.
    """

    def observe(self, problems: Sequence[Problem]) -> None:
        self.answers += 1
        if not problems:
            self.passed += 1
            return
        # By distinct rule. Rule 2 can report an unbalanced total and an
        # unclosed inline run off one page, and a term of a binary reward is
        # failed once or not at all.
        for rule in {problem.rule for problem in problems}:
            self.by_rule[rule] += 1

    def saw(self, one: Attempt) -> None:
        """One answer as the queue recorded it, rather than as scored from text.

        A weaker reading of the same question and a very much larger one. The
        history keeps one reason per rejection where `validate` hands back every
        rule that failed, so a page that was both short and headless is counted
        here against whichever the Go side reported. It can therefore undercount
        a gate. It cannot overcount one, and it cannot show a gate firing that
        never did, which are the two things a claim about constant terms rests
        on.
        """
        self.recorded = True
        self.answers += 1
        if one.ok:
            self.passed += 1
        elif one.rule is not None:
            self.by_rule[one.rule] += 1

    @property
    def silent(self) -> tuple[Rule, ...]:
        """The gates that rejected nothing at all."""
        return tuple(gate for gate in GATES if not self.by_rule[gate])

    @property
    def rate(self) -> float:
        """The share of answers that passed every gate."""
        return self.passed / self.answers if self.answers else 0.0

    @property
    def carries(self) -> float:
        """The largest share of rejections any one gate accounts for.

        A reward of eight terms where one term is nearly all of the movement is
        a reward of one term with seven along for the ride, and it should be
        compared against whatever fixes that one term directly.
        """
        total = sum(self.by_rule.values())
        return max(self.by_rule.values()) / total if total else 0.0


@dataclass(frozen=True)
class Attempt:
    """One answer that came back from a model and was judged."""

    ok: bool
    rule: Rule | None
    """Which gate rejected it, and None when it was accepted."""


def attempt(entry: object) -> Attempt | None:
    """One history entry as an answer, or None when nothing was answered.

    A lease that expired, a connection that never opened and a call that timed
    out are all written into the history the same way a rejection is, and none
    of them is one. Counting them would put the infrastructure's bad night into
    the reward's variance, which is the one place it must not go: it would show
    a group disagreeing with itself when what happened is that the model was
    asked twice because the first ask never arrived.
    """
    if not isinstance(entry, dict):
        return None
    if entry.get("ok"):
        return Attempt(True, None)
    head = str(entry.get("reason") or "").split(":", 1)[0].strip()
    for gate in GATES:
        if head == gate.value:
            return Attempt(False, gate)
    return None


@dataclass
class Spread:
    """What the production resamples say about variance within a group."""

    groups: int = 0
    singles: int = 0
    """Pages with one recorded answer. A group of one has no spread."""
    flat: int = 0
    """More than one answer, and every one of them agreed."""
    varied: int = 0
    """More than one answer, and they did not agree. The trainable case."""

    def observe(self, attempts: Sequence[Attempt]) -> None:
        if not attempts:
            return
        self.groups += 1
        if len(attempts) == 1:
            self.singles += 1
        elif len({one.ok for one in attempts}) == 1:
            self.flat += 1
        else:
            self.varied += 1

    @property
    def usable(self) -> float:
        """The share of groups that would contribute an advantage.

        Singletons count against it rather than being left out. A page read once
        is a group of one, drawing eight of it at temperature might turn up a
        disagreement and might not, and the honest denominator is every page and
        not the ones that happened to be interesting. Reporting `varied` over
        `varied + flat` would quote a number off a set selected for having
        varied, which is how a saturated reward comes to look healthy.
        """
        return self.varied / self.groups if self.groups else 0.0


def history(root: Path) -> Iterator[tuple[str, list[Attempt]]]:
    """Every page the queue holds a record for, with the answers it got.

    Reads the Go queue's job files, `<root>/<state>/<id>.json`, each carrying a
    `target` and a `history` of entries with `ok` and `reason`. The shape is
    pinned here rather than imported, because the queue lives in another
    repository in another language and there is no package to depend on. A
    reader that guesses is a reader that returns nothing the day the format
    moves and says so by reporting a very clean zero, so the caller is expected
    to treat no groups at all as a fault and not as an answer.
    """
    for state in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(state.glob("*.json")):
            try:
                job = json.loads(path.read_text(encoding="utf8"))
            except (OSError, ValueError):
                continue
            if not isinstance(job, dict):
                continue
            entries = job.get("history")
            if not isinstance(entries, list):
                continue
            answers = [one for one in (attempt(e) for e in entries) if one is not None]
            if answers:
                yield str(job.get("target") or path.stem), answers


def queue(root: Path) -> tuple[Fires, Spread]:
    """Both halves of the question off one walk of a queue stage.

    One walk because the corpus queue is fourteen thousand small files and
    reading it twice to answer two questions about the same records would be
    two minutes spent to keep two functions tidy.
    """
    fired, moved = Fires(), Spread()
    for _, answers in history(root):
        moved.observe(answers)
        for one in answers:
            fired.saw(one)
    return fired, moved


def spread(root: Path) -> Spread:
    """The group spread over a whole queue."""
    return queue(root)[1]


def fires(pages: Sequence[object], readings: Path) -> Fires:
    """The reward's terms, over one directory of readings.

    Imported inside the call the way `residual` does it, because `evaluate`
    pulls in the metrics and this module is also asked for the queue reader,
    which needs none of that.
    """
    from local_ocr import evaluate

    out = Fires()
    for page in pages:
        found = evaluate.find_reading(readings, page.id)  # type: ignore[attr-defined]
        if found is None:
            continue
        out.observe(validate(found.read_text(encoding="utf8"), evaluate.expect_from(page)))
    return out


def _terms(fired: Fires) -> list[str]:
    total = sum(fired.by_rule.values())
    lines = [
        "| Gate | Rejections | Share of rejections |",
        "| --- | --- | --- |",
    ]
    for gate in GATES:
        count = fired.by_rule[gate]
        share = f"{count / total:.1%}" if total else "n/a"
        lines.append(f"| {gate.value} | {count} | {share} |")
    lines.append("")
    return lines


def report(fired: Fires | None = None, moved: Spread | None = None) -> str:
    """The whole thing, as the Markdown the milestone asks to have written down."""
    lines = ["# Does the gate suite carry a reward", ""]
    if fired is not None:
        how = (
            "counted off the reasons a queue recorded, so a page that failed two gates is "
            "counted against one of them"
            if fired.recorded
            else "scored from the text, the way a trainer would"
        )
        lines.extend(
            [
                "## Which terms are constant",
                "",
                f"{fired.answers} answers, {fired.passed} passed every gate, {fired.rate:.1%}. "
                f"These were {how}.",
                "",
                *_terms(fired),
            ]
        )
        silent = fired.silent
        if silent:
            names = ", ".join(gate.value for gate in silent)
            lines.extend(
                [
                    f"{len(silent)} of the {len(GATES)} gates rejected nothing: {names}. "
                    "Those terms are 1 in every sample of every group and cannot enter an "
                    "advantage.",
                    "",
                ]
            )
            withheld = [gate for gate in silent if gate in WITHHELD]
            if withheld:
                names = ", ".join(gate.value for gate in withheld)
                lines.extend(
                    [
                        f"Of those, {names} is opt in and was not asked, so its silence is not a "
                        "result. It is a term the reward could be widened with rather than one "
                        "that has been shown to pass.",
                        "",
                    ]
                )
        else:
            lines.extend(["Every gate rejected something at least once.", ""])
        if fired.by_rule:
            lines.extend([f"The largest single gate is {fired.carries:.1%} of all rejections.", ""])
    if moved is not None:
        lines.extend(
            [
                "## Whether the groups differ",
                "",
                f"{moved.groups} pages have a recorded answer. {moved.singles} were answered "
                f"once, {moved.flat} were answered more than once and agreed, {moved.varied} "
                "were answered more than once and did not.",
                "",
                f"So {moved.usable:.1%} of pages are observed to carry a spread a group would "
                "learn from.",
                "",
            ]
        )
    return "\n".join(lines)
