"""The six serving benchmarks of §01, measured in accepted pages an hour.

    local-ocr bench --plan --markdown bench.md

Not tokens a second. That is the number every serving framework prints and it
is the wrong one here for two reasons. It counts a page that came back
truncated the same as a page that came back whole, and truncation is the
failure mode that gets worse exactly when the concurrency is raised, so a sweep
scored on tokens a second recommends the setting that produces the most
unusable output. And it counts a refusal as nothing at all rather than as a
page that has to be read again, which hides the cost that a bad setting
actually imposes. Accepted pages an hour is pages that came back and passed the
eight rules, over wall clock, and it is the only number in this file that
anything downstream should be chosen by.

§01 asks six questions:

  engine       vLLM against SGLang
  concurrency  `--max-num-seqs` over 4, 8, 16, 32
  weights      BF16 against FP8
  kv cache     FP8 KV cache on and off
  power        450 W against 350 W
  resolution   300 dpi against 600 dpi

Thirteen arms between them, but not thirteen runs. Five of the six questions
have an arm that is the shipping configuration, and that arm is the same
configuration in all five, so it is measured once and enters every question it
answers. Hence `Variant.questions` is a tuple rather than a string. This is not
only a saving: measuring the same configuration five times and reporting five
slightly different numbers for it would invite somebody to read the spread
between them as a difference between the questions.

What is not here is anything that knows how to start a server or set a power
cap on a card. The reader runs in WSL on a machine reached over ssh, the
details of that belong to whoever is holding the machine, and a benchmark
harness that hard codes them is a harness that cannot be pointed at the next
one. `measure` takes the staging, the reading and the judging as callables, and
the shell that drives a real run passes real ones in.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

ENGINE = "engine"
CONCURRENCY = "concurrency"
WEIGHTS = "weights"
KV = "kv cache"
POWER = "power"
RESOLUTION = "resolution"

QUESTIONS: tuple[str, ...] = (ENGINE, CONCURRENCY, WEIGHTS, KV, POWER, RESOLUTION)
"""The six, in the order §01 asks them."""

ASKED = {
    ENGINE: "vLLM against SGLang",
    CONCURRENCY: "--max-num-seqs over 4, 8, 16 and 32",
    WEIGHTS: "BF16 weights against FP8",
    KV: "an FP8 KV cache on and off",
    POWER: "a 450 W board limit against 350 W",
    RESOLUTION: "300 dpi against 600 dpi",
}


@dataclass(frozen=True)
class Variant:
    """One configuration to be measured, and which questions it is an arm of."""

    name: str
    questions: tuple[str, ...]
    model: str = "reader-a"
    binary: str = "vllm"
    serve_extra: tuple[str, ...] = ()
    watts: int | None = None
    """The board limit to set before the run, or None to leave the card alone."""
    dpi: int = 300
    jobs: int = 16
    """Client side concurrency, kept level with `--max-num-seqs` where it is swept.

    Sweeping the server's queue depth while the client only ever has one request
    in flight measures nothing: the server is never given the chance to batch,
    so every arm reports the same number and the sweep concludes that the
    setting does not matter.
    """

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError(f"{self.name}: a variant that answers no question is dead weight")
        for question in self.questions:
            if question not in QUESTIONS:
                raise ValueError(f"{self.name}: {question!r} is not one of the six")


BASELINE = Variant(
    name="shipping",
    questions=(ENGINE, WEIGHTS, KV, POWER, RESOLUTION),
    serve_extra=("--max-num-seqs", "16"),
    watts=450,
    dpi=300,
    jobs=16,
)
"""What the fleet reads with today, and the arm five of the six questions share.

It is also the concurrency sweep's 16, but it is not listed under
`concurrency`: the sweep's four arms want to be four rows measured the same way
on the same afternoon, and folding one of them in from elsewhere would make the
one row that is not comparable the one the sweep is read off.
"""

PLAN: tuple[Variant, ...] = (
    BASELINE,
    Variant("sglang", (ENGINE,), binary="sglang", serve_extra=()),
    Variant("seqs-4", (CONCURRENCY,), serve_extra=("--max-num-seqs", "4"), jobs=4),
    Variant("seqs-8", (CONCURRENCY,), serve_extra=("--max-num-seqs", "8"), jobs=8),
    Variant("seqs-16", (CONCURRENCY,), serve_extra=("--max-num-seqs", "16"), jobs=16),
    Variant("seqs-32", (CONCURRENCY,), serve_extra=("--max-num-seqs", "32"), jobs=32),
    Variant("bf16", (WEIGHTS,), model="reader-a-bf16", serve_extra=("--max-num-seqs", "16")),
    Variant(
        "kv-fp8",
        (KV,),
        serve_extra=("--max-num-seqs", "16", "--kv-cache-dtype", "fp8"),
    ),
    Variant("350w", (POWER,), serve_extra=("--max-num-seqs", "16"), watts=350),
    Variant("600dpi", (RESOLUTION,), serve_extra=("--max-num-seqs", "16"), dpi=600),
)
"""Ten runs for thirteen arms, because the shipping configuration answers five.

The order is the order to run them in and not only the order to read them in.
`sglang` is second because it is the one arm that can fail to run at all, and
finding that out after eight other runs have already taken the card is finding
it out too late to do anything else with the window.
"""


@dataclass(frozen=True)
class Measured:
    """What one arm did.

    `pages` is what it was given, `read` is what came back at all, and
    `accepted` is what passed the eight rules. All three are kept because the
    gap between the first two is refusals and the gap between the second two is
    bad readings, and a setting can be bad in either way alone.
    """

    variant: Variant
    pages: int
    read: int
    accepted: int
    seconds: float
    joules: float | None = None
    note: str = ""
    """Why this arm has no numbers, empty when it has them."""

    @property
    def skipped(self) -> bool:
        return bool(self.note)

    @property
    def accepted_per_hour(self) -> float:
        """The number everything is chosen by."""
        return 3600 * self.accepted / self.seconds if self.seconds > 0 else 0.0

    @property
    def read_per_hour(self) -> float:
        return 3600 * self.read / self.seconds if self.seconds > 0 else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.pages if self.pages else 0.0

    @property
    def seconds_per_page(self) -> float:
        return self.seconds / self.read if self.read else 0.0

    @property
    def watts(self) -> float | None:
        if self.joules is None or self.seconds <= 0:
            return None
        return self.joules / self.seconds

    @property
    def wh_per_accepted(self) -> float | None:
        """Watt hours to get one page that a person does not have to redo.

        Per accepted and not per page, for the same reason the headline rate is.
        Energy spent on a page that comes back refused has to be spent again,
        and a setting that is efficient per request and refuses a fifth of them
        is not efficient.
        """
        if self.joules is None or not self.accepted:
            return None
        return self.joules / 3600 / self.accepted


class Power:
    """Watt seconds over a run, by sampling the board.

    Sampling rather than reading `total_energy_consumption`, which is a counter
    the driver exposes on some cards and not others and which this one does not
    carry. Two second samples, integrated as a trapezoid, which over a run of
    several minutes is well inside the noise of the reading itself.

    A sample that fails is dropped rather than counted as zero. The card is
    behind an ssh hop into WSL and a sample that times out says something about
    the hop, and counting it as an idle card would report the arm as more
    efficient the worse the link was.
    """

    def __init__(self, sample: Callable[[], float | None], every: float = 2.0) -> None:
        self._sample = sample
        self._every = every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._taken: list[tuple[float, float]] = []

    def __enter__(self) -> Power:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._every * 3)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                watts = self._sample()
            except Exception:
                watts = None
            if watts is not None:
                self._taken.append((time.monotonic(), watts))
            self._stop.wait(self._every)

    @property
    def samples(self) -> int:
        return len(self._taken)

    @property
    def joules(self) -> float | None:
        """None rather than zero when there is nothing to integrate.

        A run with one sample has an instantaneous reading and no duration, and
        turning that into an energy figure would be inventing the part that
        matters.
        """
        if len(self._taken) < 2:
            return None
        total = 0.0
        for (t0, w0), (t1, w1) in zip(self._taken, self._taken[1:], strict=False):
            total += (t1 - t0) * (w0 + w1) / 2
        return total


def nvidia_smi(run: Callable[[Sequence[str]], str]) -> Callable[[], float | None]:
    """A sampler that asks the card what it is drawing, through whatever `run` is.

    `run` is the caller's way onto the machine, which on gamingpc is an ssh hop
    into WSL. Passed in rather than built here so that the harness can be
    pointed at a second card without learning a second way to reach one.
    """

    def sample() -> float | None:
        try:
            out = run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw",
                    "--format=csv,noheader,nounits",
                ]
            )
        except (subprocess.SubprocessError, OSError):
            return None
        line = out.strip().splitlines()[0] if out.strip() else ""
        try:
            return float(line)
        except ValueError:
            return None

    return sample


def environment(variant: Variant) -> dict[str, str]:
    """What an arm looks like to the shell command that has to bring it up.

    Environment and not positional arguments, because the staging command is
    written by whoever owns the machine and every one of these is optional to
    it. A script that only ever restarts vLLM with different flags can read two
    of them and ignore the rest, and adding a seventh here does not break it.
    """
    return {
        "LOCAL_OCR_BENCH_ARM": variant.name,
        "LOCAL_OCR_BENCH_MODEL": variant.model,
        "LOCAL_OCR_BENCH_BINARY": variant.binary,
        "LOCAL_OCR_BENCH_EXTRA": shlex.join(variant.serve_extra),
        "LOCAL_OCR_BENCH_WATTS": "" if variant.watts is None else str(variant.watts),
        "LOCAL_OCR_BENCH_DPI": str(variant.dpi),
        "LOCAL_OCR_BENCH_JOBS": str(variant.jobs),
    }


def shell_stage(
    command: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    base: dict[str, str] | None = None,
) -> Callable[[Variant], None]:
    """Staging by handing the arm to a shell command.

    Nothing in this package knows how to reach the card. It is a 4090 in WSL on
    a machine at the end of an ssh hop, bringing a reader up there means killing
    the one that is already serving and waiting for thirty gigabytes to page in,
    and every one of those details belongs to whoever is holding the machine
    rather than to a benchmark. So the command is given on the command line and
    the arm arrives in its environment.

    A non zero exit raises, and `measure` turns that into a row that says the
    arm would not start. That is the path SGLang is expected to take.
    """

    def stage(variant: Variant) -> None:
        env = dict(base if base is not None else os.environ)
        env.update(environment(variant))
        done = run(command, shell=True, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "").strip().splitlines()
            why = tail[-1] if tail else f"exit {done.returncode}"
            raise RuntimeError(why)

    return stage


def wait_ready(
    poll: Callable[[], bool],
    *,
    timeout: float = 900.0,
    every: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Block until the reader answers, or say how long it did not.

    Generous by default, because the honest floor is how long the weights take
    to come off disk and that is thirty to ninety seconds for reader-a and
    longer the first time a repository is pulled. This sits outside the clock
    `measure` keeps, so being generous here costs the report nothing.
    """
    started = clock()
    while True:
        if poll():
            return
        if clock() - started >= timeout:
            raise TimeoutError(f"the reader did not answer within {timeout:.0f}s")
        sleep(every)


def measure(
    variant: Variant,
    *,
    stage: Callable[[Variant], None],
    read: Callable[[Variant], tuple[int, int]],
    accept: Callable[[Variant], int],
    pages: int,
    power: Callable[[Variant], Power | None] = lambda _v: None,
    clock: Callable[[], float] = time.monotonic,
) -> Measured:
    """One arm, start to finish.

    The clock starts after staging and stops before judging, on purpose. Loading
    seven billion parameters off disk is thirty to ninety seconds and belongs to
    the model rather than to the setting under test, and judging happens on this
    laptop and would put the laptop into a number about the card.

    A staging failure is a `Measured` with a note rather than an exception,
    because the one arm that is expected to fail is SGLang and losing the nine
    arms behind it to that would waste the window.
    """
    try:
        stage(variant)
    # Broad on purpose. Staging is somebody's shell command over ssh and there
    # is no useful list of what it can raise.
    except Exception as err:
        return Measured(variant, pages, 0, 0, 0.0, None, note=f"would not start: {err}")

    meter = power(variant)
    started = clock()
    try:
        if meter is None:
            got, back = read(variant)
        else:
            with meter:
                got, back = read(variant)
    except Exception as err:  # The same, for the batch.
        return Measured(variant, pages, 0, 0, clock() - started, None, note=f"died: {err}")
    seconds = clock() - started

    return Measured(
        variant=variant,
        pages=got,
        read=back,
        accepted=accept(variant),
        seconds=seconds,
        joules=meter.joules if meter is not None else None,
    )


def run(
    variants: Iterable[Variant],
    *,
    stage: Callable[[Variant], None],
    read: Callable[[Variant], tuple[int, int]],
    accept: Callable[[Variant], int],
    pages: int,
    power: Callable[[Variant], Power | None] = lambda _v: None,
    say: Callable[[str], None] = print,
) -> list[Measured]:
    out: list[Measured] = []
    for variant in variants:
        say(f"{variant.name}: staging")
        got = measure(variant, stage=stage, read=read, accept=accept, pages=pages, power=power)
        out.append(got)
        if got.skipped:
            say(f"{variant.name}: {got.note}")
        else:
            say(
                f"{variant.name}: {got.accepted} of {got.pages} accepted in "
                f"{got.seconds:.0f}s, {got.accepted_per_hour:.0f} accepted pages an hour"
            )
    return out


@dataclass
class Board:
    """Every arm that ran, and what each of the six questions comes to."""

    model: str
    set_name: str
    measured: list[Measured] = field(default_factory=list)

    def arms(self, question: str) -> list[Measured]:
        """The arms of one question, best first, with the unmeasured ones last."""
        theirs = [m for m in self.measured if question in m.variant.questions]
        return sorted(theirs, key=lambda m: (m.skipped, -m.accepted_per_hour))

    def winner(self, question: str) -> Measured | None:
        """The fastest arm that produced accepted pages, or None if none did.

        None and not the top of the list, because a question where every arm
        failed to run has no answer and printing the least broken one as the
        winner is how a benchmark comes to recommend a configuration nobody
        measured.
        """
        for arm in self.arms(question):
            if not arm.skipped and arm.accepted:
                return arm
        return None

    def margin(self, question: str) -> float | None:
        """How much the winner beats the runner up by, as a share.

        Reported because most of these questions are expected to come out close,
        and a 2 per cent difference measured once on 40 pages is not a reason to
        change anything. A number next to the winner is what stops the table
        being read as a decision when it is a measurement.
        """
        ran = [m for m in self.arms(question) if not m.skipped and m.accepted]
        if len(ran) < 2 or ran[1].accepted_per_hour <= 0:
            return None
        return ran[0].accepted_per_hour / ran[1].accepted_per_hour - 1

    def unmeasured(self) -> list[Measured]:
        return [m for m in self.measured if m.skipped]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "set": self.set_name,
            "arms": [
                {
                    "name": m.variant.name,
                    "questions": list(m.variant.questions),
                    "model": m.variant.model,
                    "binary": m.variant.binary,
                    "serve_extra": list(m.variant.serve_extra),
                    "watts_cap": m.variant.watts,
                    "dpi": m.variant.dpi,
                    "jobs": m.variant.jobs,
                    "pages": m.pages,
                    "read": m.read,
                    "accepted": m.accepted,
                    "seconds": round(m.seconds, 2),
                    "joules": None if m.joules is None else round(m.joules, 1),
                    "accepted_pages_per_hour": round(m.accepted_per_hour, 2),
                    "acceptance": round(m.acceptance, 4),
                    "wh_per_accepted_page": (
                        None if m.wh_per_accepted is None else round(m.wh_per_accepted, 4)
                    ),
                    "note": m.note,
                }
                for m in self.measured
            ],
            "answers": {
                question: (
                    None
                    if self.winner(question) is None
                    else {
                        "winner": self.winner(question).variant.name,  # type: ignore[union-attr]
                        "margin": (
                            None
                            if self.margin(question) is None
                            else round(self.margin(question), 4)
                        ),
                    }
                )
                for question in QUESTIONS
            },
        }

    def to_markdown(self) -> str:
        out = [
            f"# {self.model} on {self.set_name}, the six serving benchmarks",
            "",
            "Every number below is accepted pages an hour: pages that came back and passed "
            "the eight rules, over wall clock. Tokens a second is not reported, because it "
            "counts a truncated page the same as a whole one and counts a refusal as nothing "
            "at all, and both of those get worse in exactly the direction a sweep pushes.",
            "",
            "The clock starts after the server is up and stops before the readings are "
            "judged, so neither the model load nor this laptop is inside a number about the "
            "card.",
            "",
        ]
        for question in QUESTIONS:
            arms = self.arms(question)
            if not arms:
                continue
            out.append(f"## {question}: {ASKED[question]}")
            out.append("")
            out.append(
                "| Arm | Accepted | Of | Accepted pages an hour | Acceptance | s a page "
                "| W | Wh an accepted page |"
            )
            out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for arm in arms:
                if arm.skipped:
                    out.append(f"| {arm.variant.name} | | | not measured: {arm.note} | | | | |")
                    continue
                watts = "" if arm.watts is None else f"{arm.watts:.0f}"
                wh = "" if arm.wh_per_accepted is None else f"{arm.wh_per_accepted:.3f}"
                out.append(
                    f"| {arm.variant.name} | {arm.accepted} | {arm.pages} | "
                    f"{arm.accepted_per_hour:.0f} | {arm.acceptance:.1%} | "
                    f"{arm.seconds_per_page:.1f} | {watts} | {wh} |"
                )
            out.append("")
            won = self.winner(question)
            if won is None:
                out.append("No arm of this one produced an accepted page, so it has no answer yet.")
            else:
                margin = self.margin(question)
                if margin is None:
                    out.append(
                        f"Only {won.variant.name} produced accepted pages here, so there is "
                        "nothing to compare it against."
                    )
                elif margin < 0.05:
                    out.append(
                        f"{won.variant.name} is ahead by {margin:.1%}, which on {won.pages} "
                        "pages measured once is not a reason to change anything."
                    )
                else:
                    out.append(f"{won.variant.name} wins by {margin:.1%}.")
            out.append("")

        missed = self.unmeasured()
        if missed:
            out.append("## Arms that did not run")
            out.append("")
            out.append(
                "Listed rather than dropped. An arm that is missing from a table reads as an "
                "arm that lost."
            )
            out.append("")
            for arm in missed:
                out.append(f"- {arm.variant.name}: {arm.note}")
            out.append("")
        return "\n".join(out) + "\n"


def write(board: Board, *, json_path: Path | None, markdown_path: Path | None) -> None:
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(board.to_dict(), indent=2) + "\n", encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(board.to_markdown(), encoding="utf-8")


def plan_lines(variants: Iterable[Variant] = PLAN) -> str:
    """What a run would do, without doing any of it.

    Every arm of this takes the whole card for the length of a batch, so the
    window has to be asked for before it is taken and this is what to paste when
    asking.
    """
    out = [
        "| Arm | Answers | Model | Binary | Appended | W | dpi | -j |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for one in variants:
        out.append(
            f"| {one.name} | {', '.join(one.questions)} | {one.model} | {one.binary} | "
            f"`{shlex.join(one.serve_extra) or '-'}` | {one.watts or '-'} | "
            f"{one.dpi} | {one.jobs} |"
        )
    return "\n".join(out) + "\n"
