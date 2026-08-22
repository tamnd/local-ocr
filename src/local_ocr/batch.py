"""The batch protocol, as the Go side already speaks it.

This module is an impersonation and not a design. `ocr/batch.go` in
`tamnd/bourbaki-solver` drives a machine that reads pages, and it was debugged
the hard way: a batch that reported four pages in forty seconds because it had
inherited the previous run's answers, a run that killed itself with a `cd` that
doubled the scratch path, a host that decayed from 63 seconds a page to 1381
because abandoned batches piled up on it. None of that is worth relearning, so
the contract here is copied rather than invented.

Four of the requirements below are invisible in the command line and all four
are load bearing. They are marked where they appear.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

# ---------------------------------------------------------------------------
# What a reader is


class Reader(Protocol):
    """One page image and one prompt in, Markdown out.

    Everything model shaped lives behind this. The batch machinery has never
    heard of a GPU, which is what lets the contract tests in `tests/` run on a
    laptop with the reader stubbed.
    """

    async def read(self, image: Path, prompt: str) -> str: ...


class Refused(Exception):
    """The reader will not answer for this page.

    Distinct from an error, because a refusal is a fact about the page and an
    error is a fact about the run. Both leave a marker and neither leaves a
    `.md`, but only one of them is worth retrying on a different day.
    """


# ---------------------------------------------------------------------------
# Options, which are exactly the command line in spec 2028 section 04

DEFAULT_TIMEOUT = 900.0
"""Seconds per page. The Go side's `DefaultPageTimeout` is fifteen minutes."""


@dataclass(frozen=True)
class Options:
    src: Path
    dst: Path
    lanes: int = 1
    rate_delay: float = 0.0
    ext: tuple[str, ...] = ("png",)
    skip_existing: bool = False
    recursive: bool = False
    timeout: float = DEFAULT_TIMEOUT


# ---------------------------------------------------------------------------
# Finding the pages, and naming their answers


def _suffixes(ext: Iterable[str]) -> tuple[str, ...]:
    out = []
    for name in ext:
        name = name.strip().lstrip(".").lower()
        if name:
            out.append("." + name)
    return tuple(dict.fromkeys(out))


def pages(opts: Options) -> list[Path]:
    """Every image the batch should consider, in a stable order.

    Sorted, because the order pages are read in is the order they appear in a
    log, and a log that cannot be compared against yesterday's is worth much
    less. The Go side pushes the images in its own order and matches the output
    back by name, so nothing downstream depends on this; it is for whoever is
    reading the log at two in the morning.
    """
    wanted = _suffixes(opts.ext)
    walk = opts.src.rglob("*") if opts.recursive else opts.src.glob("*")
    found = [p for p in walk if p.is_file() and p.suffix.lower() in wanted]
    return sorted(found)


def output_for(image: Path, opts: Options) -> Path:
    """Where the Markdown for an image goes.

    `OutputName` on the Go side is `TrimSuffix(image, Ext(image)) + ".md"`, and
    `missing()` stats exactly that path under the destination. A different
    naming scheme reads as every page missing, so this mirrors the input tree
    and swaps the extension and does nothing else.
    """
    rel = image.relative_to(opts.src)
    return opts.dst / rel.with_suffix(".md")


def refusal_for(answer: Path) -> Path:
    """The sentinel for a page that produced no Markdown.

    A `.refused` next to where the `.md` would have been. The poller counts
    `\\.md$` and nothing else, so this is invisible to it, which is correct: a
    page that could not be read must not count as done.
    """
    return answer.with_suffix(".refused")


# ---------------------------------------------------------------------------
# Writing


def write_atomic(path: Path, text: str) -> None:
    """Write through a temporary name in the same directory, then rename.

    LOAD BEARING. `wait()` on the Go side polls

        ls -1 <out> | grep -c '\\.md$'

    and returns as soon as the count reaches the page count. A `.md` that exists
    while it is still being written is counted as an answer and the caller pulls
    it. The same directory matters because rename is only atomic within one
    filesystem.

    The temporary name is dot prefixed and does not end in `.md`, so it is
    neither counted by the poller nor confusing to a person looking at the
    directory.
    """
    if not text.strip():
        # LOAD BEARING. `missing()` treats a zero byte file as absent, so an
        # empty answer must never reach a final path at all.
        raise Refused("the reader returned nothing")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        # The page has been read, so nothing beside it may still say it was not.
        #
        # A refusal is deliberately cheap: the run hands the page straight back
        # with its attempts intact and offers it again later. That retry lands in
        # the same output directory, because a refusal does not spend an attempt
        # and the batch id is a hash over the attempts. So the marker from the
        # failure and the Markdown from the read that followed it end up side by
        # side, and anything downstream has to decide which of the two is the
        # truth about the page. It should never have to. Clearing it here is one
        # unlink at the only point in the program where an answer becomes final.
        with suppress(OSError):
            os.unlink(refusal_for(path))
    except BaseException:
        # A failed write leaves nothing behind. Half a page in the output
        # directory is worse than no page, because the run would have to guess
        # which it was.
        with suppress(OSError):
            os.unlink(tmp)
        raise


def write_refusal(answer: Path, reason: str) -> None:
    """Record that a page was not read, without producing a `.md`."""
    marker = refusal_for(answer)
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=marker.parent, prefix=".", suffix=".part")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(reason.strip() + "\n")
    os.replace(tmp, marker)


# ---------------------------------------------------------------------------
# Running


@dataclass
class Summary:
    """What a batch did. Every page is in exactly one of these."""

    pages: int = 0
    wrote: int = 0
    skipped: int = 0
    refused: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refused

    def line(self) -> str:
        out = f"{self.wrote} written, {self.skipped} skipped, {len(self.refused)} refused"
        return f"{out}, of {self.pages} pages"


class _Gate:
    """Holds the gap between one job starting and the next.

    `--rate-delay` exists to stop a browser pool opening every tab in the same
    instant. There is no browser here and no pool, and the flag is honoured
    anyway, because the contract is worth keeping exact and the cost of doing so
    is this class.
    """

    def __init__(self, delay: float) -> None:
        self._delay = max(0.0, delay)
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        if self._delay <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            gap = self._last + self._delay - now
            if gap > 0:
                await asyncio.sleep(gap)
                now = time.monotonic()
            self._last = now


async def run(
    opts: Options,
    reader: Reader,
    prompt: str,
    log: Callable[[str], None] = print,
) -> Summary:
    """Read every page under `opts.src` into `opts.dst`.

    One page fails on its own. A batch does not abandon 599 pages because the
    600th was a photograph of a blank verso.
    """
    todo: list[Path] = pages(opts)
    summary = Summary(pages=len(todo))
    if not todo:
        log(f"no {'/'.join(opts.ext)} images under {opts.src}")
        return summary

    gate = _Gate(opts.rate_delay)
    lanes = asyncio.Semaphore(max(1, opts.lanes))
    counted = asyncio.Lock()

    async def one(image: Path) -> None:
        answer = output_for(image, opts)
        if opts.skip_existing and answer.exists() and answer.stat().st_size > 0:
            async with counted:
                summary.skipped += 1
            return
        if image.stat().st_size == 0:
            # An empty input is a broken input, and sending it costs a lane, a
            # round trip and a confusing answer. Fifty pages once came back as
            # 400 "cannot identify image file", which read like a broken server
            # and was in fact fifty zero byte files left by a copy that had been
            # interrupted. Say which it is, here, before the model is involved.
            reason = "refused: the page image is empty"
            write_refusal(answer, reason)
            async with counted:
                summary.refused.append(image.name)
            log(f"{image.name}: {reason}")
            return
        async with lanes:
            await gate.wait()
            started = time.monotonic()
            try:
                text = await asyncio.wait_for(reader.read(image, prompt), opts.timeout)
                write_atomic(answer, text)
            except TimeoutError:
                reason = f"timed out after {opts.timeout:g}s"
                write_refusal(answer, reason)
                async with counted:
                    summary.refused.append(image.name)
                log(f"{image.name}: {reason}")
                return
            except Refused as err:
                reason = f"refused: {err}"
                write_refusal(answer, reason)
                async with counted:
                    summary.refused.append(image.name)
                log(f"{image.name}: {reason}")
                return
            except Exception as err:
                # A page that fails for a reason nobody anticipated is still one
                # page. The batch keeps going and the marker keeps the evidence.
                reason = f"{type(err).__name__}: {err}"
                write_refusal(answer, reason)
                async with counted:
                    summary.refused.append(image.name)
                log(f"{image.name}: {reason}")
                return
        async with counted:
            summary.wrote += 1
            done = summary.wrote + summary.skipped
        log(f"{image.name}: {time.monotonic() - started:.1f}s, {done} of {len(todo)}")

    await _gather(todo, one)
    log(summary.line())
    return summary


async def _gather(items: Sequence[Path], work: Callable[[Path], object]) -> None:
    tasks = [asyncio.create_task(work(item)) for item in items]  # type: ignore[arg-type]
    if tasks:
        await asyncio.gather(*tasks)
