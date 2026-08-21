"""The batch machinery, with the reader stubbed.

No GPU, no server, no network. Everything here is about the six requirements in
spec 2028 section 04 that are invisible in the command line.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from local_ocr.batch import (
    Options,
    Refused,
    output_for,
    pages,
    refusal_for,
    run,
    write_atomic,
)


class Stub:
    """A reader whose answers and delays a test chooses."""

    def __init__(self, answer: str = "# page\n\nbody\n", delay: float = 0.0) -> None:
        self.answer = answer
        self.delay = delay
        self.seen: list[Path] = []
        self.starts: list[float] = []
        self.prompts: list[str] = []
        self.live = 0
        self.most = 0

    async def read(self, image: Path, prompt: str) -> str:
        self.starts.append(time.monotonic())
        self.seen.append(image)
        self.prompts.append(prompt)
        self.live += 1
        self.most = max(self.most, self.live)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if callable(self.answer):
                return self.answer(image)
            return self.answer
        finally:
            self.live -= 1


def book(root: Path, names: object = ("0001.png", "0002.png"), sub: str = "") -> Options:
    src = root / "in"
    (src / sub).mkdir(parents=True, exist_ok=True)
    for name in names:  # type: ignore[union-attr]
        (src / sub / name).write_bytes(b"\x89PNG not really")
    return Options(src=src, dst=root / "out")


def test_the_output_tree_mirrors_the_input_and_swaps_the_extension(tmp_path: Path) -> None:
    opts = book(tmp_path, ("0042.png",), sub="chapter/1")
    opts = Options(**{**opts.__dict__, "recursive": True})
    image = opts.src / "chapter/1/0042.png"
    assert output_for(image, opts) == opts.dst / "chapter/1/0042.md"


def test_a_batch_writes_one_markdown_per_image(tmp_path: Path) -> None:
    opts = book(tmp_path)
    summary = asyncio.run(run(opts, Stub(), "read this", log=lambda _: None))
    assert summary.wrote == 2
    assert summary.ok
    assert sorted(p.name for p in opts.dst.iterdir()) == ["0001.md", "0002.md"]


def test_only_the_named_extension_is_read(tmp_path: Path) -> None:
    opts = book(tmp_path, ("0001.png", "0002.jpg", "notes.txt"))
    assert [p.name for p in pages(opts)] == ["0001.png"]


def test_recursive_walks_and_flat_does_not(tmp_path: Path) -> None:
    opts = book(tmp_path, ("0001.png",))
    (opts.src / "deeper").mkdir()
    (opts.src / "deeper" / "0002.png").write_bytes(b"x")
    assert [p.name for p in pages(opts)] == ["0001.png"]
    deep = Options(**{**opts.__dict__, "recursive": True})
    assert [p.name for p in pages(deep)] == ["0001.png", "0002.png"]


def test_skip_existing_leaves_a_page_that_is_already_read(tmp_path: Path) -> None:
    opts = Options(**{**book(tmp_path).__dict__, "skip_existing": True})
    opts.dst.mkdir(parents=True)
    (opts.dst / "0001.md").write_text("read yesterday\n")
    stub = Stub()
    summary = asyncio.run(run(opts, stub, "read this", log=lambda _: None))
    assert summary.skipped == 1
    assert summary.wrote == 1
    assert [p.name for p in stub.seen] == ["0002.png"]
    assert (opts.dst / "0001.md").read_text() == "read yesterday\n"


def test_a_zero_byte_answer_is_not_skipped(tmp_path: Path) -> None:
    """missing() on the Go side treats a zero byte file as absent, so this must too."""
    opts = Options(**{**book(tmp_path, ("0001.png",)).__dict__, "skip_existing": True})
    opts.dst.mkdir(parents=True)
    (opts.dst / "0001.md").write_text("")
    summary = asyncio.run(run(opts, Stub(), "read this", log=lambda _: None))
    assert summary.skipped == 0
    assert summary.wrote == 1


def test_an_empty_answer_never_reaches_a_final_path(tmp_path: Path) -> None:
    opts = book(tmp_path, ("0001.png",))
    summary = asyncio.run(run(opts, Stub(answer="   \n"), "read this", log=lambda _: None))
    assert summary.wrote == 0
    assert summary.refused == ["0001.png"]
    assert not (opts.dst / "0001.md").exists()
    assert (opts.dst / "0001.refused").exists()


def test_a_zero_byte_page_image_is_refused_without_asking_the_reader(tmp_path: Path) -> None:
    # An empty input is a broken input. Sending it costs a lane and a round trip
    # and comes back as a server error, which reads like a broken server and was
    # in fact an interrupted copy.
    opts = book(tmp_path, ("0001.png", "0002.png"))
    (opts.src / "0002.png").write_bytes(b"")
    stub = Stub()
    summary = asyncio.run(run(opts, stub, "read this", log=lambda _: None))
    assert summary.wrote == 1
    assert summary.refused == ["0002.png"]
    assert [p.name for p in stub.seen] == ["0001.png"]
    assert "empty" in (opts.dst / "0002.refused").read_text()


def test_a_refused_page_leaves_a_marker_and_no_markdown(tmp_path: Path) -> None:
    class Declines:
        async def read(self, image: Path, prompt: str) -> str:
            raise Refused("not going to read that")

    opts = book(tmp_path, ("0001.png",))
    summary = asyncio.run(run(opts, Declines(), "read this", log=lambda _: None))
    assert not summary.ok
    assert not (opts.dst / "0001.md").exists()
    assert "not going to read that" in (opts.dst / "0001.refused").read_text()


def test_one_bad_page_does_not_abandon_the_rest(tmp_path: Path) -> None:
    class Fussy:
        async def read(self, image: Path, prompt: str) -> str:
            if image.name == "0002.png":
                raise RuntimeError("the socket went away")
            return "# fine\n\nbody\n"

    opts = book(tmp_path, ("0001.png", "0002.png", "0003.png"))
    summary = asyncio.run(run(opts, Fussy(), "read this", log=lambda _: None))
    assert summary.wrote == 2
    assert summary.refused == ["0002.png"]
    assert "RuntimeError" in (opts.dst / "0002.refused").read_text()


def test_a_page_that_runs_over_its_timeout_is_refused_alone(tmp_path: Path) -> None:
    opts = Options(**{**book(tmp_path, ("0001.png",)).__dict__, "timeout": 0.05})
    summary = asyncio.run(run(opts, Stub(delay=5.0), "read this", log=lambda _: None))
    assert summary.refused == ["0001.png"]
    assert "timed out" in (opts.dst / "0001.refused").read_text()


def test_lanes_bound_how_many_pages_are_read_at_once(tmp_path: Path) -> None:
    names = tuple(f"{n:04d}.png" for n in range(1, 9))
    opts = Options(**{**book(tmp_path, names).__dict__, "lanes": 3})
    stub = Stub(delay=0.05)
    asyncio.run(run(opts, stub, "read this", log=lambda _: None))
    assert stub.most == 3


def test_rate_delay_holds_the_gap_between_job_starts(tmp_path: Path) -> None:
    names = ("0001.png", "0002.png", "0003.png")
    opts = Options(**{**book(tmp_path, names).__dict__, "lanes": 3, "rate_delay": 0.05})
    stub = Stub()
    asyncio.run(run(opts, stub, "read this", log=lambda _: None))
    gaps = [b - a for a, b in zip(stub.starts, stub.starts[1:], strict=False)]
    assert all(gap >= 0.04 for gap in gaps), gaps


def test_the_prompt_reaches_the_reader_unedited(tmp_path: Path) -> None:
    """The prompt is the specification. A model that improves the prose destroys data."""
    prompt = "Use $\\mathbf{Z}$ and never $\\mathbb{Z}$.\nA dangerous bend is ⚡ alone.\n"
    opts = book(tmp_path, ("0001.png",))
    stub = Stub()
    asyncio.run(run(opts, stub, prompt, log=lambda _: None))
    assert stub.prompts == [prompt]


def test_nothing_appears_at_a_final_path_before_it_is_complete(tmp_path: Path) -> None:
    """LOAD BEARING. wait() counts *.md and pulls as soon as the count is reached."""
    seen: list[list[str]] = []

    class Watched:
        async def read(self, image: Path, prompt: str) -> str:
            # While one page is being written, look at the output directory the
            # way the poller does.
            for _ in range(3):
                await asyncio.sleep(0)
                if (tmp_path / "out").exists():
                    seen.append(sorted(p.name for p in (tmp_path / "out").iterdir()))
            return "# page\n\nbody\n"

    opts = Options(**{**book(tmp_path, ("0001.png", "0002.png")).__dict__, "lanes": 2})
    asyncio.run(run(opts, Watched(), "read this", log=lambda _: None))
    for listing in seen:
        for name in listing:
            if name.endswith(".md"):
                # Any .md the poller could have counted must already be whole.
                assert (opts.dst / name).read_text().endswith("body\n")
            else:
                # Everything else is dot prefixed and does not match '\.md$'.
                assert name.startswith("."), name


def test_write_atomic_leaves_nothing_behind_when_the_answer_is_empty(tmp_path: Path) -> None:
    target = tmp_path / "out" / "0001.md"
    with pytest.raises(Refused):
        write_atomic(target, "\n  \n")
    assert not target.exists()
    # The check comes before the directory is made, so an empty answer costs
    # nothing at all: no file, and not even the directory it would have sat in.
    assert not target.parent.exists()


def test_the_refusal_marker_sits_where_the_markdown_would_have(tmp_path: Path) -> None:
    assert refusal_for(Path("out/chapter/0042.md")) == Path("out/chapter/0042.refused")
