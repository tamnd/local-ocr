"""A reader that is somebody else's subscription rather than our own card.

Every referee on the M6 shortlist wanted VRAM, and there is none left. Reader A
needs 0.56 of the 4090 to hold a 16k window at eight sequences, which leaves
about six gigabytes, and the two candidates that fit in six gigabytes both
collapse into repeated tokens on a dense French page. The two that would not
collapse do not fit. That is the whole of the M6 blockage and it does not have a
solution on one card.

So this reader does not use the card. `codex exec` is the local ChatGPT
subscription, driven from the command line, and it reads an image if you hand it
one. It costs no VRAM, it runs beside reader A rather than instead of it, and it
belongs to a different model family than anything we would serve locally, which
is the property M6 actually wants from a referee. Two readers that share a base
model agree with each other about the things they are both wrong about.

## Why this exact command line

    codex exec --model gpt-5.4 --skip-git-repo-check -i page.png -- "prompt"

Measured against three alternatives on four pages, sixteen runs, and the ranking
was not the expected one:

    attach to gpt-5.4          70s   14102 tokens
    downscale to 1400px        98s   13880 tokens
    attach to gpt-5.4-mini    171s   23383 tokens
    let codex open the file   309s   42033 tokens

The full model is the cheapest, the fastest and the most accurate at once. The
cut down model reasons and retries where the full one just reads: on a dense
French page it burned 41721 tokens and returned eleven mathematical spans
against the sixty four the reference has, which is to say it gave up in the
middle and charged for the attempt. Letting codex open the file itself is the
option that looks tidiest and is three times the cost, because it becomes a
shell tool call and every tool turn re-sends the transcript. Downscaling saves
nothing measurable and loses accuracy, so the vision tokens are evidently not
priced by pixel count.

Hence: attach the image, use the full model, do not downscale, do not let it go
looking for the file.

## The `--` is not optional

`-i` takes a variadic list of files. Without the separator, `codex exec -i img
"$prompt"` reads the prompt as a second image path and dies with "No prompt
provided via stdin", which is a confusing way to say "you passed two images and
no question".

## What this refuses to do

It is not a bulk lane and the CLI should not let it become one. Fourteen
thousand tokens a page against the pages still queued in the corpus is a number
with eight digits in it, and the card reads the same pages for nothing. This is
the referee, and it earns its cost on exactly the pages where a second opinion
is worth having.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from local_ocr.batch import Refused

BIN = "codex"
MODEL = "gpt-5.4"

# Fifteen minutes, matching the Go side's page timeout. The measured mean is
# seventy seconds and the worst page in the audit took a hundred and twenty, so
# this is not a working limit; it is there so a wedged subprocess cannot hold a
# lane forever.
TIMEOUT = 900.0

MARK = "tokens used"
"""The line codex prints before the token count, and the only reliable landmark
in the transcript. See `answer` for what the shape around it is."""

# Said by codex itself rather than by the model, and each one means the run
# never reached the page. Distinguished from a refusal about content because
# these are worth retrying and a content refusal is not.
BROKEN = (
    "no prompt provided",
    "not supported when using codex with a chatgpt account",
    "stream error",
    "we're currently experiencing high demand",
)

REFUSALS = (
    "i'm sorry, i can't",
    "i cannot assist",
    "i can't help with that",
)


def answer(raw: str) -> tuple[str, int]:
    """The finished reply and what it cost, out of a codex transcript.

    The transcript is a banner, the word `user`, the prompt echoed back, the
    word `codex`, the reply streamed a token at a time, the line `tokens used`,
    the count, and then the finished reply printed a second time. Both copies of
    the reply are in the output and they are not always identical, because the
    streamed one can carry partial lines. The one after the count is the one
    codex considers final, so that is the one to take.

    Falls back to the whole transcript when the marker is missing, which happens
    when codex dies early. That text goes on to fail the acceptance rules, which
    is the correct outcome: better a page rejected with a strange body somebody
    can read than a page silently dropped.
    """
    lines = raw.splitlines()
    mark = max((i for i, line in enumerate(lines) if line.strip() == MARK), default=-1)
    if mark < 0:
        return raw.strip(), 0
    count = lines[mark + 1].strip().replace(",", "") if mark + 1 < len(lines) else ""
    return "\n".join(lines[mark + 2 :]).strip(), int(count) if count.isdigit() else 0


@dataclass
class CodexReader:
    """One page image and one prompt through `codex exec`, Markdown out.

    Holds the running token total because that is the number this reader has to
    justify itself with, and a batch log that says how many pages were refereed
    without saying what they cost is not enough to decide whether to keep doing
    it.
    """

    model: str = MODEL
    binary: str = BIN
    timeout: float = TIMEOUT
    # Cumulative across every page this instance has read. Not reset, and read
    # by the batch summary at the end of a run.
    tokens: int = field(default=0, init=False)
    pages: int = field(default=0, init=False)

    async def read(self, image: Path, prompt: str) -> str:
        if not image.exists():
            raise Refused(f"no image at {image}")

        proc = await asyncio.create_subprocess_exec(
            self.binary,
            "exec",
            "--model",
            self.model,
            "--skip-git-repo-check",
            "-i",
            str(image),
            "--",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise Refused(f"codex gave up on {image.name} after {self.timeout:.0f}s") from None
        except FileNotFoundError:
            raise Refused(f"no {self.binary} on PATH, the subscription reader needs it") from None

        raw = out.decode("utf-8", "replace")
        text, tokens = answer(raw)
        self.tokens += tokens
        self.pages += 1

        low = text.lower()
        if proc.returncode != 0 or any(phrase in low for phrase in BROKEN):
            raise Refused(f"codex failed on {image.name}: {_first(text) or 'no output'}")
        if any(phrase in low for phrase in REFUSALS):
            raise Refused(f"codex declined {image.name}: {_first(text)}")
        if not text.strip():
            raise Refused(f"codex returned nothing for {image.name}")
        return text

    def cost(self) -> str:
        """One line for the batch summary, or empty if this reader never ran."""
        if not self.pages:
            return ""
        return (
            f"codex {self.model}: {self.pages} pages, {self.tokens} tokens, "
            f"{self.tokens // self.pages} a page"
        )


def _first(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""
