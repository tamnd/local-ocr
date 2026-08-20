# local-ocr

Page transcription on one GPU, speaking the batch protocol that `tamnd/bourbaki-solver` already uses.

## What it is

A directory of scanned page images goes in and a directory of Markdown comes out. The command line is a drop-in for the `chatgpt-tool ocr-batch` subcommand, so the Bourbaki fleet can drive a machine running this with no change to its own transport, its queue, its acceptance rules or its page contract.

That last part is the whole design. The Go side owns the queue, the content addressed job ids, the eight acceptance rules, the prompts and the page files. This project replaces one thing, the transport and the model behind it.

## Why

OCR is the bottleneck. The fleet reads through browser accounts with a handful of uploads a day between them, a page costs about 151 seconds when it works, and the queue currently holds more than eleven thousand pages. A single consumer GPU at even 180 pages an hour clears that in a few days. The hardware is already on the desk and does nothing at night.

## Running it

```
local-ocr ocr-batch <in> <out> -j 8 --rate-delay 0 --ext png \
  --skip-existing --recursive --timeout 900 --prompt "$(cat prompt.md)"
```

That is the exact shape the Go side builds, quoting and all. Every path is relative to the process working directory, because there is no `cd` in the command the fleet sends.

Anything not in that list of arguments is an error rather than an argument quietly ignored, so a future change on the Go side fails on the first batch instead of reading a thousand pages with the wrong settings.

## The invisible requirements

Four things are not visible in the command line and all four matter.

The output tree mirrors the input tree with the extension swapped, so `0042.png` becomes `0042.md`. The caller stats exactly that path, and a different naming scheme reads as every page missing.

Output is written through a temporary name in the same directory and renamed. The caller polls `ls -1 <out> | grep -c '\.md$'` and pulls as soon as the count reaches the page count, so a file that exists while it is still being written counts as an answer.

A zero byte file counts as missing on the caller's side, so an empty answer never reaches a final path at all. It leaves a `.refused` marker instead, and so does a page that times out or that the model declines.

The process prints its pid, detaches, survives the ssh session that started it, and dies entirely under a process group kill.

## Development

```
uv sync
uv run pytest
uv run ruff check .
uv run ruff format .
```

The tests need no GPU. The reader is behind a protocol and the contract tests stub it, which is deliberate: they are the tests that decide whether the integration works, so they have to run on a laptop and in CI.

`tests/test_go_contract.py` builds the literal shell command `Batch.start` builds in `ocr/batch.go`, runs it through `sh -c`, and then behaves the way `wait()` and `missing()` do. If that file passes, the fleet can drive this.

## Layout

```
src/local_ocr/batch.py       the protocol: walk, skip, lanes, timeout, atomic write
src/local_ocr/cli.py         the command line, which is the contract
src/local_ocr/backends/      one adapter per way of reading a page
tests/test_go_contract.py    the command line the Go side builds, run for real
```

## Licence

MIT.
