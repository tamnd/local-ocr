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

## The reader behind it

`ocr-batch` talks to a model server over the OpenAI vision API and does not care which one, so the server is started separately and lives in its own configuration.

```
local-ocr serve --list          what the shortlist holds
local-ocr serve reader-a        start one, replacing this process with vLLM
local-ocr serve reader-a --print    the command line, without starting anything
```

The shortlist is `src/local_ocr/models.toml`: one table per candidate, holding the repository, a pinned revision, the port and the flags it is served under. Adding a candidate is an entry rather than a patch, and every revision is pinned because a reading is only comparable to another reading of the same weights.

`deploy/local-ocr-reader@.service` runs that under systemd, templated on the entry name, so two readers side by side is two instances rather than two unit files. `deploy/venv.sh` builds the environment it runs in.

vLLM is not a dependency of this project and is not in `pyproject.toml`, because it pulls a CUDA build of torch chosen by the driver on the machine and nothing in CI or on a laptop wants to resolve that. It follows that plain `uv sync` on the reader host removes it, so use `uv sync --inexact` there, or `deploy/venv.sh`. It is worth knowing what that mistake looks like, because it does not look like itself: the running server keeps answering, and then every request comes back 400 "cannot identify image file", because Pillow loads its PNG plugin on the first image and by then the files are gone. Two hundred pages were refused that way before anybody looked at what was installed.

```
sudo cp deploy/local-ocr-reader@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now local-ocr-reader@reader-a
```

## The pages to read

The bake off scores every model on the same 200 pages, the `golden-dev` set drawn from six volumes, so those pages have to exist as images before any of it means anything.

```
local-ocr pages --set golden-dev --dpi 300
```

That builds the same command the corpus builds, `pdftoppm -png -r 300 -gray -f N -l N`, and writes to the same place under the same four digit name, so a bake off page is the page the fleet would have sent and not a page rendered some other way. It exists because `bourbaki render` refuses to rasterise a born digital volume, which is right for the corpus and wrong here: a page with a perfect text layer is the most valuable page in a golden set, because the answer is already written down.

A page already on disk at the dpi asked for is left alone. A page on disk at any other dpi is rendered again and reported, which is not a hypothetical. Two `lie-vii-ix` pages in `golden-dev` were sitting at 600 dpi from an old fleet retry, and a run that read them would have scored some models on one page and the rest on another. Images are never committed, on either side.

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
src/local_ocr/models.toml    the shortlist: repository, revision, port, flags
src/local_ocr/serving.py     an entry in that file turned into a vLLM command line
src/local_ocr/pageimages.py  a golden set rasterised the way the fleet rasterises
deploy/                      the systemd unit that runs one, and the venv it needs
tests/test_go_contract.py    the command line the Go side builds, run for real
```

## Licence

MIT.
