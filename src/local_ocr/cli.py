"""The command line, which is a contract with a Go program.

`ocr/batch.go` builds exactly this, with `launcher()` being
`DISPLAY=<display> setsid nohup` on a remote host:

    <launcher> '<tool>' ocr-batch '<in>' '<out>' \\
      -j <lanes> --rate-delay <float> --ext png \\
      --skip-existing --recursive --timeout <seconds> \\
      --prompt "$(cat '<prompt>')" >'<log>' 2>&1 </dev/null & echo $!

Every path is single quoted and every path is relative to the login home. There
is no `cd`, so nothing here may resolve an argument against anything other than
the process working directory.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from local_ocr import __version__
from local_ocr.batch import DEFAULT_TIMEOUT, Options, Reader, run

PROG = "local-ocr"


def _batch_parser() -> argparse.ArgumentParser:
    """The `ocr-batch` arguments, and only these.

    Anything else is an error rather than an argument quietly ignored. A future
    change on the Go side then fails loudly on the first batch instead of
    reading a thousand pages with the wrong settings.
    """
    parser = argparse.ArgumentParser(
        prog=f"{PROG} ocr-batch",
        description="Read a directory of page images into a directory of Markdown.",
        add_help=True,
    )
    parser.add_argument("src", type=Path, help="input directory of page images")
    parser.add_argument("dst", type=Path, help="output directory for Markdown")
    parser.add_argument("-j", type=int, default=1, dest="lanes", help="pages read at once")
    parser.add_argument(
        "--rate-delay",
        type=float,
        default=0.0,
        help="seconds between one job starting and the next",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=None,
        help="image extension to read, repeatable, comma separated accepted",
    )
    parser.add_argument("--skip-existing", action="store_true", help="leave pages already read")
    parser.add_argument("--recursive", action="store_true", help="walk subdirectories")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="seconds allowed for one page, after which that page is refused",
    )
    parser.add_argument("--prompt", default="", help="the instruction, sent to the model unedited")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="read the instruction from a file instead, for running this by hand",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("LOCAL_OCR_BACKEND", "vllm"),
        help="which reader to use",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LOCAL_OCR_MODEL", "reader"),
        help="served model name",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LOCAL_OCR_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="where the model server is",
    )
    return parser


def _extensions(given: list[str] | None) -> tuple[str, ...]:
    if not given:
        return ("png",)
    out: list[str] = []
    for item in given:
        out.extend(part for part in item.split(",") if part.strip())
    return tuple(out)


def _options(args: argparse.Namespace) -> Options:
    return Options(
        src=args.src,
        dst=args.dst,
        lanes=args.lanes,
        rate_delay=args.rate_delay,
        ext=_extensions(args.ext),
        skip_existing=args.skip_existing,
        recursive=args.recursive,
        timeout=args.timeout,
    )


def _prompt(args: argparse.Namespace) -> str:
    """The instruction, exactly as it arrived.

    The Go side passes it as `--prompt "$(cat file)"`. Command substitution
    output is not re-expanded by the shell, so the dollar signs of the LaTeX in
    that prompt arrive as themselves and must leave here as themselves.

    local-ocr does not own the prompt. It is not appended to, not summarised,
    not replaced by a model's recommended template. The prompt is the
    specification: a model that improves the prose is destroying data.
    """
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    return args.prompt


def _reader(args: argparse.Namespace) -> Reader:
    from local_ocr.backends import build

    return build(args.backend, model=args.model, base_url=args.base_url)


def ocr_batch(argv: Sequence[str], reader: Reader | None = None) -> int:
    parser = _batch_parser()
    args, unknown = parser.parse_known_args(list(argv))
    if unknown:
        # Named, so that whoever changed the Go side can see what it sent.
        print(
            f"{PROG}: unrecognised argument {unknown[0]!r}: "
            "this command is a drop-in for chatgpt-tool ocr-batch and accepts only "
            "the arguments that protocol sends",
            file=sys.stderr,
        )
        return 2

    prompt = _prompt(args)
    if not prompt.strip():
        print(f"{PROG}: no prompt, and the prompt is the specification", file=sys.stderr)
        return 2
    if not args.src.is_dir():
        print(f"{PROG}: {args.src} is not a directory", file=sys.stderr)
        return 2

    opts = _options(args)
    if reader is None:
        reader = _reader(args)

    def log(line: str) -> None:
        # Standard output is redirected to the log file by the caller, so this
        # is the remote log and the only account of what the tool thought it was
        # doing. Flushed, because a batch that dies with a buffer full of the
        # explanation is a batch nobody can debug.
        print(line, flush=True)

    summary = asyncio.run(run(opts, reader, prompt, log=log))
    return 0 if summary.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"-V", "--version"}:
        print(f"{PROG} {__version__}")
        return 0
    if not argv or argv[0] in {"-h", "--help"}:
        print(_usage())
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command == "ocr-batch":
        return ocr_batch(rest)
    print(f"{PROG}: no command {command!r}\n\n{_usage()}", file=sys.stderr)
    return 2


def _usage() -> str:
    return (
        f"usage: {PROG} <command> [options]\n"
        "\n"
        "commands:\n"
        "  ocr-batch <in> <out>   read a directory of page images into Markdown\n"
        "\n"
        f"{PROG} ocr-batch is a drop-in for the chatgpt-tool subcommand of the same\n"
        "name, so that the Bourbaki fleet can drive this machine with no change to\n"
        "its own transport. See spec 2028 section 04.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
