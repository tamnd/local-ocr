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
import shlex
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from local_ocr import __version__, pageimages, serving
from local_ocr import corpus as corpuslib
from local_ocr.batch import DEFAULT_TIMEOUT, Options, Reader, run
from local_ocr.corpus import NoCorpus
from local_ocr.golden import Purpose
from local_ocr.headpass import HeadPass

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


def _head(reader: Reader) -> Reader:
    if os.environ.get("LOCAL_OCR_HEAD_PASS", "1") == "0":
        return reader
    # On by default, and not a command line argument, because the fleet builds
    # a fixed command and cannot pass one. A reading without its running head
    # is rejected by rule 4, so producing one is part of what reading a page
    # means here rather than an option somebody has to know about. The
    # environment variable is for measuring the pass, which is the only reason
    # to turn it off.
    return HeadPass(reader)


def _entry(name: str) -> serving.Model | None:
    """The shortlist entry for a name, or nothing if there is not one."""
    try:
        return serving.model(name)
    except (serving.NoSuchModel, OSError, ValueError):
        return None


def _budget(default: int) -> int:
    """Adjudications a page may pay for, from the environment.

    Zero is allowed and means compare but never spend, which is how the catch
    rate is measured without paying for the crops. Anything unparseable falls
    back rather than failing the batch, because this arrives through a unit file
    and a typo there should not cost a night of reading.
    """
    raw = os.environ.get("LOCAL_OCR_BUDGET", "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _referee_prompt(log: Callable[[str], None]) -> str:
    """What to ask the referee, when it does not read the fleet prompt.

    A path rather than the text itself, because these are model cards' words and
    they belong in a file next to the run that used them. Unreadable is not
    fatal: the referee is then asked what the primary was asked, which is the
    behaviour without the variable at all.
    """
    raw = os.environ.get("LOCAL_OCR_REFEREE_PROMPT_FILE", "").strip()
    if not raw:
        return ""
    try:
        return Path(raw).read_text(encoding="utf-8")
    except OSError as err:
        log(f"{PROG}: referee prompt {raw}: {err}, asking it what the primary was asked")
        return ""


def _second(args: argparse.Namespace, primary: Reader, log: Callable[[str], None]) -> Reader:
    """Put a referee behind the primary, if one is configured.

    Through the environment, for the reason `_head` gives: the fleet builds a
    fixed command line and cannot pass a flag. `LOCAL_OCR_REFEREE` names an
    entry in `models.toml`, which is where the port lives, so a referee is
    switched on with one variable and no other configuration.

    A name that is not in the shortlist and has no URL is not fatal. The batch
    runs on the primary alone and says so, because a referee misspelt in a unit
    file should cost the second opinion and not the pages.

    `codex` is the exception to the URL rule. It is a subprocess against the
    local subscription rather than a server, so there is nothing to point a URL
    at, and it is the only referee that needs no VRAM. Set
    `LOCAL_OCR_REFEREE=codex` and it runs beside reader A on the same card.
    """
    name = os.environ.get("LOCAL_OCR_REFEREE", "").strip()
    if not name:
        return primary

    from local_ocr.backends import build
    from local_ocr.second import BUDGET, SecondPass

    backend, entry = args.backend, _entry(name)
    url = os.environ.get("LOCAL_OCR_REFEREE_URL", "").strip() or (entry.url if entry else "")
    if name.startswith("codex"):
        from local_ocr.backends.codex import MODEL

        backend, url = "codex", ""
        name = name[6:].lstrip(":") or MODEL
    elif not url:
        log(
            f"{PROG}: no referee {name!r} in models.toml and no LOCAL_OCR_REFEREE_URL, "
            "reading with one reader"
        )
        return primary

    mine = _entry(args.model)
    return SecondPass(
        primary,
        _head(build(backend, model=name, base_url=url)),
        second_prompt=_referee_prompt(log),
        crop_prompt=_referee_prompt(log),
        budget=_budget(BUDGET),
        names=(args.model, name),
        models=(mine.repo if mine else "", entry.repo if entry else ""),
        revisions=(mine.revision if mine else "", entry.revision if entry else ""),
    )


def _reader(args: argparse.Namespace, log: Callable[[str], None]) -> Reader:
    from local_ocr.backends import build

    primary = _head(build(args.backend, model=args.model, base_url=args.base_url))
    return _second(args, primary, log)


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

    def log(line: str) -> None:
        # Standard output is redirected to the log file by the caller, so this
        # is the remote log and the only account of what the tool thought it was
        # doing. Flushed, because a batch that dies with a buffer full of the
        # explanation is a batch nobody can debug.
        print(line, flush=True)

    if reader is None:
        reader = _reader(args, log)

    from local_ocr.second import SecondPass

    if isinstance(reader, SecondPass):
        # Before the run, so the one time warning about a missing referee lands
        # in the same log as the pages.
        reader.log = log

    summary = asyncio.run(run(opts, reader, prompt, log=log))

    if isinstance(reader, SecondPass):
        _sidecars(reader, opts, log)
        log(reader.summary())
    head = reader.first if isinstance(reader, SecondPass) else reader
    if isinstance(head, HeadPass) and head.asked:
        # The one line that says whether the second look was worth its time. A
        # run where asked is high and fixed is low is a run where the crop is
        # landing in the wrong place or the model is answering with prose, and
        # neither shows up anywhere else: the reading looks the same either way.
        log(f"head pass: asked on {head.asked} pages, put a head on {head.fixed}")
    return 0 if summary.ok else 1


def _sidecars(reader: object, opts: Options, log: Callable[[str], None]) -> int:
    """Write one record beside each page the second pass read.

    After the run rather than during it, because a `Reader` is handed an image
    and hands back text and has no idea where the Markdown for it is going. This
    does, so this writes them.

    Only beside a page that actually landed. A refused page has no `.md` and a
    record of how two readers failed to produce one belongs in the log, not in a
    file the miner will later read as if it described a reading.

    The name ends in `.ocr.json`, which the Go poller's `grep -c '\\.md$'` does
    not count, so this is invisible to the transport by construction.
    """
    from local_ocr.batch import output_for

    wrote = 0
    for image, record in reader.records.items():  # type: ignore[attr-defined]
        answer = output_for(image, opts)
        if not answer.exists():
            continue
        try:
            record.write(answer)
        except OSError as err:
            log(f"{answer.name}: sidecar not written: {err}")
            continue
        wrote += 1
    return wrote


def _eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} eval",
        description="Judge a directory of readings against a golden set.",
    )
    parser.add_argument("--set", dest="set_name", default="golden-dev", help="which golden set")
    parser.add_argument(
        "--readings", type=Path, required=True, help="directory of Markdown readings to judge"
    )
    parser.add_argument("--model", default="", help="name for the reader, recorded in the report")
    parser.add_argument("--corpus", type=Path, default=None, help="a checkout of tamnd/bourbaki")
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--markdown", dest="markdown_path", type=Path, default=None)
    parser.add_argument(
        "--purpose",
        default="development",
        choices=[p.value for p in Purpose],
        help="what the numbers are for; the held out set only opens for a milestone",
    )
    parser.add_argument(
        "--disagreements",
        type=Path,
        default=None,
        help="also write the adjudication work list here",
    )
    parser.add_argument(
        "--drift",
        type=Path,
        default=None,
        help="page ids extract drift has already flagged, one to a line",
    )
    return parser


def evaluate_cmd(argv: Sequence[str]) -> int:
    from local_ocr import disagree, evaluate, golden

    args = _eval_parser().parse_args(list(argv))
    try:
        report = evaluate.evaluate(
            args.set_name,
            args.readings,
            purpose=golden.Purpose(args.purpose),
            model=args.model,
            corpus=args.corpus,
        )
    except (golden.Burned, KeyError, NoCorpus) as err:
        print(f"{PROG}: {err}", file=sys.stderr)
        return 2

    evaluate.write(report, json_path=args.json_path, markdown_path=args.markdown_path)
    if args.markdown_path is None:
        print(report.to_markdown())

    if args.disagreements is not None:
        drift = _drift(args.drift)
        pages = golden.load(args.set_name, purpose=golden.Purpose(args.purpose), corpus=args.corpus)
        work = disagree.Work()
        for page in pages:
            path = evaluate.find_reading(args.readings, page.id)
            text = path.read_text(encoding="utf-8") if path is not None else ""
            work.items.extend(disagree.classify(page, text, drift=drift))
        args.disagreements.parent.mkdir(parents=True, exist_ok=True)
        args.disagreements.write_text(work.to_markdown(), encoding="utf-8")
        print(f"{len(work.items)} disagreements, written to {args.disagreements}", file=sys.stderr)
    return 0


def _drift(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip() and not line.startswith("#")
    ]


def golden_cmd(argv: Sequence[str]) -> int:
    """`golden draw` and `golden check`, which are deliberately separate.

    Drawing rewrites the manifests and is done once. Checking says how far the
    corpus has moved under them since, and is safe to run whenever.
    """
    from local_ocr import golden

    parser = argparse.ArgumentParser(prog=f"{PROG} golden")
    parser.add_argument("action", choices=["draw", "check", "show"])
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--name", default="", help="for show, which set to list")
    args = parser.parse_args(list(argv))

    if args.action == "show":
        if not args.name:
            for entry in golden.SETS.values():
                held = ", held out" if entry.held_out else ""
                print(f"{entry.name}: tier {entry.tier}, {entry.size} pages{held}")
            return 0
        for page_id in golden.read_manifest(args.name):
            print(page_id)
        return 0

    try:
        corpus = corpuslib.root(args.corpus)
    except NoCorpus as err:
        print(f"{PROG}: {err}", file=sys.stderr)
        return 2

    if args.action == "check":
        for drift in golden.check(corpus):
            print(drift.line())
        return 0

    for path in golden.write_manifests(golden.draw(corpus)):
        print(path)
    return 0


def _pages_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} pages",
        description="Rasterise the pages of a golden set, ready to be read.",
    )
    parser.add_argument("--set", dest="set_name", default="golden-dev", help="which golden set")
    parser.add_argument("--dpi", type=int, default=pageimages.DEFAULT_DPI)
    parser.add_argument("--corpus", type=Path, default=None, help="a checkout of tamnd/bourbaki")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-render pages already on disk, which is how a dpi change takes effect",
    )
    return parser


def pages_cmd(argv: Sequence[str]) -> int:
    """Rasterise a golden set, and say what it could not do.

    A page that cannot be rendered is a page missing from every model's score,
    so this exits non zero and names each one rather than reporting a round
    number of successes.
    """
    args = _pages_parser().parse_args(list(argv))
    try:
        corpus = corpuslib.root(args.corpus)
        ids = pageimages.read_ids(args.set_name)
    except (NoCorpus, FileNotFoundError) as err:
        print(f"{PROG}: {err}", file=sys.stderr)
        return 2

    built = pageimages.build(ids, corpus, dpi=args.dpi, overwrite=args.overwrite)
    line = f"{len(built.made)} rendered, {len(built.had)} already there, {len(built.failed)} failed"
    if built.redone:
        line += f", {len(built.redone)} of them replacing an image at another dpi"
    print(line)
    for page_id, why in built.failed:
        print(f"{page_id}: {why}", file=sys.stderr)
    return 1 if built.failed else 0


def _mine_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} mine",
        description="Mine training candidates out of adjudicated disagreements.",
    )
    parser.add_argument("root", type=Path, help="a directory of readings and their sidecars")
    parser.add_argument("--jsonl", type=Path, default=None, help="write the pairs here")
    parser.add_argument("--markdown", type=Path, default=None, help="write the summary here")
    parser.add_argument(
        "--caught",
        action="store_true",
        help="report what the referee found that the acceptance rules did not",
    )
    return parser


def mine_cmd(argv: Sequence[str]) -> int:
    """Turn a directory of sidecars into labelled pairs, and say how many.

    Reads only. Running it twice on the same output produces the same pairs,
    which is what makes it safe to point at a tree that a batch is still
    writing into.
    """
    from local_ocr import mine as mining
    from local_ocr import sidecar as sidecarlib
    from local_ocr.second import caught

    args = _mine_parser().parse_args(list(argv))
    if not args.root.is_dir():
        print(f"{PROG}: {args.root} is not a directory", file=sys.stderr)
        return 2

    if args.caught:
        records = []
        for path in mining.sidecars(args.root):
            try:
                records.append(sidecarlib.load(path))
            except (OSError, ValueError):
                continue
        shape = caught(records)
        print(
            f"{shape['pages']} pages, {shape['passed_gates']} passed every rule, "
            f"{shape['disagreed']} disagreed, {shape['caught']} caught, "
            f"{shape['high']} of high severity"
        )
        return 0

    candidates = mining.mine(args.root)
    if args.jsonl is not None:
        args.jsonl.parent.mkdir(parents=True, exist_ok=True)
        args.jsonl.write_text(mining.to_jsonl(candidates), encoding="utf-8")
        print(f"{len(candidates)} candidates, written to {args.jsonl}", file=sys.stderr)
    text = mining.report(candidates)
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
        return 0
    print(text, end="")
    return 0


def _serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} serve",
        description="Start a shortlisted reader, or print the command line that would.",
    )
    parser.add_argument("name", nargs="?", default="", help="an entry in models.toml")
    parser.add_argument("--list", action="store_true", help="what the shortlist holds")
    parser.add_argument(
        "--print",
        dest="dry",
        action="store_true",
        help="print the command line and do not start anything",
    )
    parser.add_argument("--vllm", default="vllm", help="the binary, for a venv that is not on PATH")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="FLAGS",
        help="vLLM flags appended after the entry's own, for a benchmark sweep",
    )
    return parser


def serve_cmd(argv: Sequence[str], exec_: Callable[[str, list[str]], None] | None = None) -> int:
    """Hand the process over to vLLM, rather than supervise it.

    `os.execvp` replaces this process, so what systemd watches is the server
    itself. A Python parent would be one more thing to get the signal handling
    wrong in, and `Restart=on-failure` would then be restarting a wrapper whose
    child had already gone. There is nothing here to keep running after the
    command line is built, so nothing here keeps running.
    """
    args = _serve_parser().parse_args(list(argv))

    if args.list:
        for entry in serving.load().values():
            pinned = "" if entry.pinned else ", unpinned"
            print(f"{entry.name}: {entry.repo} on {entry.port}{pinned}")
            for line in entry.what.split("\n"):
                print(f"    {line}")
        return 0

    if not args.name:
        print(f"{PROG}: serve needs a model name, or --list", file=sys.stderr)
        return 2

    try:
        entry = serving.model(args.name)
    except (serving.NoSuchModel, OSError) as err:
        print(f"{PROG}: {err}", file=sys.stderr)
        return 2

    if not entry.pinned:
        # Not fatal. reader-b and reader-c are unpinned today because their
        # repositories publish no revision worth pinning to yet, and refusing to
        # serve them would mean not measuring them at all. Said out loud so the
        # report that follows is read with that in mind.
        print(
            f"{PROG}: {entry.name} is on {entry.revision}, so a report of this run "
            "cannot be reproduced from the report alone",
            file=sys.stderr,
        )

    extra = [word for flags in args.extra for word in shlex.split(flags)]
    if extra:
        # Loud on purpose. A sweep starts the server with something the
        # shortlist does not say, and the log of the run has to carry that or
        # the number it produces is attributed to the wrong configuration.
        print(f"{PROG}: {entry.name} with {shlex.join(extra)} appended", file=sys.stderr)

    command = entry.command(args.vllm, extra)
    if args.dry:
        print(entry.shell(args.vllm, extra))
        return 0

    (exec_ or os.execvp)(command[0], command)
    return 1  # unreachable when execvp succeeds, which is the point of it


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
    if command == "eval":
        return evaluate_cmd(rest)
    if command == "golden":
        return golden_cmd(rest)
    if command == "serve":
        return serve_cmd(rest)
    if command == "pages":
        return pages_cmd(rest)
    if command == "mine":
        return mine_cmd(rest)
    print(f"{PROG}: no command {command!r}\n\n{_usage()}", file=sys.stderr)
    return 2


def _usage() -> str:
    return (
        f"usage: {PROG} <command> [options]\n"
        "\n"
        "commands:\n"
        "  ocr-batch <in> <out>   read a directory of page images into Markdown\n"
        "  eval --set S --readings D   judge readings against a golden set\n"
        "  golden draw|check|show      the four golden sets and their drift\n"
        "  serve <model>          start a shortlisted reader under vLLM\n"
        "  pages --set S          rasterise a golden set into the corpus images tree\n"
        "  mine <dir>             training pairs out of the readers' disagreements\n"
        "\n"
        "A second reader is switched on with LOCAL_OCR_REFEREE, naming an entry in\n"
        "models.toml. It reads every page after the primary does, the two readings\n"
        "are compared, and a disagreement is settled by sending the disputed strip\n"
        "back cropped. LOCAL_OCR_BUDGET caps that at three adjudications a page.\n"
        "\n"
        f"{PROG} ocr-batch is a drop-in for the chatgpt-tool subcommand of the same\n"
        "name, so that the Bourbaki fleet can drive this machine with no change to\n"
        "its own transport. See spec 2028 section 04.\n"
        "\n"
        "eval needs a checkout of tamnd/bourbaki, found through BOURBAKI_CORPUS or\n"
        "given with --corpus. Nothing in this repository copies pages out of it.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
