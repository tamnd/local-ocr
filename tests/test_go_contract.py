"""The command line `ocr/batch.go` builds, run for real.

This file is the one that decides whether the integration works. It does not
import anything from `local_ocr`: it builds the shell command the way
`Batch.start` builds it, hands it to `sh -c`, and then behaves like `wait()`,
`missing()` and `tail()` do.

The Go original, for reference:

    command := fmt.Sprintf(
        "%s %s ocr-batch %s %s -j %d --rate-delay %s --ext png "+
            "--skip-existing --recursive --timeout %d --prompt \\"$(cat %s)\\" "+
            ">%s 2>&1 </dev/null & echo $!",
        b.launcher(), quote(b.Host.Tool), quote(in), quote(out),
        b.Host.lanes(), strconv.FormatFloat(b.Host.rateDelay(), 'f', -1, 64),
        int(b.Host.pageTimeout().Seconds()), quote(prompt), quote(logFile))
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# The prompt the fleet actually sends is 1400 tokens of house rules with LaTeX in
# it. The dollar signs are the point: command substitution output is not
# re-expanded by the shell, so they must arrive at the tool as themselves.
PROMPT = """Transcribe this page of Bourbaki verbatim.

The standard rings are $\\mathbf{Z}$, $\\mathbf{Q}$, $\\mathbf{R}$ and never $\\mathbb{Z}$.
A statement head is bold and keeps its spaced em rule.
A passage that forward references is fenced with \\* and never a bare asterisk.
What cannot be read is the illegible marker and never a guess.
An inline formula is $x \\in E$ and a display is on its own lines.
"""


def quote(text: str) -> str:
    """`quote` from batch.go: single quotes, with the usual escape inside them."""
    return "'" + text.replace("'", "'\\''") + "'"


def shim(tmp_path: Path) -> Path:
    """The wrapper spec 2028 section 04 puts on the reading machine.

    On gamingpc this lives at `$HOME/chatgpt-tool/.venv/bin/chatgpt-tool`,
    because that is the second branch of the probe script's tool discovery and
    it does not depend on what PATH looks like in a non interactive ssh session.
    """
    path = tmp_path / "chatgpt-tool"
    path.write_text(f'#!/bin/sh\nexec {sys.executable} -m local_ocr.cli "$@"\n')
    path.chmod(0o755)
    return path


def start(tmp_path: Path, *, lanes: int = 2, rate_delay: float = 0.0, timeout: int = 900) -> dict:
    """Do to a directory of pages exactly what `Batch.Run` does."""
    home = tmp_path
    root = "bourbaki-ocr"
    batch_id = "ens-i-iv-0001"
    src = f"{root}/in/{batch_id}"
    dst = f"{root}/out/{batch_id}"
    log = f"{root}/logs/{batch_id}.log"
    prompt_file = f"{root}/prompt-deadbeef.md"

    # prepare(): the answers directory is emptied rather than reused, because
    # what the poll counts is the files in it.
    for name in (src, dst, f"{root}/logs"):
        (home / name).mkdir(parents=True, exist_ok=True)
    (home / prompt_file).write_text(PROMPT)

    for n in range(1, 5):
        (home / src / f"{n:04d}.png").write_bytes(b"\x89PNG pretend")
    # --recursive is always sent, so a subdirectory has to come back mirrored.
    (home / src / "plates").mkdir(exist_ok=True)
    (home / src / "plates" / "0009.png").write_bytes(b"\x89PNG pretend")
    # --ext png is always sent, so this must be left alone.
    (home / src / "notes.txt").write_text("not a page")

    tool = shim(tmp_path)
    launcher = "nohup"  # Host.Local() on the Go side; a rented box adds DISPLAY and setsid.
    command = (
        f"{launcher} {quote(str(tool))} ocr-batch {quote(src)} {quote(dst)} "
        f"-j {lanes} --rate-delay {rate_delay:g} --ext png "
        f"--skip-existing --recursive --timeout {timeout} "
        f'--prompt "$(cat {quote(prompt_file)})" >{quote(log)} 2>&1 </dev/null & echo $!'
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["LOCAL_OCR_BACKEND"] = "echo"

    # There is no cd in the remote command. Every path is relative to the login
    # home, which is where an ssh command starts and where rsync puts the
    # images, so the working directory is the whole of the resolution rule.
    began = time.monotonic()
    done = subprocess.run(
        ["sh", "-c", command],
        cwd=home,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "home": home,
        "src": home / src,
        "dst": home / dst,
        "log": home / log,
        "stdout": done.stdout,
        "returned_in": time.monotonic() - began,
        "images": 5,
    }


def wait(out: Path, pid: int, seconds: float = 30.0) -> int:
    """`wait()` from batch.go, in the same two facts per poll."""
    deadline = time.monotonic() + seconds
    count = 0
    while time.monotonic() < deadline:
        count = len([p for p in out.iterdir() if p.name.endswith(".md")]) if out.exists() else 0
        alive = True
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
        if count >= 5 or not alive:
            return count
        time.sleep(0.1)
    return count


def test_the_command_returns_a_pid_and_does_not_hold_the_channel(tmp_path: Path) -> None:
    """`start` reads the last line of the output and calls Atoi on it.

    And it must come back promptly: ssh will not return while a backgrounded
    process still holds the channel open, which is why the tool's own output
    goes to a log file and its input comes from /dev/null.
    """
    run = start(tmp_path)
    last = run["stdout"].strip().splitlines()[-1]
    pid = int(last)
    assert pid > 0
    assert run["returned_in"] < 15, "the command held the ssh channel open"
    wait(run["dst"], pid)


def test_the_tree_comes_back_mirrored_with_the_extensions_swapped(tmp_path: Path) -> None:
    run = start(tmp_path)
    pid = int(run["stdout"].strip().splitlines()[-1])
    wait(run["dst"], pid)

    got = sorted(str(p.relative_to(run["dst"])) for p in run["dst"].rglob("*") if p.is_file())
    assert got == [
        "0001.md",
        "0002.md",
        "0003.md",
        "0004.md",
        os.path.join("plates", "0009.md"),
    ], got
    # notes.txt was not a page and produced nothing.
    assert not (run["dst"] / "notes.md").exists()


def test_the_prompt_arrives_byte_for_byte_but_for_its_trailing_newline(tmp_path: Path) -> None:
    """--prompt "$(cat file)" is one argument and the shell does not re-expand it.

    So every dollar sign of the LaTeX in the corpus prompt arrives as itself,
    which is the whole reason the Go side spells it that way.

    It is not quite an identity. Command substitution strips trailing newlines,
    measured here as 394 bytes arriving for a 395 byte file, so the prompt the
    model sees is the file with its final newline removed. That is harmless and
    it is worth pinning, because the prompt is hashed for provenance and a hash
    taken on the wrong side of that strip would never match.
    """
    run = start(tmp_path)
    pid = int(run["stdout"].strip().splitlines()[-1])
    wait(run["dst"], pid)
    sent = PROMPT.rstrip("\n")
    digest = hashlib.sha256(sent.encode("utf-8")).hexdigest()
    body = (run["dst"] / "0001.md").read_text()
    assert f"prompt {digest} {len(sent)} bytes" in body, body


def test_skip_existing_leaves_a_page_the_run_already_has(tmp_path: Path) -> None:
    run = start(tmp_path)
    pid = int(run["stdout"].strip().splitlines()[-1])
    wait(run["dst"], pid)
    (run["dst"] / "0001.md").write_text("read on the first pass\n")
    stamp = (run["dst"] / "0002.md").stat().st_mtime_ns

    second = start(tmp_path)  # same tmp_path, so the same directories
    pid = int(second["stdout"].strip().splitlines()[-1])
    wait(second["dst"], pid)
    assert (run["dst"] / "0001.md").read_text() == "read on the first pass\n"
    assert (run["dst"] / "0002.md").stat().st_mtime_ns == stamp


def test_an_unknown_argument_is_named_and_fatal(tmp_path: Path) -> None:
    """Accept and ignore nothing silently.

    A future change on the Go side then fails on the first batch rather than
    reading a thousand pages with the wrong settings.
    """
    tool = shim(tmp_path)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["LOCAL_OCR_BACKEND"] = "echo"
    (tmp_path / "in").mkdir()
    done = subprocess.run(
        [
            str(tool),
            "ocr-batch",
            str(tmp_path / "in"),
            str(tmp_path / "out"),
            "--reasoning",
            "high",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode != 0
    assert "--reasoning" in done.stderr


def test_the_batch_dies_entirely_under_a_process_group_kill(tmp_path: Path) -> None:
    """`stop` sends kill -TERM -<pid> then kill -KILL -<pid>.

    The negative pid is the whole point, so nothing may escape the group. On a
    rented box `setsid` makes the pid a group leader; here the group is the
    shell's own, so this asserts the weaker and still necessary property that
    the process is reachable and dies.
    """
    run = start(tmp_path, lanes=1, rate_delay=5.0, timeout=900)
    pid = int(run["stdout"].strip().splitlines()[-1])
    os.kill(pid, 15)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
    pytest.fail("the batch survived a TERM")


@pytest.mark.skipif(shutil.which("setsid") is None, reason="no setsid, which is macOS")
def test_the_remote_launcher_form_also_starts(tmp_path: Path) -> None:
    """On a rented box the launcher is `DISPLAY=<display> setsid nohup`."""
    tool = shim(tmp_path)
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "0001.png").write_bytes(b"\x89PNG")
    (tmp_path / "prompt.md").write_text(PROMPT)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["LOCAL_OCR_BACKEND"] = "echo"
    command = (
        f"DISPLAY={quote(':99')} setsid nohup {quote(str(tool))} ocr-batch "
        f"{quote('in')} {quote('out')} -j 1 --rate-delay 0 --ext png "
        f"--skip-existing --recursive --timeout 900 "
        f'--prompt "$(cat {quote("prompt.md")})" >{quote("run.log")} 2>&1 </dev/null & echo $!'
    )
    done = subprocess.run(
        ["sh", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    pid = int(done.stdout.strip().splitlines()[-1])
    assert pid > 0
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if (tmp_path / "out" / "0001.md").exists():
            return
        time.sleep(0.1)
    pytest.fail(f"nothing came back: {(tmp_path / 'run.log').read_text()}")
