"""Turning a name into a vLLM command line.

The whole of the serving configuration is `models.toml` and the whole of this
module is reading it. That split is deliberate and it is what §02's shelf life
paragraph is about: the survey it came from will be out of date in weeks, so a
candidate has to be an entry in a file rather than a patch to a program.

Three things are decided here rather than in the unit file.

**The revision is pinned and always passed.** A model repository is a moving
branch. A reading is only comparable to another reading of the same weights, so
a report that names `reader-a` and no revision is a report about nothing in
particular. `main` is allowed, and it is recorded as `main`, which at least says
plainly that the run cannot be reproduced from the report alone.

**The served name is the entry name and not the repository.** `--served-model-
name reader-a` means the batch path asks for `reader-a` and gets whatever this
file currently points at, which is how a candidate is swapped without touching
anything that calls it. The page front matter records the resolved name, not
this one, because the front matter has to say which weights read the page.

**Sampling is greedy and is not a flag here.** It belongs to the request rather
than the server, and it is not negotiable: this is transcription, there is
nothing to be creative about, and a nonzero temperature turns an already hard
formula into a lottery.
"""

from __future__ import annotations

import shlex
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

MODELS = Path(__file__).with_name("models.toml")

# What a reader listens on when its entry names no port. Not 8000: vLLM's own
# default is 8000 and a stray hand started server on the same box would take
# the batch path's requests without anybody noticing.
DEFAULT_PORT = 8801


class NoSuchModel(KeyError):
    """The name is not in models.toml, with the ones that are."""


@dataclass(frozen=True)
class Model:
    """One entry of the shortlist."""

    name: str
    repo: str
    revision: str
    port: int
    what: str
    args: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pinned(self) -> bool:
        """Whether a run of this entry can be reproduced from a report of it."""
        return self.revision not in ("", "main")

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def command(self, binary: str = "vllm", extra: Sequence[str] = ()) -> list[str]:
        """The whole command line, in the order a person would read it.

        `extra` goes last, after the entry's own flags, and vLLM's parser takes
        the last occurrence of a repeated option. That is what makes a sweep
        possible without six near identical entries in the shortlist: the
        benchmark starts `reader-a` with `--max-num-seqs 32` appended and every
        other flag is the one the report will name. It is for measuring. A
        setting that wins a sweep is edited into the entry rather than left
        living in whatever shell ran it.
        """
        return [
            binary,
            "serve",
            self.repo,
            "--revision",
            self.revision,
            "--served-model-name",
            self.name,
            "--port",
            str(self.port),
            *self.args,
            *extra,
        ]

    def shell(self, binary: str = "vllm", extra: Sequence[str] = ()) -> str:
        """The same thing, quoted, for a unit file or a person to paste."""
        return shlex.join(self.command(binary, extra))


def load(path: Path | None = None) -> dict[str, Model]:
    """Every entry, keyed by name."""
    source = path or MODELS
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    out: dict[str, Model] = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{source}: {name} is not a table")
        out[name] = Model(
            name=name,
            repo=str(entry["repo"]),
            revision=str(entry.get("revision", "main")),
            port=int(entry.get("port", DEFAULT_PORT)),
            what=str(entry.get("what", "")).strip(),
            args=tuple(str(arg) for arg in entry.get("args", ())),
        )
    return out


def model(name: str, path: Path | None = None) -> Model:
    """One entry, or an error that says which names exist.

    Named rather than looked up with `.get`, because the failure a person hits
    here is a typo in a systemd unit at three in the morning and the useful
    thing to print is the list.
    """
    entries = load(path)
    if name not in entries:
        known = ", ".join(sorted(entries))
        raise NoSuchModel(f"{name} is not a model in models.toml; there is {known}")
    return entries[name]
