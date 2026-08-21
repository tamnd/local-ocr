"""The record of how a page came to say what it says.

A page in the corpus is a Markdown file with eight lines of front matter, and
that is the right amount of provenance for a file somebody is going to read.
It is nowhere near enough to answer the questions this project has to answer
later: which reader read this, at what revision, with what sampling, did the
referee agree, what did it disagree about, what was adjudicated and how, and
what did any of it cost.

So the detail goes beside the page rather than in it, as `<page>.ocr.json`. The
front matter stays short and human, the sidecar carries everything, and neither
has to compromise for the other.

## Why this exists rather than a log line

Three things need it, and none of them can be served by a log.

§08 mines fine tuning candidates out of adjudicated disagreements. A training
pair is worth having only if you can say what the two readings were, which one
won and on what evidence, and a grep over last month's log cannot say that.

M4 compares readers, and a comparison is only honest if both sides record the
revision they were served at. `models.toml` pins revisions for exactly this
reason and the sidecar is where the pin ends up next to the output it produced.

And the ordinary case: a page in the corpus turns out to be wrong, six months
from now, and the question is whether the reader got it wrong or the referee
overruled it correctly. That question has one answer and it is in this file.

## Why JSON and not the front matter

The front matter is parsed by the Go side and by anything else that reads the
corpus, and adding fields to it is a compatibility event. The sidecar is read by
this repository and by the miner, is not committed to the corpus, and can grow a
field whenever there is something new worth recording. Different lifetimes, so
different files.

Sidecars are written next to the Markdown in the batch output directory, which
the Go side rsyncs selectively: it pulls `*.md` and leaves everything else, so a
sidecar stays on the reader host unless somebody goes and gets it. That is the
correct default. A sidecar is roughly the size of the page it describes and the
corpus does not need a second copy of every reading.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from local_ocr.batch import write_atomic
from local_ocr.compare import Comparison, Difference

SUFFIX = ".ocr.json"

VERSION = 1
"""Bumped when a field changes meaning rather than when one is added.

The miner reads sidecars written weeks apart and has to know whether a field it
understands means what it used to. Adding a field is safe and does not bump
this; changing what `agreed` counts does.
"""


def sidecar_for(answer: Path) -> Path:
    """Where the sidecar for a page goes.

    `0027.md` gets `0027.ocr.json` and not `0027.json`, so that a directory of
    pages sorts with each sidecar next to its page and so that nothing else
    which writes JSON into an output directory can collide with it.
    """
    return answer.with_name(answer.stem + SUFFIX)


def digest(path: Path) -> str:
    """The sha256 of a file, as the front matter writes it.

    The image hash is the one field here that lets a sidecar be matched back to
    the exact bytes that produced it. A page re rendered at a different dpi has
    a different hash, which is the point: two readings of what looks like the
    same page are only comparable if this matches.
    """
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Read:
    """One reader's attempt at one page."""

    reader: str
    """The name in `models.toml`, as `reader-a`."""

    model: str = ""
    """The repository, as `allenai/olmOCR-2-7B-1025-FP8`."""

    revision: str = ""
    """The pinned commit. A reading without one is not comparable to anything."""

    temperature: float = 0.0
    max_tokens: int = 0
    prompt_sha256: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    text_sha256: str = ""
    refused: str = ""
    """Why this reader produced nothing, or empty when it produced something."""


@dataclass
class Adjudication:
    """One difference, and what was spent resolving it."""

    where: str
    what: str
    first: str
    second: str
    severity: str
    why: str
    step: str
    """Which rung of the ladder settled it: `crop`, `reread`, `illegible`, `budget`."""
    winner: str
    """`first`, `second`, `neither`, or empty when nothing was spent."""
    evidence: str = ""
    seconds: float = 0.0
    score: float | None = None


@dataclass
class Record:
    """Everything known about one page after both readers have had a turn."""

    version: int = VERSION
    page: str = ""
    image_sha256: str = ""
    width: int = 0
    height: int = 0
    dpi: int = 0
    """Inferred from the pixel dimensions, and 0 when it could not be.

    Recorded because it is the field that explains an outlier. A page that took
    four times as long as its neighbours is usually a page rendered at 600.
    """

    first: Read | None = None
    second: Read | None = None
    referee_ran: bool = False
    """False when the referee was absent, which is a supported configuration."""

    agreed: bool = True
    """Whether anything above low severity came up. True when no referee ran,
    since a page nobody argued about has nothing to report."""

    counts: dict[str, int] = field(default_factory=dict)
    adjudicated: list[Adjudication] = field(default_factory=list)
    unadjudicated: int = 0
    """Differences the budget would not pay for. Counted, never hidden."""

    chose: str = "first"
    """Which reading was written: `first`, `second`, or `merged`."""

    gates: dict[str, str] = field(default_factory=dict)
    """The acceptance rules and their verdicts, as `head: ok` or `math: ...`."""

    def to_json(self) -> str:
        return json.dumps(_clean(asdict(self)), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def write(self, answer: Path) -> Path:
        path = sidecar_for(answer)
        write_atomic(path, self.to_json())
        return path


def _clean(value: Any) -> Any:
    """Drop the empty fields, keeping the false ones.

    A sidecar with thirty null fields is harder to read than one with twelve
    real ones, and every field here has a meaningful zero. But `agreed: false`
    and `referee_ran: false` are the two most important facts a sidecar can
    carry, so booleans are kept whatever they are, and so is a zero score.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if isinstance(item, bool):
                out[key] = item
                continue
            cleaned = _clean(item)
            if cleaned in ("", None, [], {}):
                continue
            out[key] = cleaned
        return out
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def load(path: Path) -> Record:
    """Read a sidecar back.

    Unknown fields are dropped rather than raising, because a miner running
    against sidecars written by a newer version should skip what it does not
    understand and still count what it does. `version` is what protects against
    the dangerous case, a field that kept its name and changed its meaning.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    record = Record(version=int(raw.get("version", 0)))
    for name in (
        "page",
        "image_sha256",
        "chose",
    ):
        if isinstance(raw.get(name), str):
            setattr(record, name, raw[name])
    for name in ("width", "height", "dpi", "unadjudicated"):
        if isinstance(raw.get(name), int):
            setattr(record, name, raw[name])
    for name in ("referee_ran", "agreed"):
        if isinstance(raw.get(name), bool):
            setattr(record, name, raw[name])
    for name in ("first", "second"):
        item = raw.get(name)
        if isinstance(item, dict):
            setattr(record, name, _read(item))
    if isinstance(raw.get("counts"), dict):
        record.counts = {k: int(v) for k, v in raw["counts"].items() if isinstance(v, int)}
    if isinstance(raw.get("gates"), dict):
        record.gates = {str(k): str(v) for k, v in raw["gates"].items()}
    for item in raw.get("adjudicated", []) or []:
        if isinstance(item, dict):
            record.adjudicated.append(_adjudication(item))
    return record


def _fields(cls: type) -> set[str]:
    return set(cls.__dataclass_fields__)


def _read(raw: dict[str, Any]) -> Read:
    return Read(**{k: v for k, v in raw.items() if k in _fields(Read)})


def _adjudication(raw: dict[str, Any]) -> Adjudication:
    keep = {k: v for k, v in raw.items() if k in _fields(Adjudication)}
    for name in ("where", "what", "first", "second", "severity", "why", "step", "winner"):
        keep.setdefault(name, "")
    return Adjudication(**keep)


def from_comparison(comparison: Comparison) -> dict[str, int]:
    return comparison.counts()


def as_adjudication(
    difference: Difference,
    *,
    step: str,
    winner: str,
    evidence: str = "",
    seconds: float = 0.0,
) -> Adjudication:
    return Adjudication(
        where=str(difference.where),
        what=difference.what,
        first=difference.first,
        second=difference.second,
        severity=str(difference.severity),
        why=difference.why,
        step=step,
        winner=winner,
        evidence=evidence,
        seconds=seconds,
        score=difference.score,
    )
