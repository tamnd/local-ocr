"""The style LoRA, as a recipe and a dataset rather than as a training loop.

§08 asks for a small adapter that teaches the reader the house conventions, and
it names one setting that will cost a week if it is got wrong. This module is
that setting written down where a test can read it, plus the two steps either
side of the training run: turning the pool into what the trainer reads, and
writing down afterwards what was trained on what.

The training loop itself is four lines of Unsloth and is not worth wrapping. It
runs on the card, it needs CUDA, and it cannot run in CI or on this laptop. What
can run anywhere, and therefore what is here, is everything that decides whether
the run was a valid one.

## The setting that costs a week

`finetune_vision_layers = False`. It is not a quality judgement, it is a serving
constraint: vLLM does not serve LoRA adapters on vision layers. An adapter that
touches them has to be merged into the base weights and served as a full model,
which costs a full set of weights per experiment and destroys the ability to
hold one base resident and switch adapters against it. The whole of §01's VRAM
budget assumes the second thing.

For a style adapter it costs nothing anyway, because the conventions are output
formatting and formatting lives in the language layers. `check` refuses a recipe
that turns it on rather than leaving it to whoever reads the config.

## Where the prompt is attached

Here, and not in the pool. The pool holds a completion and an image path and no
prompt, for the reasons `pool.py` gives, so the prompt joins the example on the
way into the trainer. Which prompt a run used is recorded in the run card by
digest, because two adapters trained on the same pages under two prompts are two
different experiments and the only thing that says so afterwards is the card.

§08 trains on the completion only. The prompt is 1 400 identical tokens on every
example and training on them teaches the model to recite its own instructions,
which it already does perfectly and which nobody asked for.

## Why the pool is checked again on the way in

`pool.build` asserts that no held out page is in what it produced. This reads a
file that was produced by that build, possibly on another machine, possibly
weeks ago, possibly edited by hand to add a few pages somebody thought would
help. The manifests are right here and checking against them costs nothing, so
the file is not taken on trust.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from local_ocr import golden
from local_ocr import pool as poollib

# Unsloth's own default, and the one §08 quotes. Rank 16 because this is a style
# adjustment on top of a competent model rather than a domain transplant, and
# the loss is what says to go higher.
RANK = 16


class Refused(Exception):
    """A recipe or a dataset that must not be trained, and why."""


@dataclass(frozen=True)
class Recipe:
    """Every setting a run of this is allowed to vary, and the one it is not."""

    base: str
    """The entry in models.toml the adapter sits on top of."""

    revision: str
    """The base weights, pinned. An adapter is only meaningful against the
    weights it was fitted to, and `main` is a moving branch."""

    finetune_vision_layers: bool = False
    """Never true. See the module docstring: vLLM cannot serve it."""

    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = RANK
    lora_alpha: int = RANK
    completion_only: bool = True
    """Mask the prompt out of the loss. §08 is explicit about this."""

    temperature: float = 0.0
    """Every evaluation of every checkpoint, so that two checkpoints differ by
    their weights and by nothing else."""

    def peft_kwargs(self) -> dict[str, object]:
        """What `FastVisionModel.get_peft_model` is called with."""
        return {
            "finetune_vision_layers": self.finetune_vision_layers,
            "finetune_language_layers": self.finetune_language_layers,
            "finetune_attention_modules": self.finetune_attention_modules,
            "finetune_mlp_modules": self.finetune_mlp_modules,
            "r": self.r,
            "lora_alpha": self.lora_alpha,
        }


def check(recipe: Recipe) -> None:
    """Refuse a recipe that cannot be served, before the card is warmed up."""
    if recipe.finetune_vision_layers:
        raise Refused(
            "finetune_vision_layers is on. vLLM does not serve LoRA adapters on "
            "vision layers, so this adapter would have to be merged into the base "
            "and served as a full set of weights, which is a different deployment "
            "and a different VRAM budget. See spec 2028 section 08."
        )
    if recipe.revision in ("", "main"):
        raise Refused(
            "the base revision is not pinned. An adapter is fitted to one set of "
            "weights and main is a moving branch, so a run against it cannot be "
            "reproduced from its own run card."
        )
    if recipe.r < 1 or recipe.lora_alpha < 1:
        raise Refused("rank and alpha are both positive or there is no adapter")
    if recipe.temperature != 0.0:
        raise Refused(
            "sampling is greedy for every evaluation of every checkpoint. A "
            "nonzero temperature makes two checkpoints differ by the dice."
        )


def digest(text: str) -> str:
    """The sha256 of a prompt, which is what a run card records instead of it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_pool(path: Path) -> list[poollib.Example]:
    """The pool off disk, checked against the manifests again on the way in."""
    out: list[poollib.Example] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            got = json.loads(line)
            out.append(
                poollib.Example(
                    id=got["id"],
                    book=got["book"],
                    image=got["image"],
                    completion=got["completion"],
                    split=got.get("split", "train"),
                )
            )
        except (ValueError, KeyError) as err:
            raise Refused(f"{path}:{number} is not a pool line: {err}") from err
    poollib.check(poollib.Pool(examples=out))
    return out


def conversation(example: poollib.Example, prompt: str) -> dict[str, object]:
    """One example in the chat shape Unsloth's vision trainer reads.

    The image goes in as a path rather than as bytes. The trainer loads it, the
    pool is 2 500 pages of 300 dpi PNG, and a dataset file with the pages inline
    would be gigabytes of base64 that no person could read and no diff could
    show anything useful about.
    """
    return {
        "id": example.id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example.image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example.completion}],
            },
        ],
    }


def dataset(examples: list[poollib.Example], prompt: str, split: str = "train") -> str:
    """The pool as conversations, one a line, for one side of the split."""
    wanted = [one for one in examples if one.split == split]
    if not wanted:
        raise Refused(f"the pool has no {split} pages in it")
    return "".join(
        json.dumps(conversation(one, prompt), ensure_ascii=False, sort_keys=True) + "\n"
        for one in wanted
    )


@dataclass
class Run:
    """What a training run was, in the terms a later report has to quote."""

    recipe: Recipe
    prompt_sha256: str
    train: int
    val: int
    books: dict[str, int] = field(default_factory=dict)
    dev_pages: int = 0

    def card(self) -> str:
        """The Markdown card, written beside the adapter.

        Beside it rather than in a log, because the question this answers is
        asked six months later by somebody holding an adapter directory and
        wondering whether the number next to it means anything.
        """
        out = ["# Training run", ""]
        out.append(
            f"{self.train} pages trained on, {self.val} held back for the loss curve, "
            f"on `{self.recipe.base}` at revision `{self.recipe.revision}`."
        )
        out.append("")
        out.append("| Setting | Value |")
        out.append("| --- | --- |")
        for key, value in asdict(self.recipe).items():
            out.append(f"| `{key}` | `{value}` |")
        out.append(f"| `prompt_sha256` | `{self.prompt_sha256}` |")
        out.append("")
        if self.books:
            out.append("| Volume | Pages |")
            out.append("| --- | --- |")
            for book in sorted(self.books):
                out.append(f"| {book} | {self.books[book]} |")
            out.append("")
        out.append(
            "`finetune_vision_layers` is false because vLLM does not serve LoRA "
            "adapters on vision layers, and an adapter it cannot serve is an "
            "adapter that has to be merged into the base and shipped as a full "
            "set of weights. That is a different deployment and a different VRAM "
            "budget from the one section 01 sizes."
        )
        out.append("")
        out.append(
            f"{self.dev_pages} of the training pages are `golden-dev` pages, which "
            "section 08 allows and which is why no number reported off `golden-dev` "
            "after this run says anything about how this adapter generalises. "
            "`golden-test` and `golden-hard` are excluded from the pool by id and "
            "the exclusion is checked twice, once where the pool is built and once "
            "where it is read back."
        )
        return "\n".join(out) + "\n"


def plan(examples: list[poollib.Example], recipe: Recipe, prompt: str) -> Run:
    """Everything about a run that can be known before the card is warmed up."""
    check(recipe)
    train = [one for one in examples if one.split == "train"]
    books: dict[str, int] = {}
    for one in train:
        books[one.book] = books.get(one.book, 0) + 1
    # Counted off the manifest rather than off a field in the pool file, so that
    # a pool built before the manifests were redrawn is described by the sets as
    # they are now rather than as they were.
    dev = set(golden.read_manifest("golden-dev"))
    return Run(
        recipe=recipe,
        prompt_sha256=digest(prompt),
        train=len(train),
        val=len(examples) - len(train),
        books=books,
        dev_pages=sum(1 for one in train if one.id in dev),
    )
