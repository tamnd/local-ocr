"""Readers, one per way of getting text out of a page image.

A model is a name, a serving command and an adapter. Adding a candidate is one
file and one line here, which is what makes the re-evaluation rule in spec 2028
section 02 practical rather than aspirational: that survey has a shelf life of
weeks, and the thing that outlives it is the ability to swap and measure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_ocr.batch import Reader


def build(name: str, *, model: str, base_url: str) -> Reader:
    if name in {"vllm", "openai", "sglang"}:
        # One adapter for all three. vLLM and SGLang both serve the OpenAI
        # vision API, which is the entire reason section 01 can propose
        # benchmarking one against the other without a rewrite.
        from local_ocr.backends.openai_vision import OpenAIVisionReader

        return OpenAIVisionReader(model=model, base_url=base_url)
    if name == "codex":
        # The local ChatGPT subscription. No URL and no VRAM, which is why it
        # is the one referee on the M6 shortlist that can run beside reader A.
        from local_ocr.backends.codex import MODEL, CodexReader

        return CodexReader(model=model or MODEL)
    if name == "echo":
        from local_ocr.backends.echo import EchoReader

        return EchoReader()
    raise SystemExit(f"local-ocr: no backend {name!r}")
