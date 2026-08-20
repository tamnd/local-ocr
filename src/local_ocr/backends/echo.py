"""A reader that invents nothing and needs no GPU.

It exists so that the contract tests exercise the real batch machinery, the real
command line and the real atomic write on a machine with no model anywhere near
it. Those are the tests that decide whether the integration works, so they must
run on the laptop and in CI.

It reports the digest of the prompt it was handed, which is how a test proves
that a 1400 token instruction full of LaTeX survived a shell, an ssh, and a
command substitution byte for byte.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EchoReader:
    async def read(self, image: Path, prompt: str) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return f"# {image.stem}\n\nprompt {digest} {len(prompt)} bytes\n"
