"""A stand in for pdftoppm, and the smallest PNG that says what it was rendered at.

Two test modules rasterise pages now, `test_pageimages` for the Bourbaki sets
and `test_kvant` for the Russian one, and both need the same two things: a fake
that writes what pdftoppm would write, and a PNG small enough to write a hundred
of without noticing. Copying them would let the two copies drift, and a fake
that pads its page numbers differently from the real one is a test that passes
on a bug.

This is not a conftest because nothing here is a fixture. It is imported.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def png(path: Path, dpi: int | None, width: int = 8, height: int = 8) -> None:
    """The smallest PNG that says what it was rendered at, or refuses to."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
        )

    out = [
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
    ]
    if dpi is not None:
        metre = round(dpi / 0.0254)
        out.append(chunk(b"pHYs", struct.pack(">IIB", metre, metre, 1)))
    out.append(chunk(b"IDAT", zlib.compress(b"\x00" * (width + 1) * height)))
    out.append(chunk(b"IEND", b""))
    path.write_bytes(b"".join(out))


class Fake:
    """Stands in for pdftoppm, writing what pdftoppm would write."""

    def __init__(self, width: int = 3) -> None:
        self.width = width
        self.commands: list[list[str]] = []

    def run(self, command: list[str]) -> None:
        self.commands.append(command)
        page = int(command[command.index("-f") + 1])
        dpi = int(command[command.index("-r") + 1])
        prefix = Path(command[-1])
        png(prefix.with_name(f"{prefix.name}-{page:0{self.width}d}.png"), dpi)
