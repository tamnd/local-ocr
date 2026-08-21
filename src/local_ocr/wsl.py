"""Starting a reader on a card that sits behind Windows.

gamingpc is the only 4090 this project has and vLLM reaches it from inside
WSL2. That is not the same machine as a Linux box with the same card in it, and
one of the differences stops a reader from starting at all.

vLLM refuses to pin host memory under WSL. `is_pin_memory_available` tests
`in_wsl()` first and returns False when it is true, pointing at NVIDIA's own
list of known limitations for Linux CUDA applications under WSL. Most of vLLM
copes with that by copying through unpinned memory and losing a little
overlap.

The V2 model runner does not cope with it. It stages every request's token ids
through a `UvaBuffer`, and that constructor calls `is_uva_available()`, which
is `is_pin_memory_available() or current_platform.is_cpu()`. On this box both
are false, so the buffer raises while the engine core is still being built,
before a single weight is read:

    RuntimeError: UVA is not available
      vllm/v1/worker/gpu/buffer_utils.py:47   UvaBuffer.__init__
      vllm/v1/worker/gpu/states.py:34         __init__
      vllm/v1/worker/gpu/model_runner.py      construction
      vllm/v1/engine/core.py:1074             __init__

What the server prints at the end is `Engine core initialization failed. See
root cause above. Failed core proc(s): {}`, and the root cause is forty frames
above the line that says so. That is the whole reason this module exists with a
docstring this long: the failure names neither WSL nor pinned memory nor the
model runner, and the next person to hit it will be reading a log at three in
the morning.

Which readers it hits is not a property of the card. `use_v2_model_runner`
picks V2 for any generate model whose architecture is on vLLM's default V2
list, unless something else in the config is unsupported there. reader-a is
FP8, lands on V1 for its own reasons and has served on this box for weeks.
reader-b, on 2026-08-22, was the first entry on the shortlist that did not, and
it died in under a minute every time.

`VLLM_USE_V2_MODEL_RUNNER` is read before any of that reasoning happens and
short circuits all of it, so setting it to 0 puts every reader on the V1
runner, which has no UVA buffer in it. The cost is written down rather than
hidden: V2 is where vLLM's own work is going, and a throughput number measured
on V1 is a number for the older path. It is the only path this box has, so
every Russian bake off number is a V1 number and the report has to say so.

The variable is only set when nothing has set it. Somebody sweeping V1 against
V2 on a real Linux box types it and means it, and a wrapper that overrules them
is worse than one that does nothing at all.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Callable, Mapping, Sequence

RUNNER = "VLLM_USE_V2_MODEL_RUNNER"


def in_wsl(*, uname: Callable[[], Sequence[object]] = platform.uname) -> bool:
    """The same test vLLM makes, so this predicts what vLLM is about to decide.

    vLLM's own `in_wsl` joins the fields of `platform.uname()` and looks for
    microsoft in the result, case folded. Repeating the test here rather than
    importing it is deliberate. This has to answer the question on a laptop
    with no vLLM installed and no CUDA in sight, and the answer there is no.
    """
    return "microsoft" in " ".join(str(part) for part in uname()).lower()


def additions(
    env: Mapping[str, str] | None = None, *, wsl: bool | None = None
) -> tuple[dict[str, str], list[str]]:
    """What has to be in the environment before vLLM starts here, and why.

    Returns only what is being added, not the whole environment, so a caller
    can put the additions in front of a printed command line and hand it to
    somebody to paste. An empty mapping is the ordinary case and means this box
    needs nothing said to it.

    The notes come back rather than being printed, because this is a library
    and the CLI owns stderr.
    """
    here = in_wsl() if wsl is None else wsl
    base = os.environ if env is None else env
    if not here:
        return {}, []
    if RUNNER in base:
        return {}, [f"{RUNNER} is already {base[RUNNER]} here, so this leaves it alone"]
    return {RUNNER: "0"}, [
        "WSL cannot pin host memory, so the V2 model runner cannot allocate its "
        f"UVA buffers and would fail before loading weights; {RUNNER}=0 puts this "
        "reader on the V1 runner and every number from it is a V1 number"
    ]
