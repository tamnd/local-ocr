#!/bin/sh
# Build the serving venv on the reader host.
#
# vLLM is not a dependency of this project and is deliberately not in
# pyproject.toml. It pulls a CUDA build of torch, about 3 GB of wheels chosen by
# the driver on the machine, and putting that in the lock file would make every
# CI run and every laptop checkout resolve it. The reader host installs it here
# instead, pinned, and the version below is the one the numbers in the issues
# were measured with.
#
# The cost of keeping it out of the project is that `uv sync` on this host will
# remove it, because sync makes the environment match the lock file exactly.
# That failure is worth describing because it does not look like itself. The
# running server keeps answering, because Python had already imported what it
# needed, and then every request comes back 400 "cannot identify image file":
# Pillow loads its PNG plugin lazily, on the first image, and by then the files
# are gone. Two hundred pages were refused that way before anybody looked at
# what was actually installed.
#
# So: on this host, `uv sync --inexact`, or this script.
set -e
cd "$(dirname "$0")/.."

VLLM=${VLLM:-0.27.1}
export UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu130}

test -d .venv || uv venv
uv pip install --python .venv/bin/python "vllm==$VLLM"
uv pip install --python .venv/bin/python -e .

.venv/bin/python - <<'PY'
import torch
import vllm

print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
