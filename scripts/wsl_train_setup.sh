#!/usr/bin/env bash
# Local GPU training setup for WSL2 (Debian) — Astromesh Nebula / Centinela-4B.
# Runs the QLoRA pipeline on the host RTX 2060 via CUDA passthrough, since HF Jobs
# (cloud GPU) requires a paid plan. Publishing to the Hub stays free.
set -euo pipefail

# Normalize PATH (non-login WSL shells from PowerShell drop coreutils/uv).
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export HF_HOME="$HOME/.cache/huggingface"   # keep big model cache on native WSL fs, not /mnt/d

cd /mnt/d/monaccode/astromesh-nebula

echo "=== whoami / env ==="
whoami; echo "HOME=$HOME"; echo "HF_HOME=$HF_HOME"
command -v uv || { echo "uv not found on PATH"; exit 1; }

echo "=== create venv (py3.12) ==="
uv venv .venv-wsl --python 3.12

echo "=== install torch (CUDA build from PyPI) ==="
uv pip install --python .venv-wsl/bin/python torch

echo "=== verify CUDA visibility ==="
.venv-wsl/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
