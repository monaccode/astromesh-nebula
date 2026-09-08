#!/usr/bin/env bash
# Install the QLoRA training stack into the WSL venv and smoke-test imports.
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export HF_HOME="$HOME/.cache/huggingface"
cd /mnt/d/monaccode/astromesh-nebula
PY=.venv-wsl/bin/python

echo "=== install numpy + core training deps (TRL/PEFT/bitsandbytes path) ==="
uv pip install --python "$PY" \
  numpy "transformers>=4.44" "trl>=0.9" "peft>=0.11" "bitsandbytes>=0.43" \
  "datasets>=2.19" accelerate pyyaml "huggingface_hub>=0.24"

echo "=== smoke-test imports ==="
"$PY" - <<'PY'
import torch, transformers, trl, peft, bitsandbytes, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY

echo "=== try unsloth (optional; ok if this fails on bleeding-edge torch) ==="
if uv pip install --python "$PY" unsloth 2>unsloth_install.err; then
  if "$PY" -c "import unsloth; print('unsloth', unsloth.__version__)" 2>unsloth_import.err; then
    echo "UNSLOTH_OK"
  else
    echo "UNSLOTH_IMPORT_FAILED"; tail -20 unsloth_import.err || true
  fi
else
  echo "UNSLOTH_INSTALL_FAILED"; tail -20 unsloth_install.err || true
fi
