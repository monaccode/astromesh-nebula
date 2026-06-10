#!/usr/bin/env bash
# Full local QLoRA run for Centinela-4B: ALL 31k examples, 2 epochs (~24h on RTX 2060).
# Resumable: set RESUME=1 to continue from the latest checkpoint after an interruption.
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export HF_HOME="$HOME/.cache/huggingface"
cd /mnt/d/monaccode/astromesh-nebula
PY=.venv-wsl/bin/python

echo "=== preflight: training stack intact? ==="
"$PY" - <<'PY'
import torch, transformers, trl, peft, bitsandbytes, datasets
assert torch.cuda.is_available(), "CUDA not available!"
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "trl", trl.__version__,
      "peft", peft.__version__, "bitsandbytes", bitsandbytes.__version__)
PY
if [ $? -ne 0 ]; then echo "PREFLIGHT FAILED"; exit 1; fi

RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi

echo "=== train: ALL examples, 2 epochs -> export/adapter ${RESUME_FLAG} ==="
"$PY" scripts/02_train_local.py \
  --config configs/train-4b.yaml \
  --train data/centinela/train.jsonl \
  --out export/adapter \
  --epochs 2 \
  $RESUME_FLAG
rc=$?
echo "=== train exit code: $rc ==="
if [ $rc -eq 0 ]; then ls -la export/adapter; fi
exit $rc
