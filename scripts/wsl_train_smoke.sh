#!/usr/bin/env bash
# 2-step smoke test of the local QLoRA path on the RTX 2060 (downloads Qwen3-4B once).
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export HF_HOME="$HOME/.cache/huggingface"
export UNSLOTH_RETURN_LOGITS=0
cd /mnt/d/monaccode/astromesh-nebula
PY=.venv-wsl/bin/python

echo "=== smoke train: 2 steps (standard QLoRA, no unsloth) ==="
"$PY" scripts/02_train_local.py \
  --config configs/train-4b.yaml \
  --train data/centinela/train.jsonl \
  --out export/adapter-smoke \
  --max-steps 2
rc=$?
echo "=== smoke exit code: $rc ==="
if [ $rc -eq 0 ]; then
  echo "=== adapter files ==="
  ls -la export/adapter-smoke
fi
exit $rc
