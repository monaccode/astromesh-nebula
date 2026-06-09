#!/usr/bin/env bash
# Full local QLoRA run for Centinela-4B: ~6k stratified examples, 1 epoch (~2h on RTX 2060).
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export HF_HOME="$HOME/.cache/huggingface"
cd /mnt/d/monaccode/astromesh-nebula
PY=.venv-wsl/bin/python

echo "=== train: 6000 examples, 1 epoch -> export/adapter ==="
"$PY" scripts/02_train_local.py \
  --config configs/train-4b.yaml \
  --train data/centinela/train.jsonl \
  --out export/adapter \
  --max-train-samples 6000 \
  --epochs 1
rc=$?
echo "=== train exit code: $rc ==="
if [ $rc -eq 0 ]; then ls -la export/adapter; fi
exit $rc
