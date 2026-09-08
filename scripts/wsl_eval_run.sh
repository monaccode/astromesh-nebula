#!/usr/bin/env bash
# Lean eval re-run (venv already restored, eval-small.jsonl already written).
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export HF_HOME="$HOME/.cache/huggingface"
cd /mnt/d/monaccode/astromesh-nebula
export PYTHONPATH="$PWD"   # make the nebula package importable
PY=.venv-wsl/bin/python

echo "=== evaluate base vs tuned (gate) ==="
"$PY" scripts/05_evaluate.py --base Qwen/Qwen3-4B --tuned export/merged \
  --eval data/centinela/eval-small.jsonl --out eval/results/results.md
echo "=== eval rc=$? ==="
echo "--- results.md ---"
cat eval/results/results.md
