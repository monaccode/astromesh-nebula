#!/usr/bin/env bash
# Evaluate base vs tuned (merged) on a 1000-example subsample, with the tier gate.
# Reuses the already-produced export/merged. Restores the CUDA training stack first
# (the GGUF step polluted the venv with cpu torch + a buggy transformers).
set -uo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin"
export HF_HOME="$HOME/.cache/huggingface"
cd /mnt/d/monaccode/astromesh-nebula
PY=.venv-wsl/bin/python

echo "=== restore CUDA torch 2.10 + transformers 5.5 ==="
uv pip install --python "$PY" "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$PY" "transformers==5.5.0"

echo "=== verify cuda ==="
"$PY" -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'tf',transformers.__version__)"

echo "=== subsample eval to 1000 ==="
"$PY" - <<'PY'
import random
random.seed(42)
lines = [ln for ln in open("data/centinela/eval.jsonl", encoding="utf-8") if ln.strip()]
random.shuffle(lines)
sub = lines[:1000]
with open("data/centinela/eval-small.jsonl", "w", encoding="utf-8") as f:
    f.writelines(sub)
print(f"wrote {len(sub)} eval examples")
PY

echo "=== evaluate base vs tuned (gate) ==="
"$PY" scripts/05_evaluate.py --base Qwen/Qwen3-4B --tuned export/merged \
  --eval data/centinela/eval-small.jsonl --out eval/results/results.md
echo "=== eval rc=$? ==="
echo "--- results.md ---"
cat eval/results/results.md
