#!/usr/bin/env bash
# Full GPU pipeline, run INSIDE a Hugging Face Job (cwd = freshly cloned repo).
# Required env (injected by the launching workflow / HF Jobs):
#   HF_TOKEN  — write token for the HF org (auto-available in HF Jobs)
#   HF_ORG    — target org, e.g. astromesh
#   TAG       — release tag for the model repo (e.g. v0.1)
# The eval step (05) exits non-zero if the tier gate fails, aborting publish.
set -euo pipefail

SIZE="${SIZE:-4b}"
DATASET="NickyNicky/Finance_sentiment_and_topic_classification_Translation_English_to_Spanish_v1"

pip install uv
uv sync --extra eval

# 1) Build dataset (writes data/centinela/stats.json for downstream steps)
uv run python scripts/01_build_dataset.py --dataset "$DATASET" --out data/centinela

# 2) Train QLoRA, 3) merge, 4) quantize
uv run python scripts/02_train.py --config "configs/train-${SIZE}.yaml" --train data/centinela/train.jsonl --out export/adapter
uv run python scripts/03_merge.py --base Qwen/Qwen3-4B --adapter export/adapter --out export/merged
bash scripts/04_quantize.sh export/merged "export/centinela-${SIZE}.Q4_K_M.gguf"

# 5) Evaluate + tier gate (non-zero exit aborts before publish)
uv run python scripts/05_evaluate.py --base Qwen/Qwen3-4B --tuned export/merged --eval data/centinela/eval.jsonl --out eval/results/results.md

# Dry run: stop after eval without publishing. The eval table above is already in the
# job logs; echo the report file too so the full results survive the ephemeral job.
if [ "${STOP_BEFORE_PUBLISH:-false}" = "true" ]; then
  echo "=== STOP_BEFORE_PUBLISH=true — dry run complete, skipping publish ==="
  echo "--- eval/results/results.md ---"
  cat eval/results/results.md
  exit 0
fi

# 6) Publish model (artifacts + card + tag; metadata auto-resolved from stats.json + git)
uv run python scripts/06_push_to_hub.py \
  --size "$SIZE" --tag "$TAG" \
  --gguf "export/centinela-${SIZE}.Q4_K_M.gguf"

echo "pipeline complete: ${HF_ORG}/Centinela-Qwen3-${SIZE} @ ${TAG}"
