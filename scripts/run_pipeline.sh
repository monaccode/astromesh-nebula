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

# 1) Build dataset (capture stats for the model card)
STATS="$(uv run python scripts/01_build_dataset.py --dataset "$DATASET" --out data/centinela)"
echo "$STATS"
N_TRAIN="$(printf '%s' "$STATS" | python -c 'import sys,json; print(json.load(sys.stdin)["n_train"])')"
N_EVAL="$(printf '%s' "$STATS" | python -c 'import sys,json; print(json.load(sys.stdin)["n_eval"])')"
DATASET_HASH="$(printf '%s' "$STATS" | python -c 'import sys,json; print(json.load(sys.stdin)["dataset_hash"])')"

# 2) Train QLoRA, 3) merge, 4) quantize
uv run python scripts/02_train.py --config "configs/train-${SIZE}.yaml" --train data/centinela/train.jsonl --out export/adapter
uv run python scripts/03_merge.py --base Qwen/Qwen3-4B --adapter export/adapter --out export/merged
bash scripts/04_quantize.sh export/merged "export/centinela-${SIZE}.Q4_K_M.gguf"

# 5) Evaluate + tier gate (non-zero exit aborts before publish)
uv run python scripts/05_evaluate.py --base Qwen/Qwen3-4B --tuned export/merged --eval data/centinela/eval.jsonl --out eval/results/results.md

# 6) Publish model (artifacts + card + tag)
uv run python scripts/06_push_to_hub.py \
  --size "$SIZE" --tag "$TAG" \
  --gguf "export/centinela-${SIZE}.Q4_K_M.gguf" \
  --dataset-hash "$DATASET_HASH" --commit "$(git rev-parse --short HEAD)" \
  --n-train "$N_TRAIN" --n-eval "$N_EVAL"

echo "pipeline complete: ${HF_ORG}/Centinela-Qwen3-${SIZE} @ ${TAG}"
