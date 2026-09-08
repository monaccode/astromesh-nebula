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

# SMOKE MODE — set MAX_STEPS to a small positive number (e.g. 10) to cap optimizer
# steps so the whole chain runs in minutes instead of hours. The point is to prove the
# plumbing (dataset -> train -> merge -> quantize -> eval) before paying for a real run;
# the model it produces is deliberately useless. Two consequences follow from that:
# the eval gate is EXPECTED to fail, so its exit code must not be read as a broken
# pipeline, and publishing is force-disabled regardless of STOP_BEFORE_PUBLISH.
MAX_STEPS="${MAX_STEPS:--1}"
SMOKE_EVAL_LIMIT="${SMOKE_EVAL_LIMIT:-32}"
SMOKE=false
case "$MAX_STEPS" in
  ''|-1) ;;
  *[!0-9]*)
    echo "MAX_STEPS must be a positive integer or -1; got '$MAX_STEPS'" >&2
    exit 2
    ;;
  0) ;;
  *) SMOKE=true ;;
esac
if [ "$SMOKE" = "true" ]; then
  echo "=== SMOKE MODE: max_steps=$MAX_STEPS, eval capped at ${SMOKE_EVAL_LIMIT} examples, publish disabled ==="
fi

pip install uv
uv sync --extra eval

# 1) Build dataset (writes data/centinela/stats.json for downstream steps)
uv run python scripts/01_build_dataset.py --dataset "$DATASET" --out data/centinela

# 2) Train QLoRA, 3) merge, 4) quantize
uv run python scripts/02_train.py --config "configs/train-${SIZE}.yaml" --train data/centinela/train.jsonl --out export/adapter --max-steps "$MAX_STEPS"
uv run python scripts/03_merge.py --base Qwen/Qwen3-4B --adapter export/adapter --out export/merged
bash scripts/04_quantize.sh export/merged "export/centinela-${SIZE}.Q4_K_M.gguf"

# 5) Evaluate + tier gate (non-zero exit aborts before publish)
EVAL_ARGS=()
if [ "$SMOKE" = "true" ]; then
  EVAL_ARGS+=(--limit "$SMOKE_EVAL_LIMIT")
fi
# The gate's exit code is captured rather than allowed to kill the script, so that a
# smoke run can report it as expected instead of dying at step 5.
set +e
# ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": under `set -u` an EMPTY array is an
# unbound variable in bash < 4.4, which would break the full (non-smoke) run.
uv run python scripts/05_evaluate.py --base Qwen/Qwen3-4B --tuned export/merged --eval data/centinela/eval.jsonl --out eval/results/results.md ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
gate_rc=$?
set -e

if [ "$SMOKE" = "true" ]; then
  echo "=== SMOKE: eval gate exit code ${gate_rc} (a FAIL here is expected — the model saw ${MAX_STEPS} steps) ==="
  cat eval/results/results.md || true
  echo "=== SMOKE complete: the full chain ran. Publishing skipped by design. ==="
  exit 0
fi

if [ "$gate_rc" -ne 0 ]; then
  echo "eval gate FAILED (exit ${gate_rc}) — aborting before publish"
  exit "$gate_rc"
fi

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
