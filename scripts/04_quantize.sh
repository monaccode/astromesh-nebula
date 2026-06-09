#!/usr/bin/env bash
# Quantize merged bf16 weights to GGUF Q4_K_M using llama.cpp.
# Clones and builds llama.cpp (cmake) on first run if the quantize binary is missing.
# Usage: scripts/04_quantize.sh export/merged export/centinela-4b.Q4_K_M.gguf
set -euo pipefail

MERGED="${1:-export/merged}"
OUT="${2:-export/centinela-4b.Q4_K_M.gguf}"
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"
QUANT_BIN="$LLAMA_CPP/build/bin/llama-quantize"

# 1) Source
if [ ! -d "$LLAMA_CPP" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP"
fi

# 2) Python deps for the HF -> GGUF converter
pip install -r "$LLAMA_CPP/requirements.txt"

# 3) Build the quantize binary if missing (idempotent on re-runs)
if [ ! -x "$QUANT_BIN" ]; then
  cmake -B "$LLAMA_CPP/build" "$LLAMA_CPP"
  cmake --build "$LLAMA_CPP/build" --config Release --target llama-quantize -j
fi

# 4) Convert HF -> f16 GGUF, then quantize -> Q4_K_M
python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outfile export/centinela-4b.f16.gguf --outtype f16
"$QUANT_BIN" export/centinela-4b.f16.gguf "$OUT" Q4_K_M

echo "wrote $OUT"
