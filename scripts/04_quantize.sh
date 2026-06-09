#!/usr/bin/env bash
# Quantize merged bf16 weights to GGUF Q4_K_M using llama.cpp.
# Usage: scripts/04_quantize.sh export/merged export/centinela-4b.Q4_K_M.gguf
set -euo pipefail

MERGED="${1:-export/merged}"
OUT="${2:-export/centinela-4b.Q4_K_M.gguf}"
LLAMA_CPP="${LLAMA_CPP:-./llama.cpp}"

if [ ! -d "$LLAMA_CPP" ]; then
  git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP"
  pip install -r "$LLAMA_CPP/requirements.txt"
fi

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outfile export/centinela-4b.f16.gguf --outtype f16
"$LLAMA_CPP/llama-quantize" export/centinela-4b.f16.gguf "$OUT" Q4_K_M

echo "wrote $OUT"
