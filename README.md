# Astromesh Nebula

Open-model foundry of the Astromesh ecosystem — *where models are born*. First family:
**Centinela**, Spanish-first models for finance & back-office work in LATAM.

## Centinela-4B (v0.1)
A Spanish financial **sentiment** classifier (`positivo` / `neutral` / `negativo`), QLoRA
fine-tuned from `Qwen/Qwen3-4B`, released under Apache-2.0 with a deterministic output-validation
layer. Runs cheap, self-hosted, even on CPU via GGUF.

## Pipeline
`make dataset` → `make train` (HF Jobs) → `make merge` → `make quantize` → `make eval` →
`make publish`. See `docs` and the per-script docstrings.

## Quickstart (consumers)
```bash
ollama run hf.co/astromesh/Centinela-Qwen3-4B:Q4_K_M
```
