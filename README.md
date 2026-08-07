<div align="center">

# Astromesh Nebula

**The open-model foundry — where the models this ecosystem routes to are trained, gated and published.**

[![Astromesh · Models](https://img.shields.io/badge/astromesh-models-e879f9?labelColor=161b22)](https://monaccode.github.io/astromesh/#ecosystem)
[![Version](https://img.shields.io/badge/version-v0.1.0-e879f9?labelColor=161b22)](CHANGELOG.md)
[![Eval gate](https://github.com/monaccode/astromesh-nebula/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/monaccode/astromesh-nebula/actions/workflows/eval-gate.yml)
[![Preflight](https://github.com/monaccode/astromesh-nebula/actions/workflows/preflight.yml/badge.svg)](https://github.com/monaccode/astromesh-nebula/actions/workflows/preflight.yml)
[![Release](https://github.com/monaccode/astromesh-nebula/actions/workflows/release.yml/badge.svg)](https://github.com/monaccode/astromesh-nebula/actions/workflows/release.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Trained on HF Jobs](https://img.shields.io/badge/pipeline-HF%20Jobs-FFD21E?logo=huggingface&logoColor=black)](#pipeline)
[![Docs](https://img.shields.io/badge/docs-nebula-e879f9?labelColor=161b22)](https://monaccode.github.io/astromesh/nebula/introduction/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[Documentation](https://monaccode.github.io/astromesh/nebula/introduction/) ·
[The Foundry Pipeline](https://monaccode.github.io/astromesh/nebula/pipeline/) ·
[Catalog & GitOps](https://monaccode.github.io/astromesh/nebula/catalog-and-gitops/) ·
[How a model is made](https://monaccode.github.io/astromesh/nebula/centinela/)

</div>

---

Nebula is where models are born. It is not a serving layer and not a router: it turns a dataset
and a config into a **published, versioned, evaluated model**, and it refuses to publish one that
does not clear its gate. Everything downstream — the runtime's model router, Orbit's inference
stack, an Ollama pull — consumes its output, not its internals.

First family: **Centinela**, Spanish-first models for finance and back-office work in LATAM.

## Centinela-4B (v0.1)

A Spanish financial **sentiment** classifier (`positivo` / `neutral` / `negativo`), QLoRA
fine-tuned from `Qwen/Qwen3-4B`, Apache-2.0, with a deterministic output-validation layer that
constrains generation to the three labels. Runs cheap and self-hosted, even on CPU via GGUF.

> **Status:** the pipeline is complete and locally verifiable end to end; the public Hub release
> is not out. `catalog/centinela.yaml` still carries placeholder revision SHAs and zeroed eval
> metrics on purpose — the catalog is filled in by a real run, never by hand. Treat the
> `ollama run` line below as the shape of the consumer story, not as a live artifact.

```bash
ollama run hf.co/astromesh/Centinela-Qwen3-4B:Q4_K_M
```

## Pipeline

Each stage is a script under `scripts/`, driven by the Makefile, and each one is independently
runnable — a failed quantize does not cost you the training run.

```
make scout     → survey candidate base models
make dataset   → build and split the training corpus
make train     → QLoRA fine-tune (HF Jobs, or locally via WSL2 — see below)
make merge     → fold the adapter into the base weights
make quantize  → GGUF, Q4_K_M
make eval      → the gate: per-metric thresholds, macro-F1 and invalid-rate
make publish   → push weights + model card to the Hub
make catalog   → record the revision in the GitOps catalog
```

`make lock` compiles `catalog.lock.json`, which ships inside the wheel — that is what a consumer
resolves an alias like `prod` against, so a rollback is a catalog commit rather than a redeploy.

### Where the GPU is

GitHub runners have no GPU, so the heavy stages run elsewhere:

- **HF Jobs** (`make train`, and the `release` workflow's full pipeline) — billed, needs a
  PRO/Team plan and a token with write + create-repo on the `astromesh` org.
- **Local NVIDIA GPU via WSL2** (`scripts/02_train_local.py`, `scripts/wsl_train_*.sh`) — the
  fallback when HF Jobs is unavailable. Plain transformers + PEFT + bitsandbytes + TRL, no
  Unsloth: the 2026.6.x Unsloth build globally patches TRL and is incompatible with TRL 0.24
  (it injects an out-of-vocab `<EOS_TOKEN>` sentinel). Tuned for a 12 GB card — fp16 on Turing,
  `paged_adamw_8bit`, per-device batch 2 × grad-accum 8.

## Develop

```bash
uv sync
make test     # pytest, CPU only
make lint     # ruff
```

CI runs `eval-gate` (ruff + pytest, CPU) on every push and PR. `release` is manual
(`workflow_dispatch`): it authenticates to Hugging Face and launches the whole GPU pipeline as a
single HF Job. It accepts `stop_before_publish`, which runs train → merge → quantize → eval as a
dry run and prints the eval report to the job log without touching the public Hub — the intended
way to validate the first real GPU run before committing a `v0.1` anyone can pull.

Required repository configuration:

| Name | Kind | Value / scope |
|------|------|---------------|
| `HF_ORG` | Variable | `astromesh` |
| `HF_TOKEN` | Secret | HF fine-grained token: write + create-repo on the `astromesh` org |
| `GH_PIPELINE_TOKEN` | Secret | GitHub fine-grained PAT, read-only `contents` on this repo (so the HF Job can clone it) |

## Where Nebula sits

| | |
|---|---|
| **[astromesh](https://github.com/monaccode/astromesh)** | The runtime that routes to what Nebula publishes; its model catalog reads the compiled lock. |
| **[astromesh-orbit](https://github.com/monaccode/astromesh/tree/develop/astromesh-orbit)** | Provisions the inference stack a published model runs on. |
| **[The ecosystem map](https://monaccode.github.io/astromesh/#ecosystem)** | Every other component, and where each one fits. |
