"""Render the Hugging Face model card (README.md) for a Centinela model."""

_TEMPLATE = """---
license: apache-2.0
base_model: {base_model}
language: [es, en]
pipeline_tag: text-generation
library_name: peft
tags: [finance, sentiment, spanish, latam, qlora, astromesh, nebula, centinela]
datasets: [{dataset_id}]
---

# {repo_id}

Spanish-first financial **sentiment** classifier (positivo / neutral / negativo).
First model of the **Centinela** family, born in the **Astromesh Nebula** open-model foundry.
Designed for low-cost, self-hosted agents.

## Intended use
Classify the sentiment of Spanish financial text into `positivo`, `neutral`, or `negativo`.
Outputs are constrained to that closed set by a deterministic validation layer.
NOT a substitute for a licensed accountant or financial advisor.

## How to use
transformers / Ollama (GGUF). See the repo quickstart.

## Training
- Method: QLoRA (4-bit) on `{base_model}`
- Data: {n_train} Spanish examples (eval held out: {n_eval}) from `{dataset_id}`

## Evaluation
{eval_table}

## Reproducibility
base_model: {base_model_id} · dataset_hash: {dataset_hash} · config: {config_path} · commit: {commit}

## Limitations & risks
Spanish-optimized; may misclassify domain-shifted text. Always validate before acting on outputs.
Not for autonomous high-stakes financial decisions.

## License
Apache-2.0, inheriting the base model license.
"""


def render_model_card(
    *,
    repo_id: str,
    base_model: str,
    dataset_id: str,
    eval_table: str,
    n_train: int,
    n_eval: int,
    base_model_id: str,
    dataset_hash: str,
    config_path: str,
    commit: str,
) -> str:
    return _TEMPLATE.format(
        repo_id=repo_id,
        base_model=base_model,
        dataset_id=dataset_id,
        eval_table=eval_table,
        n_train=n_train,
        n_eval=n_eval,
        base_model_id=base_model_id,
        dataset_hash=dataset_hash,
        config_path=config_path,
        commit=commit,
    )
